from abc import ABC, abstractmethod
from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp

from .common import Mutations
from ....hrm.ops import (
    get_accepting_state_id,
    get_num_rm_states,
    get_max_num_states_per_machine,
    get_max_num_machines,
    hrm_like,
)
from ....hrm.types import HRM


def build_hrm_mutator(mutation_id: Mutations, alphabet_size: int, use_sparse_reward: bool, max_num_args: int):
    match mutation_id:
        case Mutations.SWITCH_PROP:
            cls = SwitchPropMutator
        case Mutations.ADD_TRANSITION:
            cls = AddTransitionMutator
        case Mutations.RM_TRANSITION:
            cls = RemoveTransitionMutator
    return cls(alphabet_size, use_sparse_reward, max_num_args)


class HRMMutator(ABC):
    """
    Base class for HRM mutators.
    """
    def __init__(self, alphabet_size: int, use_sparse_reward: bool, max_num_args: int = 2):
        self.alphabet_size = alphabet_size
        self.use_sparse_reward = use_sparse_reward
        self.max_num_args = max_num_args

    @abstractmethod
    def is_applicable(self, hrm: HRM) -> bool:
        raise NotImplementedError

    @abstractmethod
    def apply(self, rng: chex.PRNGKey, hrm: HRM) -> Tuple[HRM, chex.Array, chex.Array]:
        raise NotImplementedError


class SwitchPropMutator(HRMMutator):
    """
    Replaces one of the propositions in a *sequential* RM for another
    chosen at random.
    """
    def __init__(self, alphabet_size: int, use_sparse_reward: bool, max_num_args: int = 2):
        assert max_num_args >= 2
        super().__init__(alphabet_size, use_sparse_reward, max_num_args)

    def is_applicable(self, hrm: HRM) -> bool:
        return True

    def apply(self, rng: chex.PRNGKey, hrm: HRM) -> Tuple[HRM, chex.Array, chex.Array]:
        state_rng, prop_rng = jax.random.split(rng)

        max_num_states = get_max_num_states_per_machine(hrm)
        num_states = get_num_rm_states(hrm, rm_id=0)
        mask = jnp.arange(max_num_states) < (num_states - 1)

        src_id = jax.random.choice(state_rng, max_num_states, p=mask / mask.sum())
        dst_id = jax.lax.select(
            src_id == num_states - 2,
            get_accepting_state_id(hrm),
            src_id + 1
        )

        new_hrm = jax.tree_util.tree_map(lambda x: jnp.copy(x), hrm)
        new_prop = jax.random.choice(prop_rng, self.alphabet_size)
        new_hrm.formulas = new_hrm.formulas.at[0, src_id, dst_id, 0, 0].set(
            new_prop + 1  # formulas are indexed with literals, hence +1
        )

        return (
            new_hrm,
            Mutations.SWITCH_PROP,
            jnp.pad(jnp.array((src_id, new_prop), dtype=jnp.int32), pad_width=(0, self.max_num_args - 2))
        )


class AddTransitionMutator(HRMMutator):
    """
    Adds a transition with a random proposition.
    """
    def __init__(self, alphabet_size: int, use_sparse_reward: bool, max_num_args: int = 2):
        assert max_num_args >= 2
        super().__init__(alphabet_size, use_sparse_reward, max_num_args)

    def is_applicable(self, hrm: HRM) -> bool:
        return get_num_rm_states(hrm, rm_id=0) < get_max_num_states_per_machine(hrm)

    def apply(self, rng: chex.PRNGKey, hrm: HRM) -> Tuple[HRM, chex.Array, chex.Array]:
        index_rng, prop_rng = jax.random.split(rng)

        max_num_states = get_max_num_states_per_machine(hrm)
        mask = jnp.arange(max_num_states) < get_num_rm_states(hrm, rm_id=0)
        index = jax.random.choice(index_rng, max_num_states, p=mask / jnp.sum(mask))

        literal = jax.random.choice(prop_rng, self.alphabet_size) + 1

        return (
            self._add_transition(hrm, index, literal),
            Mutations.ADD_TRANSITION,
            jnp.pad(jnp.array((index, literal), dtype=jnp.int32), pad_width=(0, self.max_num_args - 2))
        )

    def _add_transition(self, hrm: HRM, index: int, literal: int) -> HRM:
        def _f(carry, _):
            calls, formulas, num_literals, rewards, src_id, dst_id = carry

            def _cpy_preceeding():
                # Copies preceeding transitions to the insertion place
                cond = (src_id + 1) == (get_num_rm_states(hrm, rm_id=0) - 1)  # next will be connection to accepting state
                next_src = jax.lax.select(cond, get_accepting_state_id(hrm), src_id + 1)
                next_dst = dst_id + 1

                return (
                    calls.at[0, dst_id, next_dst].set(hrm.calls[0, src_id, next_src]),
                    formulas.at[0, dst_id, next_dst].set(hrm.formulas[0, src_id, next_src]),
                    num_literals.at[0, dst_id, next_dst].set(hrm.num_literals[0, src_id, next_src]),
                    rewards.at[0, dst_id, next_dst].set(jnp.logical_not(self.use_sparse_reward) * hrm.rewards[0, src_id, next_src]),
                    next_src,
                    next_dst
                )

            def _insert():
                # Performs insertion at the intended place
                cond = (dst_id + 1) == get_num_rm_states(hrm, rm_id=0)
                next_dst = jax.lax.select(cond, get_accepting_state_id(hrm), dst_id + 1)

                return (
                    calls.at[0, dst_id, next_dst, 0].set(get_max_num_machines(hrm)),
                    formulas.at[0, dst_id, next_dst, 0, 0].set(literal),
                    num_literals.at[0, dst_id, next_dst, 0].set(1),
                    rewards.at[0, dst_id, next_dst].set(jax.lax.select(cond, 1, 0)),
                    src_id,
                    next_dst
                )

            def _cpy_succeeding():
                # Copies the succeeding places to the insertion place
                cond = (src_id + 1) == (get_num_rm_states(hrm, rm_id=0) - 1)  # next will be connection to accepting state
                next_src = jax.lax.select(cond, get_accepting_state_id(hrm), src_id + 1)
                next_dst = jax.lax.select(cond, get_accepting_state_id(hrm), dst_id + 1)

                return (
                    calls.at[0, dst_id, next_dst, 0].set(hrm.calls[0, src_id, next_src, 0]),
                    formulas.at[0, dst_id, next_dst, 0].set(hrm.formulas[0, src_id, next_src, 0]),
                    num_literals.at[0, dst_id, next_dst, 0].set(hrm.num_literals[0, src_id, next_src, 0]),
                    rewards.at[0, dst_id, next_dst].set(hrm.rewards[0, src_id, next_src]),
                    next_src,
                    next_dst
                )

            branch_id = (
                (dst_id == index) +   # insertion place
                2 * (index < dst_id) * (dst_id < get_accepting_state_id(hrm)) +  # insertion made, copying if remaining states to get to accepting state
                3 * (dst_id == get_accepting_state_id(hrm))  # accepting state reached, nothing to do
            )
            return jax.lax.switch(
                 branch_id,
                [
                    _cpy_preceeding,
                    _insert,
                    _cpy_succeeding,
                    lambda: (calls, formulas, num_literals, rewards, src_id, dst_id)
                ]
            ), None

        # Go through the original HRM copying its calls and formulas while inserting the new transition
        new_hrm = hrm_like(hrm)
        (calls, formulas, num_literals, rewards, _, _), _ = jax.lax.scan(
            _f,
            (new_hrm.calls, new_hrm.formulas, new_hrm.num_literals, new_hrm.rewards, 0, 0),
            length=get_max_num_states_per_machine(hrm)
        )

        new_hrm.calls = calls
        new_hrm.formulas = formulas
        new_hrm.rewards = rewards
        new_hrm.num_literals = num_literals
        return new_hrm


class RemoveTransitionMutator(HRMMutator):
    """
    Removes the final transition to the accepting state.
    """
    def __init__(self, alphabet_size: int, use_sparse_reward: bool, max_num_args: int = 2):
        assert max_num_args >= 2
        super().__init__(alphabet_size, use_sparse_reward, max_num_args)

    def is_applicable(self, hrm: HRM) -> bool:
        return get_num_rm_states(hrm, rm_id=0) > 2

    def apply(self, rng: chex.PRNGKey, hrm: HRM) -> Tuple[HRM, chex.Array, chex.Array]:
        max_num_states = get_max_num_states_per_machine(hrm)
        mask = jnp.arange(max_num_states) < (get_num_rm_states(hrm, rm_id=0) - 1)
        index = jax.random.choice(rng, max_num_states, p=mask / jnp.sum(mask))
        return (
            self._rm_transition(hrm, index),
            Mutations.RM_TRANSITION,
            jnp.pad(jnp.array((index,), dtype=jnp.int32), pad_width=(0, self.max_num_args - 1))
        )

    def _rm_transition(self, hrm: HRM, index: int) -> HRM:
        num_rm_states = get_num_rm_states(hrm, rm_id=0)

        def _f(carry, _):
            calls, formulas, num_literals, rewards, src_id, dst_id = carry

            def _cpy_succeeding():
                cond = (src_id + 1) == (num_rm_states - 1)  # next will be connection to accepting state
                next_src = jax.lax.select(cond, get_accepting_state_id(hrm), src_id + 1)
                next_dst = jax.lax.select(cond, get_accepting_state_id(hrm), dst_id + 1)
                reward = jax.lax.select(cond, 1, jnp.logical_not(self.use_sparse_reward).astype(jnp.int32))

                return (
                    calls.at[0, dst_id, next_dst, 0].set(hrm.calls[0, src_id, next_src, 0]),
                    formulas.at[0, dst_id, next_dst, 0].set(hrm.formulas[0, src_id, next_src, 0]),
                    num_literals.at[0, dst_id, next_dst, 0].set(hrm.num_literals[0, src_id, next_src, 0]),
                    rewards.at[0, dst_id, next_dst].set(reward),
                    src_id - 1,
                    dst_id - 1
                )

            def _remove():
                return calls, formulas, num_literals, rewards, src_id - 1, dst_id

            def _cpy_preceeding():
                cond = (dst_id + 1) == (get_num_rm_states(hrm, rm_id=0) - 2)  # next will be connection to accepting state
                next_src = src_id + 1
                next_dst = jax.lax.select(cond, get_accepting_state_id(hrm), dst_id + 1)
                reward = jax.lax.select(cond, 1, jnp.logical_not(self.use_sparse_reward).astype(jnp.int32))

                return (
                    calls.at[0, dst_id, next_dst].set(hrm.calls[0, src_id, next_src]),
                    formulas.at[0, dst_id, next_dst].set(hrm.formulas[0, src_id, next_src]),
                    num_literals.at[0, dst_id, next_dst].set(hrm.num_literals[0, src_id, next_src]),
                    rewards.at[0, dst_id, next_dst].set(reward),
                    src_id - 1,
                    dst_id - 1
                )

            branch_id = (
                (src_id == index) +  # remove (skip)
                2 * (src_id < index) * (src_id >= 0) +
                3 * (src_id < 0)
            )
            return jax.lax.switch(
                branch_id,
                [
                    _cpy_succeeding,
                    _remove,
                    _cpy_preceeding,
                    lambda: (calls, formulas, num_literals, rewards, src_id, dst_id)
                ]
            ), None

        new_hrm = hrm_like(hrm)
        (calls, formulas, num_literals, rewards, _, _), _ = jax.lax.scan(
            _f,
            (new_hrm.calls, new_hrm.formulas, new_hrm.num_literals, new_hrm.rewards, num_rm_states - 2, num_rm_states - 3),
            length=get_max_num_states_per_machine(hrm)
        )

        new_hrm.calls = calls
        new_hrm.formulas = formulas
        new_hrm.rewards = rewards
        new_hrm.num_literals = num_literals
        return new_hrm


class HRMSequenceMutator(HRMMutator):
    def __init__(
        self,
        num_edits: int,
        use_add_rm_transitions: bool,
        alphabet_size: int,
        use_sparse_reward: bool,
        max_num_args: int = 2,
        max_num_edits: Optional[int] = None,
    ):
        super().__init__(alphabet_size, use_sparse_reward, max_num_args)

        self.num_edits = num_edits
        self.max_num_edits = max_num_edits

        mutation_ids = [Mutations.SWITCH_PROP]
        if use_add_rm_transitions:
            mutation_ids.extend([Mutations.ADD_TRANSITION, Mutations.RM_TRANSITION])

        self.mutators = [
            build_hrm_mutator(mutation_id, alphabet_size, use_sparse_reward, max_num_args)
            for mutation_id in mutation_ids
        ]

    def is_applicable(self, hrm: HRM) -> bool:
        return any([m.is_applicable(hrm) for m in self.mutators])

    def apply(self, rng: chex.PRNGKey, hrm: HRM) -> Tuple[HRM, chex.Array, chex.Array]:
        def _mutate_aux(h: HRM, _rng: chex.PRNGKey):
            idx_rng, mutate_rng = jax.random.split(_rng)
            cond_mask = jnp.array([f.is_applicable(h) for f in self.mutators])
            idx = jax.random.choice(idx_rng, len(self.mutators), p=cond_mask / cond_mask.sum())
            next_h, mutation_id, mutation_args = jax.lax.switch(
                idx, [f.apply for f in self.mutators], mutate_rng, h
            )
            return next_h, (mutation_id, mutation_args)

        next_hrm, (out_ids, out_args) = jax.lax.scan(
            _mutate_aux,
            init=hrm,
            xs=jax.random.split(rng, self.num_edits)
        )

        if self.max_num_edits:
            out_ids = jnp.pad(out_ids, pad_width=(0, self.max_num_edits - self.num_edits), constant_values=-1)
            out_args = jnp.pad(out_args, pad_width=((0, self.max_num_edits - self.num_edits), (0, 0)))

        return next_hrm, out_ids, out_args
