from abc import ABC, abstractmethod
from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from xminigrid.types import State

from .common import Mutations
from ..level import XMinigridLevel
from ....hrm.ops import is_accepting_state, is_initial_state, split_predecessors, split_successors
from ....hrm.types import HRM, HRMState


def build_hindsight_mutator(mutation_id: Mutations, max_num_args: int):
    match mutation_id:
        case Mutations.HINDSIGHT_LVL_ONLY:
            cls = HindsightLevelMutator
        case Mutations.HINDSIGHT_PRED:
            cls = HindsightPredMutator
        case Mutations.HINDSIGHT_SUCC:
            cls = HindsightSuccMutator
    return cls(max_num_args)


class HindsightMutator(ABC):
    def __init__(self, max_num_args: int = 1):
        self.max_num_args = max_num_args

    @abstractmethod
    def is_applicable(self, hrm: HRM, hrm_state: HRMState) -> bool:
        raise NotImplementedError

    @abstractmethod
    def apply(
        self, rng: chex.PRNGKey, level: XMinigridLevel, hrm: HRM, hrm_state: HRMState, env_state: State
    ) -> Tuple[XMinigridLevel, HRM, chex.Array, chex.Array]:
        raise NotImplementedError

    def _is_hrm_partitionable(self, hrm: HRM, hrm_state: HRMState) -> bool:
        return jnp.logical_and(
            jnp.logical_not(is_initial_state(hrm_state.state_id)),
            jnp.logical_not(is_accepting_state(hrm, hrm_state.state_id))
        )


class HindsightLevelMutator(HindsightMutator):
    def is_applicable(self, hrm: HRM, hrm_state: HRMState) -> bool:
        return True

    def apply(
        self, rng: chex.PRNGKey, level: XMinigridLevel, hrm: HRM, hrm_state: HRMState, env_state: State
    ) -> Tuple[XMinigridLevel, HRM, chex.Array, chex.Array]:
        args = jnp.array([hrm_state.state_id], jnp.int32)
        return (
            XMinigridLevel.from_env_state(env_state, level.height, level.width),
            jax.tree_util.tree_map(lambda x: jnp.copy(x), hrm),
            Mutations.HINDSIGHT_LVL_ONLY,
            jnp.pad(args, pad_width=(0, self.max_num_args - 1), constant_values=-1)
        )


class HindsightPredMutator(HindsightMutator):
    def is_applicable(self, hrm: HRM, hrm_state: HRMState) -> bool:
        return self._is_hrm_partitionable(hrm, hrm_state)

    def apply(
        self, rng: chex.PRNGKey, level: XMinigridLevel, hrm: HRM, hrm_state: HRMState, env_state: State
    ) -> Tuple[XMinigridLevel, HRM, chex.Array, chex.Array]:
        args = jnp.array([hrm_state.state_id], jnp.int32)
        return (
            level,
            split_predecessors(hrm, hrm_state),
            Mutations.HINDSIGHT_PRED,
            jnp.pad(args, pad_width=(0, self.max_num_args - 1), constant_values=-1)
        )


class HindsightSuccMutator(HindsightMutator):
    def is_applicable(self, hrm: HRM, hrm_state: HRMState) -> bool:
        return self._is_hrm_partitionable(hrm, hrm_state)

    def apply(
        self, rng: chex.PRNGKey, level: XMinigridLevel, hrm: HRM, hrm_state: HRMState, env_state: State
    ) -> Tuple[XMinigridLevel, HRM, chex.Array, chex.Array]:
        args = jnp.array([hrm_state.state_id], jnp.int32)
        return (
            XMinigridLevel.from_env_state(env_state, level.height, level.width),
            split_successors(hrm, hrm_state),
            Mutations.HINDSIGHT_SUCC,
            jnp.pad(args, pad_width=(0, self.max_num_args - 1), constant_values=-1)
        )


class HindsightAggMutator(HindsightMutator):
    """
    Produces a mutation of the HRM-level pair based on the HRM state
    and the environment state. The HRM is assumed to be flat and with
    sequentially increasing state indices.

    The mutation consists of returning at random:
      - the predecessor part of the HRM [i.e., the state sequence before the
        current HRM state] and the level we started from, or
      - the successor part [i.e., the state sequence after the current HRM state]
        and the level with the grid config we left.

    If the state is the initial state or the accepting state, the returned level
    depends on the last environment state, and the HRM is the same one.
    """
    def __init__(
        self,
        use_level_mutation: bool = False,
        max_num_args: int = 1,
        max_num_edits: Optional[int] = None,
    ):
        super().__init__(max_num_args)
        self.max_num_edits = max_num_edits

        self.use_level_mutation = use_level_mutation
        self.level_mutator = build_hindsight_mutator(Mutations.HINDSIGHT_LVL_ONLY, max_num_args)
        self.mutators = [
            build_hindsight_mutator(Mutations.HINDSIGHT_PRED, max_num_args),
            build_hindsight_mutator(Mutations.HINDSIGHT_SUCC, max_num_args),
        ]

    def is_applicable(self, hrm: HRM, hrm_state: HRMState) -> bool:
        return jnp.logical_or(
            self._is_pred_succ_applicable(hrm, hrm_state),
            jnp.logical_and(self.use_level_mutation, self.level_mutator.is_applicable(hrm, hrm_state))
        )

    def apply(
        self, rng: chex.PRNGKey, level: XMinigridLevel, hrm: HRM, hrm_state: HRMState, env_state: State
    ) -> Tuple[XMinigridLevel, HRM, chex.Array, chex.Array]:
        choice_rng, mutate_rng = jax.random.split(rng)
        idx = self._is_pred_succ_applicable(hrm, hrm_state) * jax.random.choice(choice_rng, jnp.array([1, 2]))
        next_level, next_hrm, mutation_id, mutation_args = jax.lax.switch(
            idx, [
                self.level_mutator.apply,
                *[m.apply for m in self.mutators]
            ], mutate_rng, level, hrm, hrm_state, env_state
        )

        mutation_id = mutation_id[jnp.newaxis, ...]
        mutation_args = mutation_args[jnp.newaxis, ...]
        if self.max_num_edits:
            mutation_id = jnp.pad(mutation_id, pad_width=(0, self.max_num_edits - 1), constant_values=-1)
            mutation_args = jnp.repeat(mutation_args, self.max_num_edits, axis=0)

        return next_level, next_hrm, mutation_id, mutation_args

    def _is_pred_succ_applicable(self, hrm: HRM, hrm_state: HRMState) -> bool:
        return jnp.any(jnp.array([mutator.is_applicable(hrm, hrm_state) for mutator in self.mutators]))
