from abc import abstractmethod, ABC
from typing import Tuple

import chex
from chex import dataclass
import jax
from jax import numpy as jnp
from xminigrid.types import State

from .common import Mutations
from .hindsight import HindsightAggMutator, build_hindsight_mutator, HindsightMutator
from .hrm import build_hrm_mutator, HRMMutator, HRMSequenceMutator
from .level import build_level_mutator, LevelMutator, LevelSequenceMutator
from ..level import XMinigridLevel
from ..types import XMinigridEnvParams
from ....hrm.types import HRM, HRMState


class JointMutator(ABC):
    @abstractmethod
    def is_applicable(self, level: XMinigridLevel, hrm: HRM, hrm_state: HRMState) -> bool:
        raise NotImplementedError

    @abstractmethod
    def apply(
        self, rng: chex.PRNGKey, level: XMinigridLevel, hrm: HRM, hrm_state: HRMState, env_state: State
    ) -> Tuple[XMinigridLevel, HRM, chex.Array, chex.Array]:
        raise NotImplementedError


class HRMMutatorJointWrapper(JointMutator):
    def __init__(self, hrm_mutator: HRMMutator):
        self.hrm_mutator = hrm_mutator

    def is_applicable(self, level: XMinigridLevel, hrm: HRM, hrm_state: HRMState) -> bool:
        return self.hrm_mutator.is_applicable(hrm)

    def apply(
        self, rng: chex.PRNGKey, level: XMinigridLevel, hrm: HRM, hrm_state: HRMState, env_state: State
    ) -> Tuple[XMinigridLevel, HRM, chex.Array, chex.Array]:
        next_hrm, mutation_id, mutation_args = self.hrm_mutator.apply(rng, hrm)
        return level, next_hrm, mutation_id, mutation_args


class LevelMutatorJointWrapper(JointMutator):
    def __init__(self, level_mutator: LevelMutator):
        self.level_mutator = level_mutator

    def is_applicable(self, level: XMinigridLevel, hrm: HRM, hrm_state: HRMState) -> bool:
        return self.level_mutator.is_applicable(level)

    def apply(
        self, rng: chex.PRNGKey, level: XMinigridLevel, hrm: HRM, hrm_state: HRMState, env_state: State
    ) -> Tuple[XMinigridLevel, HRM, chex.Array, chex.Array]:
        next_level, mutation_id, mutation_args = self.level_mutator.apply(rng, level)
        return next_level, jax.tree_util.tree_map(lambda x: jnp.copy(x), hrm), mutation_id, mutation_args


class HindsightMutatorJointWrapper(JointMutator):
    def __init__(self, hindsight_mutator: HindsightMutator):
        self.hindsight_mutator = hindsight_mutator

    def is_applicable(self, level: XMinigridLevel, hrm: HRM, hrm_state: HRMState) -> bool:
        return self.hindsight_mutator.is_applicable(hrm, hrm_state)

    def apply(
        self, rng: chex.PRNGKey, level: XMinigridLevel, hrm: HRM, hrm_state: HRMState, env_state: State
    ) -> Tuple[XMinigridLevel, HRM, chex.Array, chex.Array]:
        return self.hindsight_mutator.apply(rng, level, hrm, hrm_state, env_state)


class InterleavedLevelHRMMutator(JointMutator):
    """
    Applies a number of mutations sampled uniformly from a set
    containing HRM, level and hinsight mutations. If hindsight
    mutations are usable, at most one of them will be applied.
    """
    @dataclass
    class LevelConfig:
        enabled: bool
        use_move_agent: bool
        use_add_rm_objs: bool
        use_add_rm_rooms: bool

    @dataclass
    class HRMConfig:
        enabled: bool
        use_add_rm_transitions: bool

    @dataclass
    class HindsightConfig:
        enabled: bool

    def __init__(
        self,
        min_edits: int,
        max_edits: int,
        level_cfg: LevelConfig,
        hrm_cfg: HRMConfig,
        hindsight_cfg: HindsightConfig,
        env_params: XMinigridEnvParams,
        alphabet_size: int,
        use_sparse_reward: bool,
        max_num_args: int = 2,
    ):
        self.min_edits = min_edits
        self.max_edits = max_edits
        self.max_num_args = max_num_args

        # Determine the mutations
        if level_cfg.enabled:
            level_mutations = [Mutations.MOVE_NON_DOOR_OBJ, Mutations.REPLACE_DOOR, Mutations.REPLACE_NON_DOOR]
            if level_cfg.use_move_agent:
                level_mutations.append(Mutations.MOVE_AGENT)
            if level_cfg.use_add_rm_objs:
                level_mutations.extend([Mutations.ADD_NON_DOOR_OBJ, Mutations.RM_NON_DOOR_OBJ])
            if level_cfg.use_add_rm_rooms:
                level_mutations.extend([Mutations.ADD_ROOMS, Mutations.RM_ROOMS])
        else:
            level_mutations = []

        if hrm_cfg.enabled:
            hrm_mutations = [Mutations.SWITCH_PROP]
            if hrm_cfg.use_add_rm_transitions:
                hrm_mutations.extend([Mutations.ADD_TRANSITION, Mutations.RM_TRANSITION])
        else:
            hrm_mutations = []

        hindsight_mutations = []
        if hindsight_cfg.enabled:
            hindsight_mutations = [Mutations.HINDSIGHT_PRED, Mutations.HINDSIGHT_SUCC]

        # Build the mutators and wrap them accordingly to have the same signature
        self.level_mutators = [
            LevelMutatorJointWrapper(build_level_mutator(mutation_id, env_params, max_num_args))
            for mutation_id in level_mutations
        ]

        self.hrm_mutators = [
            HRMMutatorJointWrapper(build_hrm_mutator(mutation_id, alphabet_size, use_sparse_reward, max_num_args))
            for mutation_id in hrm_mutations
        ]

        self.hindsight_mutators = [
            HindsightMutatorJointWrapper(build_hindsight_mutator(mutation_id, max_num_args))
            for mutation_id in hindsight_mutations
        ]

    def is_applicable(self, level: XMinigridLevel, hrm: HRM, hrm_state: HRMState) -> bool:
        return any([
            m.is_applicable(level, hrm, hrm_state)
            for m in [*self.level_mutators, *self.hrm_mutators, *self.hindsight_mutators]
        ])

    def apply(
        self, rng: chex.PRNGKey, level: XMinigridLevel, hrm: HRM, hrm_state: HRMState, env_state: State
    ) -> Tuple[XMinigridLevel, HRM, chex.Array, chex.Array]:
        num_edits = jax.random.uniform(rng, minval=self.min_edits, maxval=self.max_edits + 1)

        def _mutate_aux(carry: Tuple[XMinigridLevel, HRM], xs: Tuple[chex.PRNGKey, chex.Array]):
            _level, _hrm = carry
            _rng, _edit_id = xs

            def _f():
                idx_rng, mutate_rng = jax.random.split(_rng)

                cond_mask = jnp.concat((
                    jnp.array([f.is_applicable(_level, _hrm, hrm_state) for f in self.level_mutators]),
                    jnp.array([f.is_applicable(_level, _hrm, hrm_state) for f in self.hrm_mutators]),
                    jnp.logical_and(
                        jnp.equal(_edit_id, 0),  # hindsight only applicable as first edit
                        jnp.array([f.is_applicable(_level, _hrm, hrm_state) for f in self.hindsight_mutators]),
                    )
                ))

                idx = jax.random.choice(idx_rng, len(cond_mask), p=cond_mask / cond_mask.sum())
                next_level, next_hrm, mutation_id, mutation_args = jax.lax.switch(
                    idx,
                    [
                        *[f.apply for f in self.level_mutators],
                        *[f.apply for f in self.hrm_mutators],
                        *[f.apply for f in self.hindsight_mutators],
                    ],
                    mutate_rng, _level, _hrm, hrm_state, env_state
                )
                return (next_level, next_hrm), (mutation_id, mutation_args)

            return jax.lax.cond(
                pred=_edit_id < num_edits,
                true_fun=_f,
                false_fun=lambda: ((_level, _hrm), (-1, -jnp.ones((self.max_num_args,), dtype=jnp.int32)))
            )

        (next_level, next_hrm), (out_ids, out_args) = jax.lax.scan(
            _mutate_aux,
            init=(level, hrm),
            xs=(jnp.array(jax.random.split(rng, self.max_edits)), jnp.arange(self.max_edits))
        )

        return next_level, next_hrm, out_ids, out_args


class ExclusiveMutator(JointMutator):
    """
    Applies a number of mutations of a specific type (hindsight,
    level, HRM).
    """
    def __init__(
        self,
        num_level_edits: int,
        num_hrm_edits: int,
        use_move_agent: bool,
        use_add_rm_objs: bool,
        use_add_rm_rooms: bool,
        env_params: XMinigridEnvParams,
        use_add_rm_transitions: bool,
        alphabet_size: int,
        use_sparse_reward: bool,
        use_hindsight_edit: bool = False,
        max_num_args: int = 2,
    ):
        self.use_hindsight_edit = use_hindsight_edit

        max_num_edits = max(num_level_edits, num_hrm_edits)
        self.hindsight_mutator = HindsightAggMutator(use_level_mutation=False, max_num_args=max_num_args, max_num_edits=max_num_edits)
        self.level_mutator = LevelMutatorJointWrapper(LevelSequenceMutator(
            num_level_edits, use_move_agent, use_add_rm_objs, use_add_rm_rooms, env_params, max_num_args, max_num_edits
        ))
        self.hrm_mutator = HRMMutatorJointWrapper(HRMSequenceMutator(
            num_hrm_edits, use_add_rm_transitions, alphabet_size, use_sparse_reward, max_num_args, max_num_edits
        ))

    def is_applicable(self, level: XMinigridLevel, hrm: HRM, hrm_state: HRMState) -> bool:
        return jnp.logical_or(
            jnp.logical_and(self.use_hindsight_edit, self.hindsight_mutator.is_applicable(hrm, hrm_state)),
            self.level_mutator.is_applicable(level, hrm, hrm_state),
            self.hrm_mutator.is_applicable(level, hrm, hrm_state)
        )

    def apply(
        self, rng: chex.PRNGKey, level: XMinigridLevel, hrm: HRM, hrm_state: HRMState, env_state: State
    ) -> Tuple[XMinigridLevel, HRM, chex.Array, chex.Array]:
        choice_rng, sample_rng = jax.random.split(rng)

        mask = jnp.array([
            jnp.logical_and(self.use_hindsight_edit, self.hindsight_mutator.is_applicable(hrm, hrm_state)),
            self.level_mutator.is_applicable(level, hrm, hrm_state),
            self.hrm_mutator.is_applicable(level, hrm, hrm_state)
        ])

        return jax.lax.switch(
            jax.random.choice(choice_rng, mask.shape[0], p=mask / mask.sum()),
            [self.hindsight_mutator.apply, self.level_mutator.apply, self.hrm_mutator.apply],
            sample_rng, level, hrm, hrm_state, env_state
        )


class SequentialMutator(JointMutator):
    """
    Applies a hindsight mutation (if applicable), followed by a number
    of level mutations and HRM mutations.
    """
    def __init__(
        self,
        num_level_edits: int,
        num_hrm_edits: int,
        use_move_agent: bool,
        use_add_rm_objs: bool,
        use_add_rm_rooms: bool,
        env_params: XMinigridEnvParams,
        use_add_rm_transitions: bool,
        alphabet_size: int,
        use_sparse_reward: bool,
        use_hindsight_level_mutation: bool = False,
        max_num_args: int = 2,
    ):
        self.max_num_args = max_num_args
        self.hindsight_mutator = HindsightAggMutator(use_hindsight_level_mutation, max_num_args)
        self.level_mutator = LevelMutatorJointWrapper(LevelSequenceMutator(
            num_level_edits, use_move_agent, use_add_rm_objs, use_add_rm_rooms, env_params, max_num_args
        ))
        self.hrm_mutator = HRMMutatorJointWrapper(HRMSequenceMutator(
            num_hrm_edits, use_add_rm_transitions, alphabet_size, use_sparse_reward, max_num_args
        ))

    def is_applicable(self, level: XMinigridLevel, hrm: HRM, hrm_state: HRMState) -> bool:
        return True

    def apply(
        self, rng: chex.PRNGKey, level: XMinigridLevel, hrm: HRM, hrm_state: HRMState, env_state: State
    ) -> Tuple[XMinigridLevel, HRM, chex.Array, chex.Array]:
        hindsight_rng, level_rng, hrm_rng = jax.random.split(rng, 3)

        level, hrm, hindsight_ids, hindsight_args = jax.lax.cond(
            self.hindsight_mutator.is_applicable(hrm, hrm_state),
            true_fun=lambda: self.hindsight_mutator.apply(hindsight_rng, level, hrm, hrm_state, env_state),
            false_fun=lambda: (level, hrm, -jnp.ones((1,), dtype=jnp.int32), -jnp.ones((1, self.max_num_args), dtype=jnp.int32))
        )

        level, hrm, level_ids, level_args = self.level_mutator.apply(level_rng, level, hrm, hrm_state, env_state)
        level, hrm, hrm_ids, hrm_args = self.hrm_mutator.apply(hrm_rng, level, hrm, hrm_state, env_state)

        return (
            level,
            hrm,
            jnp.concat((hindsight_ids, level_ids, hrm_ids)),
            jnp.concat((hindsight_args, level_args, hrm_args))
        )
