from typing import Callable, Dict, Optional

import chex
import jax
import jax.numpy as jnp
from xminigrid.core.constants import Tiles, Colors
from xminigrid.core.grid import room

from atlas.envs.common.level import Level
from atlas.envs.common.types import Timestep, EnvParams
from atlas.envs.common.wrappers import AutoResetHRMWrapper, HRMWrapper
from atlas.envs.xminigrid.env import XMinigridEnv
from atlas.envs.xminigrid.labeling_function import XMinigridLabelingFunction
from atlas.envs.xminigrid.level import XMinigridLevel
from atlas.envs.xminigrid.level_sampling.base import XMinigridLevelSampler
from atlas.hrm.ops import add_condition, add_leaf_call, add_reward, init_hrm
from atlas.hrm.sampling.common import HRMSampler
from atlas.hrm.types import HRM
from atlas.problem_samplers.base import ProblemSampler
from atlas.problem_samplers.independent import IndependentProblemSampler


class TestHRMWrapper:
    MAX_EPISODE_STEPS = 10
    ROOM_SIZE = 5
    SAMPLED_LEVEL = XMinigridLevel(
        grid=room(ROOM_SIZE, ROOM_SIZE)
        .at[1, ROOM_SIZE // 2]
        .set(jnp.asarray((Tiles.BALL, Colors.GREY), dtype=jnp.uint8)),
        agent_pos=jnp.array((ROOM_SIZE - 2, ROOM_SIZE // 2), dtype=jnp.int32),
        agent_dir=0,
        height=ROOM_SIZE,
        width=ROOM_SIZE
    )

    class _TestLevelSampler(XMinigridLevelSampler):
        def sample(self, key: chex.PRNGKey, extras: Optional[Dict] = None) -> Level:
            return TestHRMWrapper.SAMPLED_LEVEL

    class _TestHRMSampler(HRMSampler):
        def __init__(self):
            super().__init__(
                max_num_rms=1, max_num_states=2, max_num_edges=1, max_num_literals=2, alphabet_size=2
            )

        def sample(self, key: chex.PRNGKey, extras: Optional[Dict] = None) -> HRM:
            root_id = 0
            hrm = init_hrm(
                root_id,
                self._max_num_rms,
                self._max_num_states,
                self._max_num_edges,
                self._max_num_literals,
            )
            transition = dict(hrm=hrm, rm_id=root_id, src_id=0, dst_id=1)
            add_leaf_call(**transition, edge_id=0)
            add_condition(**transition, edge_id=0, proposition=0, is_positive=True)
            add_reward(**transition, reward=1.0)
            return hrm

    def test_auto_reset(self):
        self._test_hrm_wrapper(True)

    def test_vanilla(self):
        self._test_hrm_wrapper(False)

    def _test_hrm_wrapper(self, auto_reset: bool):
        env, env_params, problem_sampler = self._init_test(auto_reset)

        reset_fn = jax.jit(chex.assert_max_traces(env.reset, n=1))
        step_fn = jax.jit(chex.assert_max_traces(env.step, n=1))

        # Initialize and check that the first timestep has the correct type
        timestep = self._reset(reset_fn, env_params, problem_sampler)
        assert timestep.first()
        assert timestep.reward == 0.0

        # Check that all intermediate timesteps have 'intermediate' type
        for i in range(self.MAX_EPISODE_STEPS - 1):
            timestep = step_fn(env_params, timestep, 1)
            assert timestep.mid()
            assert timestep.reward == 0.0

        # Check that the last step in the episode (reaching the maximum episode length)...
        #   - resets the environment to the first step for the auto_reset case, and
        #   - does not reset the environment to the first step otherwise (then reset the
        #     environment so that we can continue the tests).
        timestep = step_fn(env_params, timestep, 1)
        assert timestep.last()
        assert timestep.reward == 0.0

        if auto_reset:
            assert jnp.array_equal(timestep.state.grid, self.SAMPLED_LEVEL.grid)
            assert jnp.array_equal(
                timestep.state.agent.position, self.SAMPLED_LEVEL.agent_pos
            )
            assert timestep.state.agent.direction == self.SAMPLED_LEVEL.agent_dir
        else:
            assert jnp.array_equal(timestep.state.grid, self.SAMPLED_LEVEL.grid)
            assert jnp.array_equal(
                timestep.state.agent.position, self.SAMPLED_LEVEL.agent_pos
            )
            assert timestep.state.agent.direction == 2
            assert not timestep.extras.task_completed
            timestep = self._reset(reset_fn, env_params, problem_sampler)

        # Perform action to get to the goal
        timestep = step_fn(env_params, timestep, 0)
        assert timestep.last()
        assert timestep.reward == 1.0

        # If `auto_reset`, the environment is reset to the initial situation (i.e.
        # that being sampled); otherwise, the agent is in the position where it
        # achieved the goal.
        if auto_reset:
            assert jnp.array_equal(timestep.state.grid, self.SAMPLED_LEVEL.grid)
            assert jnp.array_equal(
                timestep.state.agent.position, self.SAMPLED_LEVEL.agent_pos
            )
            assert timestep.state.agent.direction == self.SAMPLED_LEVEL.agent_dir
        else:
            assert jnp.array_equal(timestep.state.grid, self.SAMPLED_LEVEL.grid)
            assert jnp.array_equal(
                timestep.state.agent.position,
                self.SAMPLED_LEVEL.agent_pos + jnp.array([-1, 0], dtype=jnp.int32),
            )
            assert timestep.state.agent.direction == self.SAMPLED_LEVEL.agent_dir
            assert timestep.extras.task_completed

    def _init_test(self, auto_reset: bool):
        env = XMinigridEnv()
        env_params = env.default_params().replace(
            height=self.ROOM_SIZE,
            width=self.ROOM_SIZE,
            max_steps=self.MAX_EPISODE_STEPS,
            non_door_obj_types=(Tiles.BALL,),
            door_obj_types=(),
            color_types=(Colors.GREY,),
        )
        labeling_function = XMinigridLabelingFunction(
            env_params,
            use_carrying_props=False,
            use_next_to_props=False,
        )
        level_sampler = self._TestLevelSampler(env_params)
        hrm_sampler = self._TestHRMSampler()
        problem_sampler = IndependentProblemSampler(level_sampler, hrm_sampler, labeling_function)

        if auto_reset:
            env = AutoResetHRMWrapper(env, labeling_function, problem_sampler)
        else:
            env = HRMWrapper(env, labeling_function)

        return env, env_params, problem_sampler

    def _reset(
        self,
        reset_fn: Callable,
        env_params: EnvParams,
        problem_sampler: ProblemSampler
    ) -> Timestep:
        level, hrm = jax.jit(problem_sampler.sample)(jax.random.PRNGKey(0))
        return reset_fn(
            key=jax.random.PRNGKey(0),
            env_params=env_params,
            level=level,
            hrm=hrm,
        )
