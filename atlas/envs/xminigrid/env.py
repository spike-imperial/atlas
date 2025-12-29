from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
from xminigrid.core.constants import NUM_ACTIONS
from xminigrid.core.goals import EmptyGoal
from xminigrid.core.rules import EmptyRule
from xminigrid.environment import Environment as _XMinigridEnv, EnvParams
from xminigrid.types import AgentState, EnvCarry, State

from .level import XMinigridLevel
from .observations import egocentric, full_3d
from .types import XMinigridEnvParams
from ..common.env import Environment, Timestep
from ..common.types import StepType


class XMinigridEnv(Environment[XMinigridEnvParams, XMinigridLevel]):
    class _XMinigridDummy(_XMinigridEnv):
        """
        A dummy class to implement the abstract methods from the base XLand-Minigrid
        class. These methods (`default_params` and `generate_problem`) should become
        unused, so an exception is thrown. The `time_limit` methods is overriden to
        enable customizing the amount steps after which an episode is considered
        complete.
        """

        def default_params(self, **kwargs: Any) -> EnvParams:
            raise NotImplementedError

        def time_limit(self, env_params: XMinigridEnvParams) -> int:
            return env_params.max_steps

        def _generate_problem(
            self, env_params: XMinigridEnvParams, key: jax.Array
        ) -> State:
            raise NotImplementedError

    def __init__(self):
        self._env = self._XMinigridDummy()

    def default_params(self, **kwargs) -> XMinigridEnvParams:
        return XMinigridEnvParams(view_size=5, max_steps=128)

    def num_actions(self, env_params: XMinigridEnvParams) -> int:
        return int(NUM_ACTIONS)

    def reset(
        self,
        key: chex.PRNGKey,
        env_params: XMinigridEnvParams,
        level: XMinigridLevel,
        **kwargs
    ) -> Timestep:
        state = State(
            key=key,
            step_num=0,
            grid=level.grid,
            agent=AgentState(position=level.agent_pos, direction=level.agent_dir, pocket=level.agent_pocket),
            goal_encoding=EmptyGoal().encode(),
            rule_encoding=EmptyRule().encode()[None, ...],
            carry=EnvCarry(),
        )

        return Timestep(
            key=key,
            state=state,
            step_type=StepType.FIRST,
            reward=jnp.asarray(0.0),
            discount=jnp.asarray(1.0),
            observation=self._get_obs(env_params, state),
            num_steps=state.step_num,
        )

    def step(
        self,
        env_params: XMinigridEnvParams,
        timestep: Timestep,
        action: chex.Array,
    ) -> Timestep:
        next_timestep = self._env.step(env_params, timestep, action)
        return Timestep(
            key=timestep.key,
            state=next_timestep.state,
            step_type=StepType(next_timestep.step_type),
            reward=next_timestep.reward,
            discount=next_timestep.discount,
            observation=self._get_obs(env_params, next_timestep.state),
            num_steps=next_timestep.state.step_num,
        )

    def render(
        self, env_params: XMinigridEnvParams, timestep: Timestep
    ) -> np.ndarray | str:
        return self._env.render(env_params, timestep)

    def _get_obs(self, env_params: XMinigridEnvParams, state: State) -> Any:
        observation = {}
        if env_params.use_ego_obs:
            observation["ego"] = egocentric(env_params, state)
        if env_params.use_full_obs:
            observation["full"] = full_3d(env_params, state)
        return observation
