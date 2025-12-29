from typing import Optional

import chex
import jax
import jax.numpy as jnp
import numpy as np

from .env import Environment, EnvParams, Level
from .labeling_function import LabelingFunction
from .types import StepType, Timestep, TimestepExtras
from ...hrm.ops import (
    get_hrm_completion,
    get_initial_hrm_state,
    is_root_rm,
    is_accepting_state,
    step as hrm_step,
)
from ...hrm.types import HRM, HRMReward, HRMState
from ...problem_samplers.base import ProblemSampler


class Wrapper(Environment[EnvParams, Level]):
    def __init__(self, env: Environment[EnvParams, Level]):
        self._env = env

    def default_params(self, **kwargs) -> EnvParams:
        return self._env.default_params(**kwargs)

    def num_actions(self, env_params: EnvParams) -> int:
        return self._env.num_actions(env_params)

    def reset(
        self, key: chex.PRNGKey, env_params: EnvParams, level: Level, **kwargs
    ) -> Timestep:
        return self._env.reset(key, env_params, level)

    def step(
        self, env_params: EnvParams, timestep: Timestep, action: chex.Array
    ) -> Timestep:
        return self._env.step(env_params, timestep, action)

    def render(self, env_params: EnvParams, timestep: Timestep) -> np.ndarray | str:
        return self._env.render(env_params, timestep)


class HRMWrapper(Wrapper):
    """
    Augments timesteps with information about the HRM determining the task being
    done and the current HRM state.

    If the reward coming from the HRM is used, it is determined by that obtained
    in the root of the HRM.

    Warning: Take into account that if a transition for the initial state is
    satisfied at the time of resetting the environment, the reward will be 0
    even if the transition is labeled with another reward (the agent has not
    made any choice it should be rewarded for, it just happens to be a
    condition that initially holds).
    """

    class HRMExtras(TimestepExtras):
        hrm: HRM
        hrm_state: HRMState
        task_completed: bool
        level: Level
        hrm_completion: Optional[float]

    def __init__(
        self,
        env: Environment[EnvParams, Level],
        labeling_function: LabelingFunction,
        use_hrm_reward: bool = True,
        use_hrm_completion: bool = False,
    ):
        super().__init__(env)

        self._labeling_fn = labeling_function
        self._use_hrm_reward = use_hrm_reward
        self._use_hrm_completion = use_hrm_completion

    def reset(
        self,
        key: chex.PRNGKey,
        env_params: EnvParams,
        level: Level,
        **kwargs,
    ) -> Timestep:
        # Perform standard timestep
        timestep = super().reset(key, env_params, level)

        # Perform a step in the HRM based on the label obtained from the initial
        # observation
        hrm = kwargs["hrm"]
        hrm_state, _ = hrm_step(
            hrm,
            get_initial_hrm_state(hrm),
            self._labeling_fn.get_label(timestep),
        )

        # Add information of the HRM to the timestep, and establish that the step
        # type is LAST based on the updated HRM state (e.g., if we initially observe
        # a label that satisfies a transition from the initial to the accepting state)
        # The type of the reward is important to avoid recompilations.
        return timestep.replace(
            step_type=jax.lax.cond(
                pred=self._is_episode_complete(timestep, hrm, hrm_state),
                true_fun=lambda: StepType.LAST,
                false_fun=lambda: timestep.step_type,
            ),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            extras=self._get_extras(level, hrm, hrm_state, timestep.state),
        )

    def step(
        self, env_params: EnvParams, timestep: Timestep, action: chex.Array
    ) -> Timestep:
        # Obtain the current HRM, perform the environment step and perform a
        # transition in the HRM
        hrm = timestep.extras.hrm
        next_timestep = super().step(env_params, timestep, action)
        next_hrm_state, hrm_reward = hrm_step(
            hrm,
            timestep.extras.hrm_state,
            self._labeling_fn.get_label(next_timestep),
        )

        # Compute the reward
        next_timestep = next_timestep.replace(
            reward=self._get_reward(
                hrm,
                hrm_reward,
                self._is_hrm_task_complete(hrm, next_hrm_state),
                next_timestep.num_steps,
                env_params.max_steps,
            )
        )

        # Add the information about the HRM to the timestep and establish
        # that the step type is LAST if the episode is completed (mainly
        # to determine that the accepting state of the root of the HRM is
        # reached).
        return next_timestep.replace(
            step_type=jax.lax.cond(
                pred=self._is_episode_complete(next_timestep, hrm, next_hrm_state),
                true_fun=lambda: StepType.LAST,
                false_fun=lambda: next_timestep.step_type,
            ),
            extras=self._get_extras(timestep.extras.level, hrm, next_hrm_state, next_timestep.state),
        )

    def _get_extras(self, level: Level, hrm: HRM, hrm_state: HRMState, state: chex.ArrayTree) -> HRMExtras:
        return self.HRMExtras(
            hrm=hrm,
            hrm_state=hrm_state,
            level=level,
            task_completed=self._is_hrm_task_complete(hrm, hrm_state),
            hrm_completion=self._get_hrm_completion(hrm, hrm_state),
        )

    def _is_episode_complete(
        self, timestep: Timestep, hrm: HRM, hrm_state: HRMState
    ) -> bool:
        """
        Returns True one of the following conditions holds:
            - if the environment has reached the last timestep (i.e., the maximum
              number of steps has been performed), or
            - if the accepting state of the root in the HRM is reached.
        """
        return timestep.last() | self._is_hrm_task_complete(hrm, hrm_state)

    def _is_hrm_task_complete(self, hrm: HRM, hrm_state: HRMState) -> bool:
        """
        Returns True if the accepting state of the root in the HRM is reached.
        """
        return is_root_rm(hrm, hrm_state.rm_id) & is_accepting_state(
            hrm, hrm_state.state_id
        )

    def _get_hrm_completion(self, hrm: HRM, hrm_state: HRMState) -> float:
        """
        Returns the degree of completion of a given HRM.
        Check the method's documentation.
        """
        return get_hrm_completion(hrm, hrm_state) if self._use_hrm_completion else 0.0

    def _get_reward(
        self, hrm: HRM, hrm_reward: HRMReward, is_task_complete: bool, num_steps: int, max_steps: int,
    ) -> float:
        """
        Returns the reward for the current step. If HRM reward is used, the
        reward accumulated in the root is returned; otherwise, a value
        dependent on the number of steps required is returned upon completion
        (inspired by the usual reward function used in Minigrid).

        TODO: Implement different reward aggregation strategies. Skeleton:
            match reward_agg:
                case "root":
                    return hrm_reward.scalar[hrm.root_id]
                case "current":
                    return hrm_reward.scalar[prev_hrm_state.rm_id]
                case "sum":
                    return jnp.sum(hrm_reward.mask * hrm_reward.scalar)
         There can be more operations such as `prod`, `max` or `min` applied akin to the `sum`.

        TODO: if the rewards for each individual RMs are relevant, perhaps we should
         move this to the agent side, which would receive `HRMReward` structures
         instead.
        """
        if self._use_hrm_reward:
            return hrm_reward.scalar[hrm.root_id]
        return is_task_complete.astype(jnp.float32) * (1.0 - 0.9 * num_steps / max_steps)


class AutoResetHRMWrapper(HRMWrapper):
    """
    Wrapper that automatically resets the environment and the HRM in two cases:
        - if the environment has reached the last timestep (i.e., the maximum
          number of steps has been performed), or
        - if the accepting state of the root in the HRM is reached.

    Based on https://github.com/DramaCow/jaxued/blob/main/src/jaxued/wrappers/autoreset.py.
    """

    def __init__(
        self,
        env: Environment[EnvParams, Level],
        labeling_function: LabelingFunction,
        problem_sampler: ProblemSampler,
        use_hrm_reward: bool = True,
        use_hrm_completion: bool = False,
    ):
        super().__init__(env, labeling_function, use_hrm_reward, use_hrm_completion)
        self._problem_sampler = problem_sampler

    def step(
        self, env_params: EnvParams, timestep: Timestep, action: chex.Array
    ) -> Timestep:
        next_timestep = super().step(env_params, timestep, action)

        # If the episode is complete, we perform a reset of the HRM and the
        # environment level; otherwise, we employ the timestep returned by
        # the parent wrapper.
        return jax.lax.cond(
            pred=self._is_episode_complete(
                next_timestep, next_timestep.extras.hrm, next_timestep.extras.hrm_state
            ),
            true_fun=lambda: self._auto_reset(env_params, next_timestep),
            false_fun=lambda: next_timestep,
        )

    def _auto_reset(self, env_params: EnvParams, timestep: Timestep) -> Timestep:
        problem_key, reset_key = jax.random.split(timestep.key, 2)

        # Obtain the timestep for resetting
        level, hrm = self._problem_sampler(problem_key)
        reset_timestep = self.reset(
            key=reset_key,
            env_params=env_params,
            level=level,
            hrm=hrm,
        )

        # Keep the information about the reward and the step type (otherwise
        # it is lost and cannot be leveraged by the learning algorithm!)
        return reset_timestep.replace(
            reward=timestep.reward,
            step_type=StepType.LAST,
            extras=reset_timestep.extras.replace(
                task_completed=timestep.extras.task_completed,
                hrm_completion=timestep.extras.hrm_completion,
            )
        )


class AutoReplayHRMWrapper(HRMWrapper):
    """
    Wrapper that replays the same level by resetting to it at the end of an episode.
    This is useful for training/rolling out multiple times on the same level.

    Based on https://github.com/DramaCow/jaxued/blob/main/src/jaxued/wrappers/autoreplay.py.
    """
    class AutoReplayHRMExtras(HRMWrapper.HRMExtras):
        last_state: chex.ArrayTree
        last_hrm_state: HRMState

    def step(
        self, env_params: EnvParams, timestep: Timestep, action: chex.Array
    ) -> Timestep:
        next_timestep = super().step(env_params, timestep, action)

        # If the episode is complete, we perform a reset to the same HRM (initial state)
        # and the environment level (initial state); otherwise, we employ the timestep
        # returned by the parent wrapper.
        return jax.lax.cond(
            pred=self._is_episode_complete(
                next_timestep, next_timestep.extras.hrm, next_timestep.extras.hrm_state
            ),
            true_fun=lambda: self._auto_reset(env_params, next_timestep),
            false_fun=lambda: next_timestep,
        )

    def _get_extras(self, level: Level, hrm: HRM, hrm_state: HRMState, state: chex.ArrayTree) -> HRMWrapper.HRMExtras:
        return self.AutoReplayHRMExtras(
            hrm=hrm,
            hrm_state=hrm_state,
            level=level,
            task_completed=self._is_hrm_task_complete(hrm, hrm_state),
            hrm_completion=self._get_hrm_completion(hrm, hrm_state),
            last_state=state,
            last_hrm_state=hrm_state,
        )

    def _auto_reset(self, env_params: EnvParams, timestep: Timestep) -> Timestep:
        reset_key, _ = jax.random.split(timestep.key, 2)

        # Obtain the timestep for resetting
        reset_timestep = self.reset(
            key=reset_key,
            env_params=env_params,
            level=timestep.extras.level,
            hrm=timestep.extras.hrm,
        )

        # Keep the information about the reward and the step type (otherwise
        # it is lost and cannot be leveraged by the learning algorithm!)
        return reset_timestep.replace(
            reward=timestep.reward,
            step_type=StepType.LAST,
            extras=reset_timestep.extras.replace(
                task_completed=timestep.extras.task_completed,
                hrm_completion=timestep.extras.hrm_completion,
                last_state=timestep.state,
                last_hrm_state=timestep.extras.hrm_state,
            )
        )
