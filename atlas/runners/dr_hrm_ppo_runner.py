from typing import Tuple

import chex
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from .base_hrm_ppo_runner import BaseHRMConditionedPPORunner
from .base_runner_state import RunnerState as BaseRunnerState
from ..agents.types import ConditionedAgentState
from ..eval_loaders.types import EvaluationProblem
from ..envs.common.types import Timestep
from ..envs.common.wrappers import AutoResetHRMWrapper
from ..utils.evaluation import evaluate_agent, RolloutStats, Rollout
from ..utils.training import collect_trajectories_and_learn


class RunnerState(BaseRunnerState):
    cond_agent_state: ConditionedAgentState
    timestep: Timestep
    action: chex.Numeric  # scalar
    reward: chex.Numeric  # scalar

    # Logging
    update_count: chex.Numeric


class DRHRMConditionedPPORunner(BaseHRMConditionedPPORunner):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # Training environment
        self.training_env = AutoResetHRMWrapper(self.env, self.label_fn, self.problem_sampler, cfg.training.use_hrm_reward)
    
    def _make_train_and_eval_step_fn(self):
        def train_step(runner_state: RunnerState, _) -> Tuple[RunnerState, dict]:
            (rng, train_state, timestep, action, reward, c_a_state), (_, _, metrics) = collect_trajectories_and_learn(
                runner_state.rng,
                self.training_env,
                self.env_params,
                runner_state.train_state,
                runner_state.timestep,
                runner_state.action,
                runner_state.reward,
                runner_state.cond_agent_state,
                self.num_envs,
                self.cfg.training.num_steps,
                self.cfg.training.num_outer_steps,
                self.cfg.training.gamma,
                self.cfg.training.gae_lambda,
                self.cfg.training.num_minibatches,
                self.cfg.training.update_epochs,
                self.cfg.training.clip_eps,
                self.cfg.training.vf_coef,
                self.cfg.training.ent_coef,
                True,
                self.cfg.training.advantage_src,
            )

            return RunnerState(
                rng, train_state, c_a_state, timestep, action, reward, runner_state.update_count + 1
            ), metrics
        
        def eval(rng: chex.PRNGKey, train_state: TrainState, eval_problems: EvaluationProblem) -> Tuple[RolloutStats, Rollout]:
            eval_rng, agent_init_rng = jax.random.split(rng)
            return evaluate_agent(
                self.cfg.evaluation,
                eval_rng,
                train_state,
                self.eval_env,
                self.env_params,
                eval_problems,
                self.eval_loader.get_num_problems(),
                self.hrm_agent.initialize_state(self.eval_loader.get_num_problems(), agent_init_rng)
            )

        def train_and_eval_step(
            init_runner_state: RunnerState, eval_problems: EvaluationProblem
        ) -> Tuple[RunnerState, dict, dict, Rollout]:
            # Train
            runner_state, train_metrics = jax.lax.scan(train_step, init_runner_state, None, self.cfg.logging.interval)

            # Eval
            rng, eval_rng = jax.random.split(runner_state.rng)
            eval_metrics, eval_rollouts = eval(eval_rng, runner_state.train_state, eval_problems)

            return runner_state.replace(rng=rng), train_metrics, eval_metrics, eval_rollouts

        return train_and_eval_step

    def _init_runner_state(self) -> RunnerState:
        runner_rng, train_state_rng, env_rng, problem_rng, agent_rng = jax.random.split(self.rng, 5)
        
        # Randomly initialize the environment with a random level and a random HRM
        def _reset_fn(_env_rng: chex.PRNGKey, _problem_rng: chex.PRNGKey):
            level, hrm = self.problem_sampler(_problem_rng)
            return self.training_env.reset(_env_rng, self.env_params, level, hrm=hrm)

        timesteps = jax.vmap(_reset_fn)(
            jax.random.split(env_rng, self.num_envs),
            jax.random.split(problem_rng, self.num_envs)
        )

        # Randomly initialize the state of the agent (most parts of the agent are
        # actually static, e.g. only the R-GCN initialization is random when the
        # node features are chosen to be initially random)
        hrm_agent_init_state = self.hrm_agent.initialize_state(self.num_envs, agent_rng)

        return RunnerState(
            rng=runner_rng,
            train_state=self._init_train_state(train_state_rng),
            timestep=timesteps,
            action=jnp.zeros(self.num_envs, dtype=jnp.int32),
            reward=jnp.zeros(self.num_envs),
            cond_agent_state=hrm_agent_init_state,
            update_count=jnp.asarray(0),
        )
    
    def _get_num_updates(self, runner_state: RunnerState) -> int:
        return int(runner_state.update_count)
