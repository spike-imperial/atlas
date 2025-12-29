from collections import defaultdict
from enum import IntEnum
from typing import Callable, List, Tuple

import chex
from flax.training.train_state import TrainState
from hydra.utils import instantiate
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from .base_hrm_ppo_runner import BaseHRMConditionedPPORunner
from .base_runner_state import RunnerState as BaseRunnerState
from ..envs.common.types import Timestep
from ..envs.common.level import Level
from ..envs.common.wrappers import AutoReplayHRMWrapper
from ..envs.xminigrid.level import get_level_sizes
from ..eval_loaders.types import EvaluationProblem
from ..hrm.types import HRM
from ..ued.buffer import Buffer, BufferManager
from ..ued.scoring import get_scores_fn
from ..utils.evaluation import evaluate_agent, RolloutStats, Rollout
from ..utils.plotting_utils import (
    plotly_buffer_scores,
    plotly_buffer_staleness,
    plotly_epsilon,
    plotly_mutation_category_count,
    plotly_mutation_count,
    plotly_mutation_fraction,
    plotly_mutation_hindsight_lvl,
    plotly_num_mutations,
    plotly_prop_frequency,
    plotly_xminigrid_prob_distrib_sampling,
    plotly_xminigrid_prob_distrib_solving,
    plotly_xminigrid_num_rms,
)
from ..utils.training import (
    collect_trajectories_and_learn,
    compute_max_returns,
    compute_mean_disc_return,
    compute_hrm_completion_sum,
    compute_task_completions,
    Transition
)


class UpdateState(IntEnum):
    DR = 0
    REPLAY = 1


class RunnerState(BaseRunnerState):
    buffer: Buffer
    update_state: UpdateState

    # Logging
    num_dr_updates: int
    dr_last_level_batch: chex.ArrayTree
    dr_last_hrm_batch: chex.ArrayTree
    dr_last_task_solved_batch: chex.ArrayTree
    dr_last_scores_batch: chex.ArrayTree

    num_replay_updates: int
    replay_last_level_batch: chex.ArrayTree
    replay_last_hrm_batch: chex.ArrayTree
    replay_last_state_batch: chex.ArrayTree
    replay_last_hrm_state_batch: chex.ArrayTree
    replay_last_mutation_ids_batch: chex.Array
    replay_last_mutation_round_batch: chex.ArrayTree
    replay_last_task_solved_batch: chex.ArrayTree
    replay_last_scores_batch: chex.ArrayTree

    num_mutation_updates: int
    mutation_last_level_batch: chex.ArrayTree
    mutation_last_hrm_batch: chex.ArrayTree
    mutation_last_task_solved_batch: chex.ArrayTree
    mutation_last_scores_batch: chex.ArrayTree


class PLRHRMConditionedPPORunner(BaseHRMConditionedPPORunner):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # Score function
        score_fn_names, score_fn_args, score_fn_coeffs = zip(*[
            (score_fn.name, score_fn.args, score_fn.coeff) for score_fn in self.cfg.algorithm.score_functions
        ])
        self.score_fn: Callable = get_scores_fn(score_fn_names, score_fn_args)
        self.score_fn_names: List = score_fn_names
        self.num_scores: int = len(self.score_fn_names)

        # Training environment
        use_hrm_completion = "hrm_completion" in self.score_fn_names
        self.training_env = AutoReplayHRMWrapper(self.env, self.label_fn, cfg.training.use_hrm_reward, use_hrm_completion)

        # Buffer manager
        self.buffer_manager: BufferManager = instantiate(
            cfg.algorithm.buffer,
            num_scores=self.num_scores,
            score_coeffs=jnp.array(score_fn_coeffs)
        )

        # Mutation function
        self.mutator = instantiate(
            self.cfg.algorithm.mutator,
            env_params=self.env_params,
            alphabet_size=self.label_fn.get_alphabet_size(),
        )

    def _make_train_and_eval_step_fn(self):
        def _collect_trajectories_and_learn(rng: chex.PRNGKey, train_state: TrainState, init_timesteps: Timestep, update_grad: bool):            
            rng, agent_init_rng = jax.random.split(rng)
            (rng, train_state, last_timestep, *_), rollouts = collect_trajectories_and_learn(
                rng,
                self.training_env,
                self.env_params,
                train_state,
                init_timesteps,
                jnp.zeros(self.num_envs, dtype=jnp.int32),
                jnp.zeros(self.num_envs),
                self.hrm_agent.initialize_state(self.num_envs, agent_init_rng),
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
                update_grad,
                self.cfg.training.advantage_src,
            )

            return (rng, train_state, last_timestep), rollouts

        def train_step(runner_state: RunnerState, _) -> Tuple[RunnerState, dict]:
            def get_num_unique_insertions(indices: chex.Array) -> int:
                """
                Returns the number of indices whose content has been replaced in the buffer
                The `unique` is added since two items may sequentially overwrite the same
                position in the buffer.
                """
                return (jnp.unique(indices, size=self.num_envs) > -1).sum()

            def get_solved_task_state(transitions: Transition) -> chex.Array:
                """
                Returns whether a given task has been completed based on whether one of the
                transitions in the rollout shows it has been.
                """
                return jnp.any(transitions.task_completed, axis=0)

            def get_return_momentum(
                returns: chex.Array, prev_returns: chex.ArrayTree, prev_momentum: chex.Array,
            ) -> chex.Array:
                """
                Returns the momentum of the discounted return difference using an exponentially moving average.
                """
                momentum_coeff = self.cfg.algorithm.momentum_coeff
                delta = returns - prev_returns
                return momentum_coeff * prev_momentum + (1 - momentum_coeff) * delta

            def on_new_problems(rng: chex.PRNGKey):
                rng, env_rng, problem_rng = jax.random.split(rng, 3)

                # Reset
                reset_rngs = jax.random.split(env_rng, self.num_envs)
                new_levels, new_hrms = jax.vmap(self.problem_sampler)(
                    jax.random.split(problem_rng, self.num_envs)
                )
                init_timesteps = jax.vmap(
                    lambda _env_rng, _level, _hrm: self.training_env.reset(
                        _env_rng, self.env_params, _level, hrm=_hrm,
                    )
                )(reset_rngs, new_levels, new_hrms)

                # Collect trajectories and train
                (rng, train_state, _), (transitions, advantages, metrics) = _collect_trajectories_and_learn(
                    rng, runner_state.train_state, init_timesteps, self.cfg.algorithm.exploratory_grad_updates,
                )

                # Compute scores
                max_returns = compute_max_returns(transitions)
                mean_disc_returns = compute_mean_disc_return(transitions, self.cfg.training.gamma)
                momentums = get_return_momentum(mean_disc_returns, jnp.zeros_like(mean_disc_returns), jnp.zeros_like(mean_disc_returns))
                task_completions, episode_count = compute_task_completions(transitions)
                hrm_completions_sum, _ = compute_hrm_completion_sum(transitions)
                scores = self.score_fn(
                    transitions, max_returns, momentums, advantages, task_completions, hrm_completions_sum, episode_count
                )

                pholder_mutation_ids, pholder_mutation_args = jax.tree_util.tree_map(
                    lambda x: -jnp.ones((self.num_envs, *x.shape[1:]), dtype=jnp.int32),
                    (runner_state.buffer["extra"]["mutation_ids"], runner_state.buffer["extra"]["mutation_args"])
                )

                # Push batch to the buffer
                buffer, indices = self.buffer_manager.insert_batch(
                    runner_state.buffer, new_levels, new_hrms, scores, problem_extras={
                        "max_return": max_returns,
                        "disc_return": mean_disc_returns,
                        "return_momentum": momentums,
                        "episode_count": episode_count,
                        "task_completions": task_completions,
                        "hrm_completions": hrm_completions_sum,
                        "mutation_ids": pholder_mutation_ids,
                        "mutation_args": pholder_mutation_args,
                        "mutation_rounds": jnp.zeros((self.num_envs,), dtype=jnp.int32),
                    }
                )

                # Update data about the number of new problems in the buffer (to measure how static it is)
                metrics["num_dr_problems_added"] = get_num_unique_insertions(indices)
                metrics["num_mutation_problems_added"] = -1

                return runner_state.replace(
                    rng=rng,
                    train_state=train_state,
                    buffer=buffer,
                    update_state=UpdateState.DR,
                    num_dr_updates=runner_state.num_dr_updates + 1,
                    dr_last_level_batch=new_levels,
                    dr_last_hrm_batch=new_hrms,
                    dr_last_task_solved_batch=get_solved_task_state(transitions),
                    dr_last_scores_batch=scores,
                ), metrics

            def on_replay_problems(rng: chex.PRNGKey):
                rng, sample_rng, reset_rng = jax.random.split(rng, 3)

                # Sample the levels and HRMs
                buffer, (inds, levels, hrms) = self.buffer_manager.sample_replay_problems(
                    runner_state.buffer, sample_rng, self.num_envs,
                )

                # Reset
                init_timesteps = jax.vmap(self.training_env.reset)(
                    jax.random.split(reset_rng, self.num_envs),
                    self.env_params,
                    levels,
                    hrm=hrms,
                )

                # Collect trajectories and train
                (rng, train_state, last_timestep), (transitions, advantages, metrics) = _collect_trajectories_and_learn(
                    rng, runner_state.train_state, init_timesteps, update_grad=True,
                )

                # Compute auxiliary data for computing scores
                max_returns = compute_max_returns(transitions)
                task_completions, episode_count = compute_task_completions(transitions)
                hrm_completions_sum, _ = compute_hrm_completion_sum(transitions)

                # Update the auxiliary data
                problems_extra = self.buffer_manager.get_problems_extra(buffer, inds)
                max_returns = jnp.maximum(problems_extra["max_return"], max_returns)
                mean_disc_returns = compute_mean_disc_return(transitions, self.cfg.training.gamma)
                momentums = get_return_momentum(mean_disc_returns, problems_extra["disc_return"], problems_extra["return_momentum"])
                if self.cfg.algorithm.use_acc_completion_count:
                    task_completions += problems_extra["task_completions"]
                    hrm_completions_sum += problems_extra["hrm_completions"]
                    episode_count += problems_extra["episode_count"]

                # Compute the score
                scores = self.score_fn(
                    transitions, max_returns, momentums, advantages, task_completions, hrm_completions_sum, episode_count
                )

                # Push to the buffer
                buffer = self.buffer_manager.update_batch(buffer, inds, scores, extras={
                    "max_return": max_returns,
                    "disc_return": mean_disc_returns,
                    "return_momentum": momentums,
                    "episode_count": episode_count,
                    "task_completions": task_completions,
                    "hrm_completions": hrm_completions_sum,
                    "mutation_ids": problems_extra["mutation_ids"],
                    "mutation_args": problems_extra["mutation_args"],
                    "mutation_rounds": problems_extra["mutation_rounds"],
                })

                # Update data about the number of new problems in the buffer (to measure how static it is)
                metrics["num_dr_problems_added"] = -1
                metrics["num_mutation_problems_added"] = -1

                return runner_state.replace(
                    rng=rng,
                    train_state=train_state,
                    buffer=buffer,
                    update_state=UpdateState.REPLAY,
                    num_replay_updates=runner_state.num_replay_updates + 1,
                    replay_last_level_batch=levels,
                    replay_last_hrm_batch=hrms,
                    replay_last_state_batch=last_timestep.extras.last_state,
                    replay_last_hrm_state_batch=last_timestep.extras.last_hrm_state,
                    replay_last_mutation_ids_batch=problems_extra["mutation_ids"],
                    replay_last_mutation_round_batch=problems_extra["mutation_rounds"],
                    replay_last_task_solved_batch=get_solved_task_state(transitions),
                    replay_last_scores_batch=scores,
                ), metrics

            def on_mutate_problems(rng: chex.PRNGKey):
                rng, mutate_rng, reset_rng = jax.random.split(rng, 3)

                # Mutate
                mutated_levels, mutated_hrms, mutation_ids, mutation_args = jax.vmap(self.mutator.apply)(
                    jax.random.split(mutate_rng, self.num_envs),
                    runner_state.replay_last_level_batch,
                    runner_state.replay_last_hrm_batch,
                    runner_state.replay_last_hrm_state_batch,
                    runner_state.replay_last_state_batch,
                )

                # Reset
                init_timesteps = jax.vmap(self.training_env.reset)(
                    jax.random.split(reset_rng, self.num_envs),
                    self.env_params,
                    mutated_levels,
                    hrm=mutated_hrms,
                )

                # Collect trajectories and train
                (rng, train_state, _), (transitions, advantages, metrics) = _collect_trajectories_and_learn(
                    rng, runner_state.train_state, init_timesteps, self.cfg.algorithm.exploratory_grad_updates,
                )

                # Compute scores and push to buffer
                max_returns = compute_max_returns(transitions)
                mean_disc_returns = compute_mean_disc_return(transitions, self.cfg.training.gamma)
                momentums = get_return_momentum(mean_disc_returns, jnp.zeros_like(mean_disc_returns), jnp.zeros_like(mean_disc_returns))
                task_completions, episode_count = compute_task_completions(transitions)
                hrm_completions_sum, _ = compute_hrm_completion_sum(transitions)
                scores = self.score_fn(
                    transitions, max_returns, momentums, advantages, task_completions, hrm_completions_sum, episode_count
                )
                buffer, indices = self.buffer_manager.insert_batch(
                    runner_state.buffer, mutated_levels, mutated_hrms, scores, problem_extras={
                        "max_return": max_returns,
                        "disc_return": mean_disc_returns,
                        "return_momentum": momentums,
                        "episode_count": episode_count,
                        "task_completions": task_completions,
                        "hrm_completions": hrm_completions_sum,
                        "mutation_ids": mutation_ids,
                        "mutation_args": mutation_args,
                        "mutation_rounds": runner_state.replay_last_mutation_round_batch + 1,
                    }
                )

                # Update data about the number of new problems in the buffer (to measure how static it is)
                metrics["num_dr_problems_added"] = -1
                metrics["num_mutation_problems_added"] = get_num_unique_insertions(indices)

                return runner_state.replace(
                    rng=rng,
                    train_state=train_state,
                    buffer=buffer,
                    update_state=UpdateState.DR,
                    num_mutation_updates=runner_state.num_mutation_updates + 1,
                    mutation_last_level_batch=mutated_levels,
                    mutation_last_hrm_batch=mutated_hrms,
                    mutation_last_task_solved_batch=get_solved_task_state(transitions),
                    mutation_last_scores_batch=scores,
                ), metrics

            rng, rng_replay = jax.random.split(runner_state.rng)

            branches = [on_new_problems, on_replay_problems]
            if self.cfg.algorithm.use_accel:
                s = runner_state.update_state
                branch = (1 - s) * self.buffer_manager.sample_replay_decision(
                    runner_state.buffer, rng_replay
                ) + 2 * s
                branches.append(on_mutate_problems)
            else:
                branch = self.buffer_manager.sample_replay_decision(
                    runner_state.buffer, rng_replay
                ).astype(int)

            return jax.lax.switch(branch, branches, rng)

        def eval(rng: chex.PRNGKey, train_state: TrainState, eval_problems: EvaluationProblem) -> Tuple[RolloutStats, Rollout]:
            eval_rng, agent_init_rng = jax.random.split(rng)
            num_eval_problems = eval_problems.hrm.root_id.shape[0]
            return evaluate_agent(
                self.cfg.evaluation,
                eval_rng,
                train_state,
                self.eval_env,
                self.env_params,
                eval_problems,
                num_eval_problems,
                self.hrm_agent.initialize_state(num_eval_problems, agent_init_rng)
            )

        def train_and_eval_step(
            init_runner_state: RunnerState, eval_problems: EvaluationProblem
        ) -> Tuple[RunnerState, dict, dict, Rollout]:
            # Train
            runner_state, train_metrics = jax.lax.scan(train_step, init_runner_state, None, self.cfg.logging.interval)

            # Eval
            rng, eval_rng = jax.random.split(runner_state.rng)
            eval_problems = self._get_extended_eval_problems(runner_state, eval_problems)
            eval_metrics, eval_rollouts = eval(eval_rng, runner_state.train_state, eval_problems)

            return runner_state.replace(rng=rng), train_metrics, eval_metrics, eval_rollouts
        
        return train_and_eval_step

    def _init_runner_state(self) -> RunnerState:
        runner_rng, train_state_rng = jax.random.split(self.rng, 2)

        pholder_level, pholder_hrm = self.problem_sampler.sample(jax.random.PRNGKey(0))
        pholder_timestep = self.training_env.reset(jax.random.PRNGKey(0), self.env_params, pholder_level, hrm=pholder_hrm)

        pholder_level_batch, pholder_hrm_batch, pholder_timestep_batch = jax.tree_map(
            lambda x: jnp.array([x]).repeat(self.num_envs, axis=0),
            (pholder_level, pholder_hrm, pholder_timestep)
        )
        pholder_last_scores_batch = jnp.zeros((self.num_envs, self.num_scores))

        if self.cfg.algorithm.use_accel:
            _, _, pholder_mutation_ids, pholder_mutation_args = self.mutator.apply(
                jax.random.PRNGKey(0),
                pholder_level,
                pholder_hrm,
                pholder_timestep.extras.last_hrm_state,
                pholder_timestep.extras.last_state,
            )
        else:
            pholder_mutation_ids = jnp.zeros((1,), dtype=jnp.int32)
            pholder_mutation_args = jnp.zeros((1,), dtype=jnp.int32)

        return RunnerState(
            rng=runner_rng,
            train_state=self._init_train_state(train_state_rng),
            buffer=self.buffer_manager.initialize(pholder_level, pholder_hrm, {
                "max_return": -jnp.inf,
                "disc_return": 0.0,
                "return_momentum": 0.0,
                "episode_count": 0,
                "task_completions": 0,
                "hrm_completions": 0.0,
                "mutation_ids": pholder_mutation_ids,    # mutations performed on the original sample
                "mutation_args": pholder_mutation_args,  # mutation arguments (objects changed, propositions changed)
                "mutation_rounds": 0,                    # how many mutations have led to this sample
            }),
            update_state=jnp.asarray(UpdateState.DR),
            num_dr_updates=jnp.asarray(0),
            dr_last_level_batch=pholder_level_batch,
            dr_last_hrm_batch=pholder_hrm_batch,
            dr_last_task_solved_batch=jnp.zeros((self.num_envs,), dtype=jnp.bool),
            dr_last_scores_batch=pholder_last_scores_batch,
            num_replay_updates=jnp.asarray(0),
            replay_last_level_batch=pholder_level_batch,
            replay_last_hrm_batch=pholder_hrm_batch,
            replay_last_state_batch=pholder_timestep_batch.extras.last_state,
            replay_last_hrm_state_batch=pholder_timestep_batch.extras.last_hrm_state,
            replay_last_mutation_ids_batch=-jnp.ones((self.num_envs, *pholder_mutation_ids.shape), dtype=jnp.int32),
            replay_last_mutation_round_batch=jnp.zeros((self.num_envs,), dtype=jnp.int32),
            replay_last_task_solved_batch=jnp.zeros((self.num_envs,), dtype=jnp.bool),
            replay_last_scores_batch=pholder_last_scores_batch,
            num_mutation_updates=jnp.asarray(0),
            mutation_last_level_batch=pholder_level_batch,
            mutation_last_hrm_batch=pholder_hrm_batch,
            mutation_last_task_solved_batch=jnp.zeros((self.num_envs,), dtype=jnp.bool),
            mutation_last_scores_batch=pholder_last_scores_batch,
        )

    def _get_log_train_metrics(self, train_metrics: dict, runner_state: RunnerState, step: int, time_delta: float):
        log_metrics = super()._get_log_train_metrics(train_metrics, runner_state, step, time_delta)
        log_metrics_step = log_metrics[step]

        # Counters
        log_metrics_step.update({
            "num_dr_updates": runner_state.num_dr_updates,
            "num_replay_updates": runner_state.num_replay_updates,
            "num_mutation_updates": runner_state.num_mutation_updates,
            "num_policy_updates": (
                runner_state.num_replay_updates +
                self.cfg.algorithm.exploratory_grad_updates * (runner_state.num_dr_updates + runner_state.num_mutation_updates)
            )
        })

        # Buffer
        log_metrics_step.update(self._get_buffer_metrics(runner_state.buffer))
        
        # Batches of last sampled problems
        def _get_batch_data(batch_type: str, levels: Level, hrms: HRM, solved: chex.Array, scores: chex.Array) -> dict:
            scores_dict = {}
            for i in range(self.num_scores):
                score_fn_name = self.score_fn_names[i].lower()
                scores_dict.update({
                    f"training/{batch_type}_{score_fn_name}_mean": scores[:, i].mean(),
                    f"training/{batch_type}_{score_fn_name}_max": scores[:, i].max(),
                })

            return {
                **scores_dict,
                f"training/{batch_type}_problems": self._get_rendered_problem_batch(
                    self.training_env, levels, hrms, self.cfg.algorithm.num_rendered_training_samples
                ),
                f"training/{batch_type}_problem_distribution": plotly_xminigrid_prob_distrib_solving(
                    levels, hrms, solved,
                ),
            }

        if runner_state.num_dr_updates > 0:
            log_metrics_step.update(_get_batch_data(
                "dr",
                runner_state.dr_last_level_batch,
                runner_state.dr_last_hrm_batch,
                runner_state.dr_last_task_solved_batch,
                runner_state.dr_last_scores_batch,
            ))

        if runner_state.num_replay_updates > 0:
            log_metrics_step.update({
                **_get_batch_data(
                    "replay",
                    runner_state.replay_last_level_batch,
                    runner_state.replay_last_hrm_batch,
                    runner_state.replay_last_task_solved_batch,
                    runner_state.replay_last_scores_batch,
                ),
                "training/replay_mutation_count": plotly_mutation_count(runner_state.replay_last_mutation_ids_batch),
                "training/replay_mutation_category": plotly_mutation_category_count(runner_state.replay_last_mutation_ids_batch),
                "training/replay_mutation_fraction": plotly_mutation_fraction(runner_state.replay_last_mutation_round_batch),
                "training/replay_num_mutations": plotly_num_mutations(runner_state.replay_last_mutation_ids_batch),
            })

        if runner_state.num_mutation_updates > 0:
            log_metrics_step.update(_get_batch_data(
                "mutation",
                runner_state.mutation_last_level_batch,
                runner_state.mutation_last_hrm_batch,
                runner_state.mutation_last_task_solved_batch,
                runner_state.mutation_last_scores_batch,
            ))

        # Highest scoring and weighted problems
        log_metrics_step.update({
            "training/highest_scoring_problem": self._get_rendered_problem_batch(
                self.training_env,
                *self.buffer_manager.get_highest_scored_problems(runner_state.buffer, num_problems=1),
                num_rendered_problems=1
            ),
            "training/highest_weighted_problem": self._get_rendered_problem_batch(
                self.training_env,
                *self.buffer_manager.get_highest_weighted_problems(runner_state.buffer, num_problems=1),
                num_rendered_problems=1
            ),
        })

        return log_metrics

    def _get_buffer_metrics(self, buffer: Buffer, prefix: str = "training") -> dict:
        # Aggregate score and staleness metrics
        metrics = {
            f"{prefix}/buffer/size": buffer["size"],
            f"{prefix}/buffer/episode_count": buffer["episode_count"],
        }

        problem_weights = self.buffer_manager.problem_weights(buffer)

        # Log staleness metrics
        min_staleness, mean_staleness, max_staleness = self.buffer_manager.get_staleness_aggregates(buffer)
        metrics.update({
            f"{prefix}/buffer/min_staleness": min_staleness,
            f"{prefix}/buffer/mean_staleness": mean_staleness,
            f"{prefix}/buffer/max_staleness": max_staleness,
            f"{prefix}/buffer/staleness": plotly_buffer_staleness(
                self.buffer_manager.staleness(buffer),
                self.buffer_manager.staleness_weights(buffer),
                problem_weights,
                num_bins=50,
            ),
        })

        # Log score metrics
        if buffer["size"] > 0:
            total_score_weights = self.buffer_manager.score_weights(buffer)
            ind_score_weights = jax.vmap(
                lambda x: self.buffer_manager.score_weights(
                    buffer, score_coeffs=jax.nn.one_hot(jnp.array([x]), num_classes=self.num_scores).squeeze(axis=0)
                )
            )(jnp.arange(self.num_scores))

            # Score aggregates
            weighted_scores = self.buffer_manager.get_weighted_score(buffer)
            min_score, mean_score, max_score = self.buffer_manager.get_score_aggregates(buffer)
            for i in range(self.num_scores):
                score_fn_name = self.score_fn_names[i].lower()
                metrics.update({
                    f"{prefix}/buffer/weighted_score/{score_fn_name}": weighted_scores[i],
                    f"{prefix}/buffer/min_score/{score_fn_name}": min_score[i],
                    f"{prefix}/buffer/mean_score/{score_fn_name}": mean_score[i],
                    f"{prefix}/buffer/max_score/{score_fn_name}": max_score[i]
                })

            (levels, hrms), scores, ind_score_probs, total_score_probs, problem_probs, extras = jax.vmap(
                lambda problem_id: (
                    self.buffer_manager.get_problems(buffer, problem_id),
                    buffer["scores"][:, problem_id],
                    ind_score_weights[:, problem_id],
                    total_score_weights[problem_id],
                    problem_weights[problem_id],
                    self.buffer_manager.get_problems_extra(buffer, problem_id),
                )
            )(jnp.arange(buffer["size"]))

            # Proposition frequency
            # metrics[f"{prefix}/buffer/prop_frequency"] = plotly_prop_frequency(hrms, self.label_fn),

            # Some distributions based on the scores (sampling probabilities)
            prob_dict = {
                **{self.score_fn_names[i].lower(): ind_score_probs[:, i] for i in range(self.num_scores)},
                "total_score": total_score_probs,
                "total_score+staleness": problem_probs,
            }

            metrics.update({
                # Number of RMs and states distribution
                f"{prefix}/buffer/num_rms": plotly_xminigrid_num_rms(levels, hrms, prob_dict, get_level_sizes(levels)),
                f"{prefix}/buffer/problem_distribution": plotly_xminigrid_prob_distrib_sampling(levels, hrms, prob_dict),

                # Mutation distributions
                f"{prefix}/buffer/mutation_count": plotly_mutation_count(extras["mutation_ids"], prob_dict),
                f"{prefix}/buffer/mutation_category": plotly_mutation_category_count(extras["mutation_ids"], prob_dict),
                f"{prefix}/buffer/mutation_fraction": plotly_mutation_fraction(extras["mutation_rounds"], prob_dict),
                f"{prefix}/buffer/mutation_hindsight_lvl": plotly_mutation_hindsight_lvl(
                    hrms, extras["mutation_ids"], extras["mutation_args"], prob_dict
                ),
                f"{prefix}/buffer/num_mutations": plotly_num_mutations(extras["mutation_ids"]),

                # For random walk sampling parameters
                # f"{prefix}/buffer/epsilon": plotly_epsilon(hrms, score_probs),
            })

            for i in range(self.num_scores):
                score_fn_name = self.cfg.algorithm.score_functions[i].name.lower()
                metrics.update({
                    f"{prefix}/buffer/scores/{score_fn_name}": plotly_buffer_scores(scores[:, i], prob_dict, num_bins=50),
                    f"{prefix}/buffer/scores_mutated_problems/{score_fn_name}": plotly_buffer_scores(
                        scores[:, i], prob_dict, num_bins=50, mask=extras["mutation_rounds"] > 0,
                    ),
                    f"{prefix}/buffer/scores_nonmutated_problems/{score_fn_name}": plotly_buffer_scores(
                        scores[:, i], prob_dict, num_bins=50, mask=extras["mutation_rounds"] == 0,
                    ),
                })

        return metrics

    def _get_num_updates(self, runner_state: RunnerState) -> int:
        return int(runner_state.num_dr_updates + runner_state.num_replay_updates + runner_state.num_mutation_updates)

    def _get_eval_problems(self, runner_state: RunnerState) -> EvaluationProblem:
        return self._get_extended_eval_problems(runner_state, self.eval_loader.load())

    def _get_extended_eval_problems(self, runner_state: RunnerState, eval_problems: EvaluationProblem) -> EvaluationProblem:
        """
        Extends a set of evaluation problems with some obtained from the buffer.
        The concatenation order is important to decompose them later when logging the results!
        """
        if self.cfg.algorithm.num_high_score_eval_problems + self.cfg.algorithm.num_low_score_eval_problems > 0:
            return jax.tree_util.tree_map(
                lambda a, b, c: jnp.concatenate((a, b, c), axis=0),
                eval_problems,
                EvaluationProblem(*self.buffer_manager.get_highest_scored_problems(
                    runner_state.buffer, num_problems=self.cfg.algorithm.num_high_score_eval_problems,
                )),
                EvaluationProblem(*self.buffer_manager.get_lowest_scored_problems(
                    runner_state.buffer, num_problems=self.cfg.algorithm.num_low_score_eval_problems,
                )),
            )
        return eval_problems

    def _get_log_eval_metrics(
        self,
        eval_metrics: RolloutStats,
        rollouts: Rollout,
        runner_state: RunnerState,
        step: int,
        is_last_log: bool,
    ):
        log_metrics: dict[int, dict] = defaultdict(dict)

        # Add metrics from the buffer if using runner for evaluation only
        # (o.w. they are already logged for training)
        if self.cfg.mode == "evaluation":
            log_metrics[step].update(self._get_buffer_metrics(runner_state.buffer, prefix="eval"))

        # Rollouts for the rest of the problems
        num_eval_probs = self.eval_loader.get_num_problems()
        num_high_score_probs = min(self.cfg.algorithm.num_high_score_eval_problems, runner_state.buffer["size"])
        num_low_score_probs = min(self.cfg.algorithm.num_low_score_eval_problems, runner_state.buffer["size"])

        # Make sets of problems and get their metrics
        buffer_prefix = "eval" if self.cfg.mode == "evaluation" else "training"
        cum_probs = 0
        prob_sets = zip(
            ["eval", f"{buffer_prefix}/buffer/highest_score", f"{buffer_prefix}/buffer/lowest_score"],
            [num_eval_probs, num_high_score_probs, num_low_score_probs],
            [
                self.eval_loader.get_problem_names(),
                [f"high_score_{i}" for i in range(num_high_score_probs)],
                [f"low_score_{i}" for i in range(num_low_score_probs)],
            ]
        )

        for prefix, num_probs, prob_names in prob_sets:
            if num_probs > 0:
                next_cum_probs = cum_probs + num_probs
                log_metrics[step].update(self._get_log_eval_metrics_aux(
                    num_probs,
                    *jax.tree_util.tree_map(lambda x: x[cum_probs:next_cum_probs], (eval_metrics, rollouts)),
                    step=step,
                    is_last_log=is_last_log,
                    problem_names=prob_names,
                    prefix=prefix,
                )[step])
                cum_probs = next_cum_probs

        return log_metrics
