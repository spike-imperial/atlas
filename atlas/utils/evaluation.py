from typing import Dict, List, Tuple

import chex
from flax import struct
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from ..agents.types import ConditionedAgentState
from ..envs.common.env import Environment
from ..envs.common.level import Level
from ..envs.common.types import EnvParams
from ..eval_loaders.types import EvaluationProblem
from ..hrm.types import HRM, HRMState


class RolloutStats(struct.PyTreeNode):
    reward: chex.Array = jnp.asarray(0.0)
    disc_reward: chex.Array = jnp.asarray(0.0)
    length: chex.Array = jnp.asarray(0)
    is_task_completed: chex.Array = jnp.asarray(0, dtype=jnp.int32)


class Rollout(struct.PyTreeNode):
    states: chex.Array
    level: Level
    hrm: HRM
    hrm_states: HRMState
    length: chex.Array


def evaluate_agent(
    config: DictConfig,
    rng: chex.PRNGKey,
    train_state: TrainState,
    env: Environment,
    env_params: EnvParams,
    eval_problems: EvaluationProblem,
    num_eval_problems: int,
    cond_agent_state: ConditionedAgentState,
) -> Tuple[RolloutStats, Rollout]:
    def _rollout_fn(rollout_rng: chex.PRNGKey, eval_problem: EvaluationProblem, c_a_state: ConditionedAgentState):
        return jax.vmap(rollout, in_axes=(None, 0, None, None, None, None, None, None, None))(
            config,
            jax.random.split(rollout_rng, config.num_rollouts_per_problem),
            env,
            env_params,
            eval_problem.level,
            eval_problem.hrm,
            train_state,
            c_a_state,
            env_params.max_steps,
        )

    return jax.vmap(_rollout_fn)(
        jax.random.split(rng, num_eval_problems),
        eval_problems,
        cond_agent_state,
    )


def rollout(
    config: DictConfig,
    rng: chex.PRNGKey,
    env: Environment,
    env_params: EnvParams,
    level: Level,
    hrm: HRM,
    train_state: TrainState,
    init_cond_agent_state: ConditionedAgentState,
    max_rollout_length: int,
) -> Tuple[RolloutStats, Rollout]:
    def step(carry, _):
        rng, stats, timestep, prev_action, prev_reward, cond_agent_state = carry
        rng, _rng = jax.random.split(rng)

        # Add batch and time dim
        _timestep, _prev_action, _prev_reward = jax.tree_util.tree_map(
            lambda x: x[None, None, ...],
            (timestep, prev_action, prev_reward)
        )

        cond_agent_state, outputs = train_state.apply_fn(
            train_state.params,
            observation=_timestep.observation,
            done=_timestep.last(),
            hrm=_timestep.extras.hrm,
            hrm_state=_timestep.extras.hrm_state,
            prev_action=_prev_action,
            prev_reward=_prev_reward,
            cond_agent_state=cond_agent_state,
        )
        _, (dist, _) = outputs

        if config.use_greedy:
            action = dist.mode().squeeze()
        else:
            action = dist.sample(seed=_rng).squeeze() # policy type here

        timestep = env.step(env_params, timestep, action)

        mask = 1 - stats.is_task_completed
        return (
            rng,
            RolloutStats(
                reward=stats.reward + mask * timestep.reward,
                disc_reward=stats.disc_reward + mask * (config.gamma ** stats.length) * timestep.reward,
                length=stats.length + mask,
                is_task_completed=stats.is_task_completed | jnp.asarray(timestep.extras.task_completed, dtype=jnp.int32),
            ),
            timestep,
            action,
            timestep.reward,
            cond_agent_state,
        ), (
            timestep.state,
            timestep.extras.hrm_state,
        )

    # Split into an rng for resetting the env and one for selecting actions in the rollout
    reset_rng, rollout_rng = jax.random.split(rng, 2)

    # Perform the rollout collecting statistics and the step-by-step states and HRM states
    init_timestep = env.reset(reset_rng, env_params, level, hrm=hrm)
    (_, rollout_stats, _, _, _, _), (states, hrm_states) = jax.lax.scan(
        f=step,
        init=(
            rollout_rng,
            RolloutStats(),
            init_timestep,
            jnp.asarray(0),
            jnp.asarray(0),
            jax.tree_util.tree_map(lambda x: x[None, ...], init_cond_agent_state),
        ),
        xs=None,
        length=max_rollout_length,
    )

    # Append the first state and HRM state to the rollout
    states = jax.tree_util.tree_map(
        lambda x, y: jnp.concat((jnp.asarray(x)[None, ...], y)),
        init_timestep.state,
        states
    )
    hrm_states = jax.tree_util.tree_map(
        lambda x, y: jnp.concat((x[None, ...], y)),
        init_timestep.extras.hrm_state,
        hrm_states,
    )

    return rollout_stats, Rollout(states, level, hrm, hrm_states, rollout_stats.length)


def cvar(
    scores: chex.Array,
    unbiased_scores: chex.Array,
    percentages: List = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
) -> Tuple[Dict[int, chex.Array], Dict[int, chex.Array]]:
    """
    Computes the conditional value at risk (CVaR) as described in "No Regrets: Investigating
    and Improving Regret Approximations for Curriculum Discovery" (Section 6).

    The method takes the fraction of levels for which the performance was the worst according
    to the `scores` argument. Then returns the average performance for this fraction according
    to the `scores` and the `unbiased_scores`, where the latter is used to provide an unbiased
    estimate of the performance.

    Args:
        - scores: scores obtained by evaluating the model with a given seed on a set of problems.
        - unbiased_scores: scores obtained by evaluating the model on the same set of problems
            but with a different seed.

    Returns:
        Dictionaries mapping a fraction of the levels into the associated biased and unbiased scores.
    """
    indices = jnp.argsort(scores)
    sorted_scores = scores[indices]
    sorted_unbiased_scores = unbiased_scores[indices]

    biased = dict()
    unbiased = dict()

    for p in percentages:
        num = int(len(scores) * p / 100)
        biased[p] = jnp.sum(sorted_scores[:num]) / num
        unbiased[p] = jnp.sum(sorted_unbiased_scores[:num]) / num

    return biased, unbiased
