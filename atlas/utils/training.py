from typing import Dict, Tuple, Union

import chex
from flax import struct
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from ..agents.types import ConditionedAgentState
from ..envs.common.env import Environment
from ..envs.common.types import EnvParams, Timestep
from ..hrm.types import HRM, HRMState


class Transition(struct.PyTreeNode):
    done: chex.Array
    action: chex.Array
    value: chex.Array
    reward: chex.Array
    log_prob: chex.Array
    obs: chex.Array

    # track some data about the policy to compute ued scores
    #  - entropy: -\sum_a \pi(s,a) * log \pi(s,a)
    #  - min_margin: difference between highest prob. and second highest
    #  - least_confidence: 1 - max_a \pi(s, a)
    entropy: chex.Array
    min_margin: chex.Array
    least_confidence: chex.Array

    # for rnn policy
    prev_action: chex.Array
    prev_reward: chex.Array

    # hrm information
    hrm: HRM
    hrm_state: HRMState

    # task completion information
    # different from `done`, which captures both task completion and truncation
    # (getting to maximum steps per episode)
    task_completed: chex.Array
    hrm_completion: chex.Array


def collect_trajectories(
    rng: chex.PRNGKey,
    env: Environment,
    env_params: EnvParams,
    train_state: TrainState,
    init_timestep: Timestep,
    init_action: chex.Array,
    init_reward: chex.Array,
    init_agent: ConditionedAgentState,
    num_steps: int,
) -> Tuple[
    Tuple[chex.PRNGKey, Timestep, chex.Array, chex.Array, ConditionedAgentState, chex.Array],
    Transition,
]:
    def _env_step(carry, _):
        rng, prev_timestep, prev_action, prev_reward, agent_state = carry
        rng, sample_rng = jax.random.split(rng)

        # Doing 1 step at a time so add a time dimension
        _prev_timestep, _prev_action, _prev_reward = jax.tree_util.tree_map(
            lambda x: x[:, None], (prev_timestep, prev_action, prev_reward)
        )

        # Determine next agent state (RNN state) and output of the networks
        # - outputs: (conditioner_state, agent_policy_state), (conditioner_out, agent_policy_out)
        next_agent, outputs = train_state.apply_fn(
            train_state.params,
            _prev_timestep.observation,
            _prev_timestep.last(),
            _prev_timestep.extras.hrm,
            _prev_timestep.extras.hrm_state,
            _prev_action,
            _prev_reward,
            agent_state,
        )
        _, (dist, values) = outputs

        # Choose the action (squeeze `seq_len` dimension where possible)
        action, log_prob = dist.sample_and_log_prob(seed=sample_rng)
        action, log_prob, values = (
            action.squeeze(1),
            log_prob.squeeze(1),
            jax.tree_util.tree_map(lambda x: x.squeeze(1), values),
        )

        # Perform the step in the environment
        timestep = jax.vmap(env.step, in_axes=(None, 0, 0))(
            env_params, prev_timestep, action
        )

        # Form the transition that will constitute part of the trajectory
        transition = Transition(
            done=timestep.last(),
            action=action,
            value=values,
            reward=timestep.reward,
            log_prob=log_prob,
            obs=prev_timestep.observation,
            entropy=dist.entropy().squeeze(1),
            min_margin=jnp.abs(jnp.diff(jnp.partition(dist.probs, kth=-2)[..., -2:])).squeeze(1).squeeze(1),
            least_confidence=1 - jnp.max(dist.probs, axis=-1).squeeze(1),
            prev_action=prev_action,
            prev_reward=prev_reward,
            hrm=prev_timestep.extras.hrm,
            hrm_state=prev_timestep.extras.hrm_state,
            task_completed=timestep.extras.task_completed,
            hrm_completion=timestep.extras.hrm_completion,
        )

        return (
            rng,
            timestep,
            action,
            timestep.reward,
            next_agent,
        ), transition

    # transitions: [seq_len, batch_size, ...]
    (rng, last_timestep, last_action, last_reward, last_agent), transitions = jax.lax.scan(
        _env_step,
        init=(rng, init_timestep, init_action, init_reward, init_agent),
        xs=None,
        length=num_steps,
    )

    # Calculate value of the last step for bootstrapping
    # pad from [B,...] to [B, T, ...] ; except for states (only initial carry)
    _timestep, _action, _reward = jax.tree_util.tree_map(
        lambda x: x[:, None], (last_timestep, last_action, last_reward)
    )

    _, (_, (_, last_values)) = train_state.apply_fn(
        train_state.params,
        observation=_timestep.observation,
        done=_timestep.last(),
        hrm=_timestep.extras.hrm,
        hrm_state=_timestep.extras.hrm_state,
        prev_action=_action,
        prev_reward=_reward,
        cond_agent_state=last_agent,
    )

    return (
        rng,
        last_timestep,
        last_action,
        last_reward,
        last_agent,
        jax.tree_util.tree_map(lambda x: x.squeeze(1), last_values),
    ), transitions


def calculate_gae(
    gamma: float,
    gae_lambda: float,
    last_value: chex.Array,
    values: chex.Array,
    rewards: chex.Array,
    dones: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    def _calculate_gae_timestep(carry, xs):
        gae, next_value = carry
        value, reward, done = xs

        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _calculate_gae_timestep,
        init=(jnp.zeros_like(last_value), last_value),
        xs=(values, rewards, dones),
        reverse=True,
        unroll=16,
    )

    # advantages and values (Q)
    return advantages, advantages + values


def update_network(
    rng: chex.PRNGKey,
    init_train_state: TrainState,
    init_agent: ConditionedAgentState,
    transitions: Transition,
    advantages: Dict[str, chex.Array],
    targets: Dict[str, chex.Array],
    num_envs: int,
    num_minibatches: int,
    num_update_epochs: int,
    clip_eps: float,
    vf_coef: Union[float, Dict],
    ent_coef: float,
    update_grad: bool = True,
) -> Tuple[chex.PRNGKey, TrainState, dict]:
    # Move the `done` indicator forward so that it is checked as for
    # when the trajectories are collected
    transitions = transitions.replace(
        done=jnp.roll(transitions.done, shift=1, axis=0).at[0].set(False)
    )

    # Change dimensions from [seq_len, batch_size, ...] to [batch_size, seq_len, ...]
    batch = jax.tree_util.tree_map(
        lambda x: x.swapaxes(0, 1),
        (transitions, targets, advantages)
    )

    def _update_epoch(update_state: Tuple, _):
        def _update_minibatch(train_state: TrainState, minibatch: Tuple):
            init_agent, transitions, targets, advantages = minibatch
            return ppo_update_fn(
                train_state,
                transitions,
                init_agent,
                advantages,
                targets,
                clip_eps,
                vf_coef,
                ent_coef,
                update_grad,
            )

        rng, train_state = update_state

        # Put together agent state and transition-batch data
        agent_batch = (init_agent, *batch)

        # Shuffle
        rng, rng_perm = jax.random.split(rng)
        permutation = jax.random.permutation(rng_perm, num_envs)
        shuffled_batch = jax.tree_util.tree_map(
            lambda x: jnp.take(x, permutation, axis=0),
            agent_batch
        )

        # Split into minibatches
        minibatches = jax.tree_util.tree_map(  # [num_minibatches, minibatch_size, ...]
            lambda x: jnp.reshape(x, (num_minibatches, -1, *x.shape[1:])),
            shuffled_batch
        )

        #  Perform the update
        next_train_state, update_info = jax.lax.scan(_update_minibatch, train_state, minibatches)
        return (rng, next_train_state), update_info

    (rng, train_state), loss_info = jax.lax.scan(
        _update_epoch,
        init=(rng, init_train_state),
        xs=None,
        length=num_update_epochs,
    )

    # Averaging over minibatches then over epochs
    loss_info = jtu.tree_map(lambda x: x.mean(-1).mean(-1), loss_info)

    return rng, train_state, loss_info


def ppo_update_fn(
    train_state: TrainState,
    transitions: Transition,
    init_agent: ConditionedAgentState,
    advantages: chex.Array,
    targets: chex.Array,
    clip_eps: float,
    vf_coef: Union[float, Dict],
    ent_coef: float,
    update_grad: bool,
):
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def _value_loss_fn(type: str, last_value) -> chex.Array:
        value_pred_clipped = transitions.value[type] + (last_value[type] - transitions.value[type]).clip(-clip_eps, clip_eps)
        value_loss = jnp.square(last_value[type] - targets[type])
        value_loss_clipped = jnp.square(value_pred_clipped - targets[type])
        return 0.5 * jnp.maximum(value_loss, value_loss_clipped).mean()

    def _loss_fn(params):
        _, (_, (dist, value)) = train_state.apply_fn(
            params,
            observation=transitions.obs,
            done=transitions.done,
            hrm=transitions.hrm,
            hrm_state=transitions.hrm_state,
            prev_action=transitions.prev_action,
            prev_reward=transitions.prev_reward,
            cond_agent_state=init_agent,
        )
        log_prob = dist.log_prob(transitions.action)

        value_losses = {k: _value_loss_fn(k, value) for k in value.keys()}

        value_losses_sum = 0.0
        if isinstance(vf_coef, float):
            for v in value_losses.values():
                value_losses_sum += vf_coef * v
        else:
            for k, v in value_losses.items():
                value_losses_sum += vf_coef[k] * v

        ratio = jnp.exp(log_prob - transitions.log_prob)
        actor_loss1 = advantages * ratio
        actor_loss2 = advantages * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()

        entropy = dist.entropy().mean()
        total_loss = actor_loss + value_losses_sum - ent_coef * entropy
        return total_loss, (value_losses, actor_loss, entropy)

    (loss, (vloss, aloss, entropy)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params)

    if update_grad:
        train_state = train_state.apply_gradients(grads=grads)

    update_info = {
        "total_loss": loss,
        "actor_loss": aloss,
        "entropy": entropy,
        **{f"{k}_loss": v for k, v in vloss.items()}
    }
    return train_state, update_info


def collect_trajectories_and_learn(
    rng: chex.PRNGKey,
    env: Environment,
    env_params: EnvParams,
    init_train_state: TrainState,
    init_timesteps: Timestep,
    init_actions: chex.Array,
    init_rewards: chex.Array,
    init_agents: ConditionedAgentState,
    num_envs: int,
    num_steps: int,
    num_outer_steps: int,
    gamma: float,
    gae_lambda: float,
    num_minibatches: int,
    num_update_epochs: int,
    clip_eps: float,
    vf_coef: Union[float, Dict],
    ent_coef: float,
    update_grad: bool,
    advantage_src: str,
) -> Tuple[
    Tuple[chex.PRNGKey, TrainState, Timestep, chex.Array, chex.Array, ConditionedAgentState], 
    Tuple[Transition, chex.Array, dict]
]:
    """
    Aggregates all previous methods: sampling trajectories, computing advantages and
    performing updates. 
    
    Introduces a hyperparameter `num_outer_steps` specifying how many times the combination 
    of the three methods is sequentially performed (each iteration takes the previous iteration
    state as input). This is introduced for the UED case so that we can run an entire rollout
    for a <level, HRM> combination (see Appendix G.1, https://arxiv.org/pdf/2402.16801).

    Based on https://github.com/DramaCow/jaxued/blob/main/examples/craftax/craftax_plr.py#L224
    """
    def step(carry, _):
        rng, train_state, init_timestep, init_action, init_reward, init_agent = carry

        # Collect trajectories [T, B, ...]
        (
            (rng, last_timestep, last_action, last_reward, last_agent, last_value),
            transitions
        ) = collect_trajectories(
            rng,
            env,
            env_params,
            train_state,
            init_timestep,
            init_action,
            init_reward,
            init_agent,
            num_steps,
        )

        # Calculate advantages [PPO-specific]
        adv_targets = jax.tree_util.tree_map(
            lambda lv, v: calculate_gae(gamma, gae_lambda, lv, v, transitions.reward, transitions.done),
            last_value, transitions.value
        )
        advantages = {k: v[0] for k, v in adv_targets.items()}
        targets = {k: v[1] for k, v in adv_targets.items()}

        # Update the network [PPO-specific, PPO re-runs the network]
        rng, train_state, metrics = update_network(
            rng,
            train_state,
            init_agent,
            transitions,
            advantages[advantage_src],
            targets,
            num_envs,
            num_minibatches,
            num_update_epochs,
            clip_eps,
            vf_coef,
            ent_coef,
            update_grad,
        )

        return (rng, train_state, last_timestep, last_action, last_reward, last_agent), (transitions, advantages, metrics)

    init_carry = (rng, init_train_state, init_timesteps, init_actions, init_rewards, init_agents)
    carry, rollouts = jax.lax.scan(step, init_carry, None, length=num_outer_steps)
    
    (transitions, advantages, metrics) = rollouts
    transitions, advantages = jax.tree_map(lambda x: jnp.concatenate(x, axis=0), (transitions, advantages))
    metrics = jax.tree_map(lambda x: x[-1], metrics)

    return carry, (transitions, advantages, metrics)


def compute_max_returns(transitions: Transition):
    """
    Copied from:
    https://github.com/DramaCow/jaxued/blob/c3350ab6708c87d15648a8b29498cf979fab494a/src/jaxued/utils.py
    """
    _, max_returns, _ = accumulate_rollout_stats(
        transitions.done, transitions.reward, time_average=False
    )
    return max_returns


def compute_task_completions(transitions: Transition):
    """
    Computes the number of times a task has been successfully completed
    in a rollout and the number of episodes in the rollout.
    """
    mean, _, episode_count = accumulate_rollout_stats(
        transitions.done, transitions.task_completed.astype(jnp.int32), time_average=False
    )
    return jnp.asarray(mean * episode_count, jnp.int32), episode_count


def compute_hrm_completion_sum(transitions: Transition):
    """
    Computes the average completion score of an HRM.
    """
    return jnp.sum(transitions.done * transitions.hrm_completion, axis=0), jnp.sum(transitions.done, axis=0)


def compute_mean_disc_return(transitions: Transition, discount: float):
    mean, _, _ = accumulate_rollout_stats(
        transitions.done, transitions.reward, time_average=False, discount=discount,
    )
    return mean


def accumulate_rollout_stats(dones, metrics, *, time_average, discount=1):
    """
    Copied from:
    https://github.com/DramaCow/jaxued/blob/c3350ab6708c87d15648a8b29498cf979fab494a/src/jaxued/utils.py
    """

    def iter(carry, input):
        sum_val, max_val, accum_val, step_count, episode_count = carry
        done, step_val = input

        accum_val = jax.tree_map(lambda x, y: x + y, accum_val, jnp.pow(discount, step_count) * step_val)
        step_count += 1

        if time_average:
            # val = jax.tree_map(lambda x, b: jax.lax.select(b, x / step_count, x), accum_val, time_average)
            val = jax.tree_map(lambda x: x / step_count, accum_val)
        else:
            val = accum_val

        sum_val = jax.tree_map(lambda x, y: x + done * y, sum_val, val)
        max_val = jax.tree_map(
            lambda x, y: (1 - done) * x + done * jnp.maximum(x, y), max_val, val
        )

        episode_count += done

        accum_val = jax.tree_map(lambda x: (1 - done) * x, accum_val)
        step_count = (1 - done) * step_count

        return (sum_val, max_val, accum_val, step_count, episode_count), None

    batch_size = dones.shape[1]
    zeros = jax.tree_map(lambda x: jnp.zeros_like(x[0]), metrics)
    (sum_val, max_val, _, _, episode_count), _ = jax.lax.scan(
        iter,
        (
            zeros,
            zeros,
            zeros,
            jnp.zeros(batch_size, dtype=jnp.uint32),
            jnp.zeros(batch_size, dtype=jnp.uint32),
        ),
        (dones, metrics),
    )

    mean_val = jax.tree_map(lambda x: x / jnp.maximum(episode_count, 1), sum_val)

    return mean_val, max_val, episode_count
