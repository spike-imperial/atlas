from typing import Callable, Dict, List, Optional

import chex
import jax
import jax.numpy as jnp

from ..utils.training import Transition, accumulate_rollout_stats


def get_score_fn(name: str, args: Optional[Dict]) -> Callable[
    [Transition, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array],
    chex.Array
]:
    """
    Returns the score function associated with the specified name.
    """
    def _score_fn(
        transitions: Transition,
        max_returns: chex.Array,
        return_momentums: chex.Array,
        advantages: chex.Array,
        task_completions: chex.Array,
        hrm_completions: chex.Array,
        episode_count: chex.Array,
    ):
        value_src = args.get("value_src", "critic") if args else "critic"
        if name == "MaxMC":
            return max_mc(transitions.done, transitions.value[value_src], max_returns)
        if name == "pvl":
            return positive_value_loss(transitions.done, advantages[value_src])
        if name == "entropy":
            return policy_entropy(transitions.done, transitions.entropy)
        if name == "entropy_neg":
            return policy_entropy_neg(transitions.done, transitions.entropy)
        if name == "min_margin":
            return policy_min_margin(transitions.done, transitions.min_margin)
        if name == "least_confidence":
            return policy_least_confidence(transitions.done, transitions.least_confidence)
        if name == "gae":
            return gae_score(transitions.done, advantages[value_src])
        if name == "l1vl":
            return l1_value_loss(transitions.done, advantages[value_src])
        if name == "learnability":
            peak_at = args.get("learnability_peak", 0.5) if args else 0.5
            return learnability(task_completions, episode_count, peak_at)
        if name == "hrm_completion":
            return learnability(hrm_completions, episode_count, peak_at=0.5)
        if name == "return_momentum":
            return return_momentums

        raise RuntimeError(f"Error: Score function '{name}' does not exist.")

    return _score_fn


def get_scores_fn(names: List[str], args: List[Dict]) -> Callable[
    [Transition, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array],
    chex.Array
]:
    """
    Returns the function that upon called returns an array containing the
    scores for different functions.
    """
    def _scores_fn(
        transitions: Transition,
        max_returns: chex.Array,
        return_momentums: chex.Array,
        advantages: chex.Array,
        task_completions: chex.Array,
        hrm_completions: chex.Array,
        episode_count: chex.Array,
    ):
        return jnp.array([
            get_score_fn(n, a)(
                transitions, max_returns, return_momentums, advantages, task_completions, hrm_completions, episode_count
            )
            for n, a in zip(names, args)
        ]).swapaxes(0, 1)

    return _scores_fn


def max_mc(dones: chex.Array, values: chex.Array, max_returns: chex.Array, incomplete_value=-jnp.inf) -> chex.Array:
    return _compute_cumulative_score_metric(dones, max_returns[None, :] - values, incomplete_value)


def positive_value_loss(dones: chex.Array, advantages: chex.Array, incomplete_value=-jnp.inf) -> chex.Array:
    return _compute_cumulative_score_metric(dones, jnp.maximum(advantages, 0), incomplete_value)


def policy_entropy(dones: chex.Array, entropies: chex.Array, incomplete_value=-jnp.inf) -> chex.Array:
    # Higher scores for high policy entropy
    return _compute_cumulative_score_metric(dones, entropies, incomplete_value)


def policy_entropy_neg(dones: chex.Array, entropies: chex.Array, incomplete_value=-jnp.inf) -> chex.Array:
    # Higher scores for low policy entropy (like the original PLR paper)
    # Note that the paper omits the `minus` sign employed for
    # entropy (our computation of it in the trajectory collection
    # includes the sign, so we need to negate it here).
    return _compute_cumulative_score_metric(dones, -entropies, incomplete_value)


def policy_min_margin(dones: chex.Array, min_margins: chex.Array, incomplete_value=-jnp.inf) -> chex.Array:
    return _compute_cumulative_score_metric(dones, min_margins, incomplete_value)


def policy_least_confidence(dones: chex.Array, least_confidences: chex.Array, incomplete_value=-jnp.inf) -> chex.Array:
    return _compute_cumulative_score_metric(dones, least_confidences, incomplete_value)


def gae_score(dones: chex.Array, advantages: chex.Array, incomplete_value=-jnp.inf) -> chex.Array:
    return _compute_cumulative_score_metric(dones, advantages, incomplete_value)


def l1_value_loss(dones: chex.Array, advantages: chex.Array, incomplete_value=-jnp.inf) -> chex.Array:
    # Also known as 'GAE magnitude'
    return _compute_cumulative_score_metric(dones, jnp.abs(advantages), incomplete_value)


def learnability(task_completions: chex.Array, episode_count: chex.Array, peak_at: float) -> chex.Array:
    """
    Returns the learnability score described in "No Regrets: Investigating and Improving Regret
    Approximations for Curriculum Discovery", i.e. success_rate * (1 - success_rate).
    """
    p = task_completions / jnp.maximum(episode_count, 1)
    return _learnability_with_peak(p, peak_at)


def _compute_cumulative_score_metric(dones, metric, incomplete_value=-jnp.inf):
    mean_scores, _, episode_count = accumulate_rollout_stats(
        dones, metric, time_average=True
    )
    return jnp.where(episode_count > 0, mean_scores, incomplete_value)


def _learnability_with_peak(p: chex.Array, peak_at: float = 0.5) -> chex.Array:
    """
    Generalize the p(1-p) function to peak at arbitrary points using piecewise quadratic interpolation.

    Args:
        p: input values in range [0, 1].
        peak_at: location where the function should achieve its maximum.

    Returns: function values
    """
    peak_value = 0.25

    # Handle the standard case directly
    def standard_case():
        return p * (1.0 - p)

    # For x <= peak_at
    def left_piece():
        a_left = -peak_value / (peak_at ** 2)
        b_left = -2.0 * a_left * peak_at
        return a_left * (p ** 2) + b_left * p

    # For x > peak_at
    def right_piece():
        a_right = -peak_value / ((1.0 - peak_at) ** 2)
        b_right = -2.0 * a_right * peak_at
        c_right = peak_value - a_right * (peak_at ** 2) - b_right * peak_at
        return a_right * (p ** 2) + b_right * p + c_right

    # Use JAX's conditional for scalar inputs
    score = jax.lax.cond(
        jnp.isclose(peak_at, 0.5),
        standard_case,
        lambda: jax.lax.select(p <= peak_at, left_piece(), right_piece())
    )

    return jnp.maximum(score, 0)  # maximum added for precision problems
