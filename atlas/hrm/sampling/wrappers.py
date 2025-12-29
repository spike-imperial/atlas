from abc import abstractmethod
from typing import Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp

from .common import HRMSampler
from ..ops import get_num_rm_states
from ..types import HRM


class DummyHRMSamplerWrapper(HRMSampler):
    """
    A wrapper that does nothing.
    """
    def __init__(self, sampler: HRMSampler, **kwargs: dict):
        super().__init__(
            sampler._max_num_rms,
            sampler._max_num_states,
            sampler._max_num_edges,
            sampler._max_num_literals,
            sampler._alphabet_size,
        )
        self._sampler = sampler

    def sample(self, key: chex.PRNGKey, extras: Optional[Dict] = None) -> HRM:
        return self._sampler(key, extras)

    def unwrapped(self) -> "HRMSampler":
        return self._sampler.unwrapped()


class RewardHRMSamplerWrapper(HRMSampler):
    def __init__(self, sampler: HRMSampler, additive: bool = False):
        super().__init__(
            sampler._max_num_rms,
            sampler._max_num_states,
            sampler._max_num_edges,
            sampler._max_num_literals,
            sampler._alphabet_size,
        )
        self._sampler = sampler
        self._additive = additive

    def sample(self, key: chex.PRNGKey, extras: Optional[Dict] = None) -> HRM:
        hrm = self._sampler(key, extras)

        rewards = self._get_shaped_rewards(hrm)
        if self._additive:
            rewards += hrm.rewards

        return hrm.replace(rewards=rewards)

    @abstractmethod
    def _get_shaped_rewards(self, hrm: HRM) -> chex.Array:
        raise NotImplementedError

    def unwrapped(self) -> "HRMSampler":
        return self._sampler.unwrapped()


class SimpleRewardHRMSamplerWrapper(RewardHRMSamplerWrapper):
    """
    A wrapper that establishes a fixed reward for all transitions,
    and another one for self-transitions.
    """

    def __init__(
        self,
        sampler: HRMSampler,
        additive: bool = False,
        tx_reward: float = 1.0,
        self_tx_reward: float = -0.01,
        **kwargs: dict,
    ):
        """
        Args:
          sampler: the wrapped sampler
          additive: whether the produced shaped rewards are added to the original ones
          tx_reward: rewards for each transition to a different state
          self_tx_reward: rewards for each self-transition
        """
        super().__init__(sampler, additive)
        self._tx_reward = tx_reward
        self._self_tx_reward = self_tx_reward

    def _get_shaped_rewards(self, hrm: HRM) -> chex.Array:
        # Reward for all non-self transitions (the `max` is used to determine
        # whether there is at least one edge between two states)
        tx_rewards = self._tx_reward * jnp.max(hrm.calls >= 0, axis=-1)

        # Reward for all self-transitions (sum will be broadcasted)
        self_tx_rewards = self._self_tx_reward * jnp.diag(
            jnp.ones((self._max_num_states,))
        )

        return tx_rewards + self_tx_rewards


class ShortestPathRewardHRMSamplerWrapper(RewardHRMSamplerWrapper):
    """
    A wrapper that determines the rewards based on the distance to each
    reward machine's accepting state. The potential of each state `u` is
    determined as `\phi(u) = max_num_states - dist(u, uA)`. The reward
    is given using the potential-based reward_shaping: `\gamma * \phi(u') - \phi`.

    Original algorithm: Furelos-Blanco et al. (2020). "Induction of Subgoal
    Automata for Reinforcement Learning".

    Shortest-path algorithm from: https://jax.quantecon.org/short_path.html.
    """

    def __init__(
        self,
        sampler: HRMSampler,
        gamma: float,
        additive: bool = False,
        neginf: float = -10,
        max_iterations: int = 500,
        **kwargs: dict,
    ):
        """
        Args:
          sampler: the wrapped sampler
          gamma: the discount factor used in the RL algorithm
          additive: whether the produced shaped rewards are added to the original ones
          neginf: value for which -inf values are substituted
          max_iterations: the maximum number of iterations taken in the shortest-path algorithm
        """
        super().__init__(sampler, additive)
        self._gamma = gamma
        self._neginf = neginf
        self._max_iterations = max_iterations

    def _get_shaped_rewards(self, hrm: HRM) -> chex.Array:
        # Obtain the edge matrix and cost matrix to run the shortest path algorithm on
        edges, costs = self._get_edge_and_cost_matrices(hrm)

        # The rewards are determined for each RM separately from each other
        return jax.vmap(self._get_rm_rewards, in_axes=(None, 0, 0, 0))(
            hrm, jnp.arange(self._max_num_rms), edges, costs
        )

    def _get_rm_rewards(self, hrm: HRM, rm_id: chex.Array, edges: chex.Array, costs: chex.Array) -> chex.Array:
        # Compute the potential by computing the distances
        # The -inf are substituted here by some number that the RL
        # algorithm can deal with
        potentials = get_num_rm_states(hrm, rm_id) - self._get_distances(costs)
        potentials = jnp.nan_to_num(potentials, neginf=self._neginf)

        # Apply potential-based reward shaping formula
        rewards = self._gamma * edges * potentials - potentials[:, jnp.newaxis]
        rewards *= edges  # mask again
        return rewards

    def _get_distances(self, costs: chex.Array) -> chex.Array:
        init_distances = jnp.zeros(self._max_num_states)

        def cond_fun(values):
            i, _, break_cond = values
            return ~break_cond & (i < self._max_iterations)

        def body_fun(values):
            # Define the body function of while loop
            i, distances, _ = values

            # Update distances and break condition
            next_distances = jnp.min(costs + distances, axis=1)
            break_cond = jnp.allclose(next_distances, distances)

            # Return next iteration values
            return i + 1, next_distances, break_cond

        return jax.lax.while_loop(
            cond_fun, body_fun, init_val=(0, init_distances, False)
        )[1]

    def _get_edge_and_cost_matrices(self, hrm: HRM) -> Tuple[chex.Array, chex.Array]:
        out_edge_mask = jnp.max(hrm.calls >= 0, axis=-1)

        # The edges we are interested in deriving rewards for
        edges = jnp.logical_or(out_edge_mask, jnp.diag(jnp.ones(self._max_num_states)))

        # The costs of the matrix (self-loops are omitted except
        # for the accepting state)
        inf_mask = jnp.logical_not(
            jnp.logical_or(
                out_edge_mask,
                jnp.diag(
                    jnp.concat((jnp.zeros(self._max_num_states - 1), jnp.ones((1,))))
                ),
            )
        )
        costs = out_edge_mask + inf_mask * jnp.inf

        return edges, costs
