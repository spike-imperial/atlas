from typing import Dict, List, Optional

import chex
from chex import dataclass
import jax
import jax.numpy as jnp
import numpy as np

from .common import HRMSampler
from ..types import HRM


@dataclass
class RandomWalkHRMSamplerExtras:
    rng: chex.PRNGKey      # the key used to generate the HRM (passed to the sample function)


class RandomWalkHRMSampler(HRMSampler):
    """
    Samples an HRM via Random Walk with uniform weights.
    """

    def __init__(
        self,
        max_num_rms: int,
        max_num_states: int,
        max_num_edges: int,
        max_num_literals: int,
        alphabet_size: int,
        alphabet: List[str],
        enforce_mutex: bool = True,
        enforce_sequentiality: bool = True,
        splittiness: float = 0.5,
        avg_state_connectivity: int = 2,
        use_transition_compat_matrix: bool = False,
        use_call_compat_matrix: bool = False,
        eps: float = 1.0,
        max_num_gen_rms: Optional[int] = None,
        max_num_gen_states: Optional[int] = None,
        reward_shaping: bool = False,
        gamma: Optional[float] = None,
        **kwargs: dict,
    ):
        """
        max_num_rms: The maximum number of RMs that may appear in the sampled HRM
        max_num_states: The maximum number of states to appear per RM
        max_num_edges: Number of edges fixed at 1
        max_num_literals: The maximum number of literals per transition
        alphabet_size: The number of propositions in the alphabet
        alphabet: The set of propositions over which the HRM is defined (used for the compatibility matrices)
        enforce_mutex: Enforce mututal exclusivity of the formulas of neighbouring outgoing transitions
            by adding the negation of the respective other proposition, contributes to determinism
        enforce_sequentiality: Forces the sampler to only sample sequentially set up HRMs
        splittiness: Determines how likely the random walk is to restart once it has reached the terminal state, i.e.
            a high value will result in higher liklihood of branching behaviour
        avg_state_connectivity: Determines the average number of outgoing edges from a state (the higher, the slower
            but more complex the HRMs)
        use_transition_compat_matrix: Whether to use a matrix that constrains what propositions can appear on
            neighboring transitions.
        use_call_compat_matrix: Whether to use a matrix that constrains what propositions can appear on calls.
        min_eps_log: Minimum value for the discount factor for the transition probability to the terminal state,
            (in log10 scale) i.e. the lower the factor the more complex the HRM up to `avg_state_connectivity`.
        max_eps_log: Maximum value for the epsilon factor (see above) in log10 scale.
        max_num_gen_rms: Maximum number of generated RMs in the hierarchy
        max_num_gen_states: Maximum number of generated states per RM
        reward_shaping: Whether to determine rewards using the random walk (if False, a reward of 1 is given for
            all transitions to accepting states)
        gamma: The MDP discount factor used for the reward shaping.
        """
        assert max_num_rms >= 1, "The minimum number of RMs in an HRM produced by the sampler is 1"
        assert max_num_states >= 2, "The minimum number of states per RM in an HRM produced by the sampler is 2"
        assert max_num_edges == 1
        assert 1 >= splittiness >= 0, "Splittiness mustbe a value between 0  and 1"
        assert ((not (splittiness > 0)) | enforce_sequentiality), "Currently splittiness is only supported for sequential RMs"
        assert alphabet_size >= 2, "The minimum number of propositions in an HRM produced by the sampler is 2"
        assert avg_state_connectivity >= 1, "The minimum average connectivity is 1"
        assert max_num_literals >= 1, "Number of literals per transition must be at least 1"
        assert max_num_gen_rms is None or (max_num_gen_rms <= max_num_rms), "The maximimum number of generated RMs must be lower than the maximum of number RMs in the HRM"
        assert max_num_gen_states is None or (max_num_gen_states <= max_num_states), "The maximimum number of generated states must be lower than the maximum number of states per RM"
        assert not reward_shaping or gamma is not None, "The discount cannot be None if the reward shaping is used"

        super().__init__(max_num_rms, max_num_states, max_num_edges, max_num_literals, alphabet_size)

        self._max_num_non_root_rms = self._max_num_rms - 1
        self._enforce_mutex = enforce_mutex
        self._enforce_sequentiality = enforce_sequentiality
        self._enforce_dag = splittiness > 0
        self._splittiness = splittiness
        self._avg_connectivity = (1 if enforce_sequentiality else avg_state_connectivity) + (avg_state_connectivity-1)*(splittiness > 0)
        self._eps = eps
        self._max_num_gen_rms = max_num_gen_rms
        self._max_num_gen_states = max_num_gen_states
        self._reward_shaping = reward_shaping
        self._gamma = gamma

        # Matrix that defines propositions that cannot appear on neighboring outgoing transitions.
        # E.g., in Minigrid, `front_ball_green` is a special case of `front_ball`, hence they
        # cannot be on different outgoing transitions from the same state.
        if use_transition_compat_matrix:
            self._transition_compat_matrix = self._get_transition_compat_matrix(alphabet)
        else:
            self._transition_compat_matrix = jnp.ones((self._alphabet_size, self._alphabet_size)) - jnp.eye(self._alphabet_size)
        self._transition_compat_matrix = 1 - self._transition_compat_matrix

        # Matrix that defines predicate compatibility for call conditions, e.g. front_ball front_tile
        # is not compatible (entry 0) but front_ball carrying_tile is compatible (entry 1)
        if use_call_compat_matrix:
            self._call_compat_matrix = self._get_call_compat_matrix(alphabet)
        else:
            self._call_compat_matrix = jnp.ones((self._alphabet_size, self._alphabet_size))

        self._states = jnp.arange(self._max_num_states)
        self._propositions = jnp.arange(1, alphabet_size+1)
        self._random_walk_iters = jnp.arange(self._avg_connectivity*self._max_num_states)
        self._rm_indices = jnp.arange(self._max_num_rms)
        self._transition_mask = jnp.ones((self._max_num_states, self._max_num_states)) - jnp.eye(self._max_num_states)

    def sample(self, key: chex.PRNGKey, extras: Optional[Dict] = None, max_num_gen_rms: int = jnp.inf, max_num_gen_states: int = jnp.inf) -> HRM:
        if self._max_num_gen_rms is not None:
            max_num_gen_rms = self._max_num_gen_rms

        if self._max_num_gen_states is not None:
            max_num_gen_states = self._max_num_gen_states

        # Deduct root RM
        max_num_gen_rms -= 1

        rng, _rng = jax.random.split(key)

        # This generates the individual RMs in a tensor
        rngs = jax.random.split(key, num=self._max_num_rms + 1)
        rng = rngs[0]
        num_gen_states = jnp.minimum(max_num_gen_states, self._max_num_states)*jnp.ones((self._max_num_rms, 1))
        base_generate = lambda keys, states: self._generate_graph_base(keys, states, extras["prop_probs"] if extras is not None else None)
        rms, rewards, eps = jax.vmap(base_generate)(rngs[1:], num_gen_states)

        if self._max_num_non_root_rms > 0:
            # This array indicates how many children each RM will have in the hierarchy,
            # e.g. 0 means it is a leaf node whereas 3 means it has 3 children
            repeats = self._random_branching(_rng, max_num_gen_rms)

            # The start and terminal indices determine the range of 1-entries in the
            # proposition mask, e.g. [0, 0, 1, 1, 0] corresponds to start_idx=2, terminal_idx=4
            cum_sum_repeats = jnp.cumsum(repeats)
            start_idxs = cum_sum_repeats - repeats + 1
            terminal_idxs = cum_sum_repeats + 1

            # Masking out unused RMs
            rm_mask = jnp.zeros(self._max_num_rms)
            rm_mask = rm_mask.at[terminal_idxs[-1]].set(1)
            rm_mask = jnp.cumsum(rm_mask)
            rm_mask = (1 - rm_mask).reshape(-1, 1, 1, 1)
            rms = rm_mask*rms

            # Compatible propositions is a list of RMs with their respective compatible propositions,
            # e.g. an RM that has initial transition prop_a has compatible propositions prop_b, prop_c, etc.
            # that do not contradict prop_a semantically, i.e. this is to ensure call-realted satisfiability
            initial_props = jnp.sum(rms[:, 0], axis=1)
            compatible_props = jnp.clip(initial_props @ self._call_compat_matrix, a_min=0, a_max=1)
            compatible_props = jax.random.uniform(rng, minval=0.5, maxval=1.0, shape=compatible_props.shape)*compatible_props
            default_call = 0.01*jnp.ones(compatible_props.shape[-1]).reshape(-1, 1)

            def generate_calls(carry, _):
                rm_id, s_mask, rms = carry
                start_idx = start_idxs[rm_id]
                terminal_idx = terminal_idxs[rm_id]
                s_mask = s_mask.at[start_idx].add(1)
                start_vec = jnp.cumsum(s_mask)
                s_mask = (0 * s_mask).at[terminal_idx].add(1)
                # s_mask is of dim (max_num_rms, ) and is binary, indicating which RMs can be called from the current rm_id
                # as determined by random_branching
                s_mask = start_vec - jnp.cumsum(s_mask)
                contraction = jnp.expand_dims(s_mask, axis=-1)*compatible_props
                # rm_calls is an array of dim (alphabet_size, ) and is filled with indices of RMs that could be called
                rm_calls = jnp.hstack((contraction.T, default_call))
                rm_calls = rm_calls.at[:, rm_id].set(-1)
                # for each literal by index in rm_calls we get a call index as the entry
                rm_calls = jnp.argmax(rm_calls, axis=1)
                # prohibits a call on the initial transition
                initial_prop = jnp.sum(rms[rm_id, 0]*self._propositions.reshape(1, 1, -1), axis=-1).astype(jnp.int32)
                rm_calls = rm_calls.at[initial_prop-1].set(self._max_num_rms)
                # remain_mask is of dim (num_rms, alphabet_size) and
                # filters out RMs that should be called but cannot be called due to incompatibilty
                compat = self._rm_indices.reshape(-1, 1) == rm_calls
                remain_mask = (jnp.einsum("bcd, rd -> r", rms[rm_id], compat) > 0).astype(jnp.int32)
                # restrict remain_mask to the callable RMs from current RM, i.e. don't remove RMs that couldn't have been called
                remain_mask = (s_mask == remain_mask).astype(jnp.int32)
                inverted_s_mask = 1 - s_mask
                # delete uncalled RMs
                rms = (inverted_s_mask + remain_mask - inverted_s_mask*remain_mask).reshape(-1, 1, 1, 1)*rms
                #create call matrix for current RM
                rm_calls = jnp.einsum("bcd, d -> bc", rms[rm_id], rm_calls)
                return (rm_id + 1, 0*s_mask, rms), (rm_id, rm_calls)

            sum_mask = jnp.zeros(self._max_num_rms)
            (_, _, rms), (_, calls) = jax.lax.scan(generate_calls, (0, sum_mask, rms), self._rm_indices)

            calls = calls - (calls == 0)

            # edge padding
            calls = jnp.expand_dims(calls, axis=-1)
            calls_shape = calls.shape
            calls_shape = calls_shape[:-1] + (self._max_num_edges-1, )
            calls = jnp.concatenate((calls, jnp.zeros(calls_shape)), axis=-1)
        else:
            calls = jnp.expand_dims(2*(jnp.sum(jnp.abs(rms), axis=-1) > 0).astype(jnp.int32) - 1, axis=-1)

        # mutex
        if self._enforce_mutex:
            row_sum_formulas = jnp.sum(rms, axis=2, keepdims=True)
            mask = (jnp.sum(rms, axis=-1, keepdims=True) > 0)
            neg_formulas = jnp.tile(row_sum_formulas, (1, 1, rms.shape[1], 1))
            rms = 2*rms - mask*neg_formulas
        
        # edge padding
        formulas = jnp.expand_dims(rms, axis=-2)
        formula_shape = formulas.shape
        formula_shape = formula_shape[:-2] + (self._max_num_edges-1, ) + formula_shape[-1:]
        formulas = jnp.concatenate((formulas, jnp.zeros(formula_shape)), axis=-2)

        # representation conversion
        formulas = formulas * self._propositions.reshape(1, 1, 1, 1, -1)
        formulas_idxs = jnp.argsort(jnp.abs(formulas), axis=-1)
        formulas = jnp.take_along_axis(formulas, formulas_idxs, axis=-1)
        formulas = formulas[:, :, :, :, -self._max_num_literals:][:, :, :, :, ::-1]
        num_literals = jnp.sum(formulas != 0, axis=-1)

        return HRM(
            root_id=jnp.asarray(0),
            calls=calls.astype(jnp.int32),
            formulas=formulas.astype(jnp.int32),
            rewards=rewards,
            num_literals=num_literals,
            extras=RandomWalkHRMSamplerExtras(rng=key)
        )

    def _random_branching(self, rng: chex.PRNGKey, max_num_gen_rms: int):
        """
        Generates a sequence of integers of length max_num_rms + 1 that such that
        its sum is always max_num_rms + 1
        """
        if self._max_num_non_root_rms == 1:
            return jnp.array([1, 0])
        points = jax.random.randint(rng, shape=(self._max_num_non_root_rms // 2,), minval=0, maxval=self._max_num_non_root_rms-1, dtype=jnp.int32)
        points = jnp.sort(points)
        sequence = points[1:] - points[:-1]
        sequence = jnp.concatenate((points[0, None] + 1, sequence, self._max_num_non_root_rms-1 - points[-1, None]))
        sequence_idxs = jnp.argsort(sequence > 0, descending=True)
        sequence = sequence[sequence_idxs]
        branching = jnp.concatenate((sequence, jnp.zeros(1 + self._max_num_non_root_rms // 2, dtype=jnp.int32)))
        branching_cum_sum = jnp.cumsum(branching)
        mask = branching_cum_sum <= max_num_gen_rms
        return mask*branching

    def _discounted_lower_tri(self, n, gamma_centre=0.5):
        """
        Create an n x n strictly lower triangular matrix.
        Gamma is max at center column and decreases linearly outward.
        """
        row = jnp.arange(n).reshape(-1, 1)  # shape (n, 1)
        col = jnp.arange(n).reshape(1, -1)  # shape (1, n)

        # Strictly lower triangle mask
        mask = row > col

        # Distance from center column
        mid = np.ceil(n / 2) + 0.5
        distance = jnp.abs(col - mid)

        # Scale distance to range [0, 1]
        max_dist = mid
        distance_norm = (distance / max_dist)

        # Linearly decreasing gamma from center (gamma = gamma_center at center, increases to 1.0 at edges)
        gamma = gamma_centre * distance_norm**2

        # Compute powers (row - col)
        powers = row - col

        # Compute gamma ** powers
        values = gamma ** powers

        # Apply strictly lower triangular mask
        result = jnp.where(mask, values, 0.0)
        return result

    def _generate_graph_base(self, rng: chex.PRNGKey, max_num_gen_states: int, prop_probs: chex.Array = None):
        """
        Generates a directed graph with a single source and a single sink.
        """
        source = jnp.array([0])
        sink = jnp.array([self._max_num_states - 1])

        rngs = jax.random.split(rng, 3)

        # Create the transition matrix for the states of the RM, e.g. it has
        # uniform weights here but can be exchanged for a policy in the future
        transition_matrix = (1/self._max_num_states)*jnp.ones((self._max_num_states, self._max_num_states))
        transition_matrix = transition_matrix * self._transition_mask
        if self._enforce_sequentiality:
            if self._enforce_dag:
                transition_matrix = self._discounted_lower_tri(self._max_num_states, self._eps)
            else:
                max_vals = jnp.max(transition_matrix, axis=1, keepdims=True)
                sequentiality_mask = (transition_matrix == max_vals).astype(jnp.int32)
                sequentiality_mask = sequentiality_mask.at[-1].set(1)
                transition_matrix = transition_matrix * sequentiality_mask

        # forces an RM to only have one transition leaving the initial state
        # this makes enforcing mutual exclusivity on the call conditions easier
        transition_matrix = transition_matrix.at[0].set(0)


        # Masking out states that we do not need
        state_matrix_mask = self._states < max_num_gen_states - 1
        state_matrix_mask = state_matrix_mask.at[-1].set(True).reshape(-1, 1)
        transition_matrix = state_matrix_mask*transition_matrix


        # Create the transition matrix for the proposition labels of the RM,
        # e.g. it has uniform weights here but can be exchanged for a policy in the future
        if prop_probs is None:
            prop_probs = jax.random.uniform(rngs[1], minval=0.5, maxval=1.0, shape=(self._alphabet_size, ))

        def random_walk(carry, _):
            """
            Creates a graph based on the markov chains defined by the transition matrices
            """
            def is_terminated(carry):
                # Stand still if agent terminated
                return carry

            def is_not_terminated(carry):
                # If agent is not terminated, then perform a step in the random walk
                iter, rng, state_state_visited, prev_state_idx, _ = carry
                rngs = jax.random.split(rng, 4)

                # Determine next state
                state_idx = jax.random.choice(rngs[0], self._states, shape=(1,), p=transition_matrix[:, prev_state_idx].flatten())

                # Determine next proposition, hereby make sure that no state can
                # have the same label more than once regardless of whether the
                # transition is outgoing or incoming
                index_set_1 = (state_state_visited[prev_state_idx] - 1).flatten()
                max_index_1 = jnp.max(index_set_1)
                index_set_1 = jnp.where(index_set_1 >= 0, index_set_1, max_index_1)
                is_max_index_1_pos = max_index_1 >= 0
                
                index_set_2 = (state_state_visited[state_idx] - 1).flatten()
                max_index_2 = jnp.max(index_set_2)
                index_set_2 = jnp.where(index_set_2 >= 0, index_set_2, max_index_2)
                is_max_index_2_pos = max_index_2 >= 0

                index_set_3 = (state_state_visited[:, prev_state_idx] - 1).flatten()
                max_index_3 = jnp.max(index_set_3)
                index_set_3 = jnp.where(index_set_3 >= 0, index_set_3, max_index_3)
                is_max_index_3_pos = max_index_3 >= 0

                index_set_4 = (state_state_visited[:, state_idx] - 1).flatten()
                max_index_4 = jnp.max(index_set_4)
                index_set_4 = jnp.where(index_set_4 >= 0, index_set_4, max_index_4)
                is_max_index_4_pos = max_index_4 >= 0

                prop_mask = 1 - jnp.clip(
                    (is_max_index_1_pos*jnp.sum(jnp.take(self._transition_compat_matrix, index_set_1, axis=1), axis=1).flatten() + is_max_index_3_pos*jnp.sum(jnp.take(self._transition_compat_matrix, index_set_3, axis=0), axis=0).flatten()
                     + is_max_index_2_pos*jnp.sum(jnp.take(self._transition_compat_matrix, index_set_2, axis=1), axis=1).flatten() + is_max_index_4_pos*jnp.sum(jnp.take(self._transition_compat_matrix, index_set_4, axis=0), axis=0).flatten()),
                    a_max=1)

                prop_vector = prop_mask * prop_probs

                # Look at available nonzero probability labels, if there is no
                # propostion left with which we could label the transition flag no_option
                no_option = ~jnp.any(prop_vector > 0)
                last_step = jnp.greater_equal(iter, self._avg_connectivity*self._max_num_states - 1)

                # If no_option is flagged terminate the walk
                state_idx = (1 - no_option) * state_idx + no_option * sink
                state_idx = (1 - last_step) * state_idx + last_step * sink

                terminated = jnp.equal(state_idx, sink)
                # Sample proposition label
                prop_idx = jax.random.choice(rngs[1], self._propositions, p=prop_vector)

                # The random walk has reached the terminal node
                ups = (jax.random.uniform(rngs[2]) < self._splittiness) & self._enforce_dag & ~no_option & ~last_step
                iter = (((1 - terminated)*iter + terminated*(self._max_num_states * jnp.ceil(iter / self._max_num_states)))).astype(iter.dtype).squeeze()
                ups = ups & jnp.less(iter, self._avg_connectivity*self._max_num_states - 1).squeeze()
                # Add the label if the transition has not already been performed before
                state_state_visited = state_state_visited.at[prev_state_idx, state_idx].add((1 - (state_state_visited[prev_state_idx, state_idx] > 0))*prop_idx)
                state_idx = (1 - (ups & terminated))*state_idx
                carry = (iter+1, rngs[3], state_state_visited, state_idx, (~ups.squeeze() & terminated.squeeze()))
                return carry

            carry = jax.lax.cond(carry[-1], carry, is_terminated, carry, is_not_terminated)

            return carry, None

        # The state state prop dictionary is a tensor that is num_states x num_states x num_propositions,
        # where the propositions are one-hot encoded
        state_state_visited = jnp.zeros((self._max_num_states, self._max_num_states), dtype=jnp.int32)

        init_carry = (1, rngs[2], state_state_visited, source, False)
        # Each node in the RM will on average have avg_connections number of connections
        carry, _ = jax.lax.scan(random_walk, init_carry, self._random_walk_iters)
        state_state_visited = carry[2]

        # Determine rewards and mask out unreachable states
        reward_mask = (state_state_visited.T > 0).astype(jnp.float32)

        if self._reward_shaping:
            def reward_iteration(carry, _):
                # 0.3 + 0.7 = 1: Any other values are also legit, the more weight on the distribution bias (atm 0.7),
                # the more severe the differences between unimportant and important node potentials
                base, dist = carry
                dist = dist @ reward_mask
                dist = 0.3*dist
                dist = dist.at[-1].add(0.7)
                dist = dist/jnp.linalg.norm(dist)
                base = base + dist
                return (base, dist), None

            dist = jnp.zeros(self._max_num_states)
            dist = dist.at[-1].add(1)
            init_state = jnp.zeros(self._max_num_states)
            init_state = init_state.at[-1].add(1)
            (rewards, _), _ = jax.lax.scan(reward_iteration, (init_state, dist), self._random_walk_iters)
            rewards = rewards/(2*jnp.max(rewards))
            M = reward_mask.T+jnp.identity(len(rewards))
            reward_matrix = (self._gamma * rewards.reshape(1, -1) * M) - (rewards.reshape(-1, 1) * M)
        else:
            reward_matrix = reward_mask.T
            reward_matrix = reward_matrix.at[:, :-1].set(0)

        state_state_prop_dict = jax.nn.one_hot(state_state_visited-1, self._alphabet_size)

        return state_state_prop_dict, reward_matrix, self._eps

    def _get_transition_compat_matrix(self, alphabet: List[str]) -> chex.Array:
        # TODO: Currently domain-dependent
        order_matrix = np.ones((len(alphabet), len(alphabet)))
        for i, prop_i in enumerate(alphabet):
            pred_i = self._get_pred_from_prop(prop_i)
            const_i = self._get_constants_from_prop(prop_i)
            for j, prop_j in enumerate(alphabet):
                pred_j = self._get_pred_from_prop(prop_j)
                const_j = self._get_constants_from_prop(prop_j)
                order_matrix[i, j] = 1 - int((const_j in const_i) or (const_i in const_j) and (pred_i == pred_j))
        return jnp.asarray(order_matrix)

    def _get_call_compat_matrix(self, alphabet: List[str]) -> chex.Array:
        # TODO: Currently domain-dependent
        order_matrix = np.ones((len(alphabet), len(alphabet)))
        for i, prop_i in enumerate(alphabet):
            pred_i = self._get_pred_from_prop(prop_i)
            for j, str_j in enumerate(alphabet):
                pred_j = self._get_pred_from_prop(str_j)
                order_matrix[i, j] = 1 - int(pred_i == pred_j)
        return jnp.asarray(order_matrix)

    def _get_pred_from_prop(self, proposition: str) -> str:
        """
        Returns the predicate from the proposition's string representation.
        Example: "front_ball_red" -> "front_ball".
        TODO: Currently domain-dependent
        """
        return proposition.split("_", maxsplit=1)[0]

    def _get_constants_from_prop(self, proposition: str) -> str:
        """
        Returns the constants string representation from the proposition's
        string representation.
        Example: "front_ball_red" -> "ball_red".
        TODO: Currently domain-dependent
        """
        return proposition.split("_", maxsplit=1)[1]
