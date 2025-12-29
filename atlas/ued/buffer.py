"""
Based on: https://github.com/DramaCow/jaxued/blob/main/src/jaxued/level_sampler.py
"""

from typing import Literal, Optional, TypedDict, Tuple

import chex
import jax
import jax.numpy as jnp

from ..envs.common.level import Level
from ..hrm.types import HRM

Prioritization = Literal["rank", "topk"]


class Buffer(TypedDict):
    problems:        chex.Array # shape (capacity, ...)
    scores:          chex.Array # shape (capacity)
    init_timestamps: chex.Array # shape (capacity)
    timestamps:      chex.Array # shape (capacity)
    num_replays:     chex.Array # shape (capacity)
    size:            int
    episode_count:   int
    extra:           Optional[dict]


class BufferManager:
    """
    The `BufferManager` provides all of the functionality associated with a buffer in a PLR/ACCEL-type method. 
    In the standard Jax style, the buffer manager class does not store any data itself, and accepts a `buffer` object for most operations.

    Examples:
        >>>
        pholder_level       = ...
        pholder_hrm         = ...
        pholder_extra       = ...
        buffer_manager      = BufferManager(4000)
        buffer              = buffer_manager.initialize(pholder_level, pholder_hrm, pholder_extra)
        should_replay       = buffer_manager.sample_replay_decision(buffer, rng)
        replay_problems     = buffer_manager.sample_replay_problems(buffer, rng, 32) # 32 replay problems
        scores              = ... # eval agent
        buffer              = buffer_manager.insert_batch(buffer, level, hrm, scores)

    Args:
        capacity (int): The maximum number of problems (level + HRM) that can be stored in the buffer.
        replay_prob (float, optional): The chance of performing on_replay vs on_new. Defaults to 0.95.
        staleness_coeff (float, optional): The weighting factor for staleness. Defaults to 0.5.
        minimum_fill_ratio (float, optional): The class will never sample a replay decision until the buffer is at least as full as specified by this value. Defaults to 1.0.
        prioritization_params (dict, optional): If prioritization="rank", this has a "temperature" field; for "topk" it has a "k" field. If not provided, by default this is initialized to a temperature of 1.0 and k=1. Defaults to None.
        duplicate_check (bool, optional): If this is true, duplicate problems cannot be added to the buffer. This adds some computation to check for duplicates. Defaults to False.
    
    Based on:
        https://github.com/DramaCow/jaxued/blob/c3350ab6708c87d15648a8b29498cf979fab494a/src/jaxued/level_sampler.py
    """
    def __init__(
        self,
        capacity: int,
        replay_prob: float = 0.95,
        staleness_coeff: float = 0.5,
        minimum_fill_ratio: float = 1.0,  # minimum fill required before replay can occur
        prioritization: Prioritization = "rank",
        prioritization_params: dict = None,
        duplicate_check: bool = False,
        use_leq_insertion: bool = False,
        use_tie_breaking: bool = False,
        use_replace: bool = True,
        use_primary_score: bool = True,
        num_scores: int = 1,
        score_coeffs: chex.Array = jnp.array([1])
    ):
        self.capacity = capacity
        self.replay_prob = replay_prob
        self.staleness_coeff = staleness_coeff
        self.minimum_fill_ratio = minimum_fill_ratio
        self.prioritization = prioritization
        self.prioritization_params = prioritization_params
        self.duplicate_check = duplicate_check
        self.use_leq_insertion = use_leq_insertion
        self.use_tie_breaking = use_tie_breaking
        self.use_replace = use_replace
        self.use_primary_score = use_primary_score
        self.num_scores = num_scores
        self.score_coeffs = score_coeffs

        if prioritization_params is None:
            if prioritization == "rank":
                self.prioritization_params = {"temperature": 1.0}
            elif prioritization == "topk":
                self.prioritization_params = {"k": 1}
            else:
                raise Exception(f"\"{prioritization}\" not a valid prioritization.")
        
    def initialize(self, pholder_level: Level, pholder_hrm: HRM, pholder_extra=None) -> Buffer:
        """
        Returns the `sampler` object as a dictionary.

        Sampler Object Keys:
            * "problems" (shape (self.capacity, ...)): the levels and HRMs themselves
            * "scores" (shape (self.capacity)): the scores of the problems
            * "init_timestamps" (shape (self.capacity)): the timestamps of insertion of the problems
            * "timestamps" (shape (self.capacity)): the timestamps of the problems
            * "size" (int): the number of problems currently in the buffer
            * "episode_count" (int): the number of episodes that have been played so far

        Args:
            pholder_level (Level): A placeholder level that will be used to initialize the buffer.
            pholder_hrm (HRM): A placeholder HRM that will be used to initialize the buffer.
            pholder_extra (dict, optional): If given, this should be a dictionary with arbitrary keys that is kept track of alongside each problem. An example is "max_return" for each problem. Defaults to None.

        Returns:
            Sampler: The initialized sampler object
        """
        sampler = {
            "problems": jax.tree_map(lambda x: jnp.array([x]).repeat(self.capacity, axis=0), (pholder_level, pholder_hrm)),
            "scores": jnp.full((self.num_scores, self.capacity), -jnp.inf, dtype=jnp.float32),
            "init_timestamps": jnp.zeros(self.capacity, dtype=jnp.int32),
            "timestamps": jnp.zeros(self.capacity, dtype=jnp.int32),
            "num_replays": jnp.zeros(self.capacity, dtype=jnp.int32),
            "size": 0,
            "episode_count": 0,
        }
        if pholder_extra is not None:
            sampler["extra"] = jax.tree_map(lambda x: jnp.array([x]).repeat(self.capacity, axis=0), pholder_extra)
        return sampler
        
    def sample_replay_decision(self, sampler: Buffer, rng: chex.PRNGKey) -> bool:
        """
        Returns a single boolean indicating if a `replay` or `new` step should be taken. This is based on the proportion of the buffer that is filled and the `replay_prob` parameter.

        Args:
            sampler (Sampler): The sampler object
            rng (chex.PRNGKey): 

        Returns:
            bool: 
        """
        proportion_filled = self._proportion_filled(sampler)
        return (proportion_filled >= self.minimum_fill_ratio) & (jax.random.uniform(rng) < self.replay_prob)

    def sample_replay_problem(self, sampler: Buffer, rng: chex.PRNGKey) -> Tuple[Buffer, Tuple[int, Level, HRM]]:
        """
        Samples a replay problem from the buffer. It does this by first computing the weights of each problem (using `problem_weights`), and then sampling from the buffer using these weights. The `sampler` object is updated to reflect the new episode count and the problem that was sampled. The problem itself is returned as well as the index of the problem in the buffer.

        Args:
            sampler (Sampler): The sampler object
            rng (chex.PRNGKey): 

        Returns:
            Tuple[Sampler, Tuple[int, Level, HRM]]: The updated sampler object, the sampled level's index and the level and HRM themselves.
        """
        weights = self.problem_weights(sampler)
        idx = jax.random.choice(rng, self.capacity, p=weights)
        new_episode_count = sampler["episode_count"] + 1
        sampler = {
            **sampler,
            # "init_timestamps": sampler["init_timestamps"].at[idx].set(new_episode_count),
            "timestamps": sampler["timestamps"].at[idx].set(new_episode_count),
            "episode_count": new_episode_count,
        }
        return sampler, (idx, *jax.tree_map(lambda x: x[idx], sampler["problems"]))
    
    def sample_replay_problems(self, sampler: Buffer, rng: chex.PRNGKey, num: int) -> Tuple[Buffer, Tuple[chex.Array, Level, HRM]]:
        """
        Samples several problems by iteratively calling `sample_replay_problem`. The `sampler` object is updated to reflect the new episode count and the problems that were sampled.

        Args:
            sampler (Sampler): The sampler object
            rng (chex.PRNGKey): 
            num (int): How many problems to sample

        Returns:
            Tuple[Sampler, Tuple[chex.Array, Level, HRM]]: The updated sampler, an array of indices, and multiple problems (levels + HRMs).
        """
        if self.use_replace:
            return jax.lax.scan(self.sample_replay_problem, sampler, jax.random.split(rng, num), length=num)
        return self._sample_replay_problems_without_replacement(sampler, rng, num)

    def _sample_replay_problems_without_replacement(self, sampler: Buffer, rng: chex.PRNGKey, num: int) -> Tuple[Buffer, Tuple[chex.Array, Level, HRM]]:
        weights = self.problem_weights(sampler)
        idxs = jax.random.choice(rng, self.capacity, shape=(num,), p=weights, replace=False)
        new_episode_count = sampler["episode_count"] + num
        sampler = {
            **sampler,
            "timestamps": sampler["timestamps"].at[idxs].set(new_episode_count),
            "episode_count": new_episode_count,
        }
        return sampler, (idxs, *jax.tree_util.tree_map(lambda x: x[idxs], sampler["problems"]))

    def insert(self, sampler: Buffer, level: Level, hrm: HRM, score: float, extra: dict=None) -> Tuple[Buffer, int]:
        """
        Attempt to insert problem into the buffer.
        
        Insertion occurs when:
        - Corresponding score exceeds the score of the lowest weighted problem
          currently in the buffer (in which case it will replace it).
        - Buffer is not yet at capacity.
        
        Optionally, if the problem to be inserted already exists in the
        buffer, the corresponding buffer entry will be updated instead.
        (See, `duplicate_check`)

        Args:
            sampler (Sampler): The sampler object
            level (Level): Level to insert
            hrm (HRM): HRM to insert
            score (float): Its score
            extra (dict, optional): If extra was given in `initialize`, then it must be given here too. Defaults to None.

        Returns:
            Tuple[Sampler, int]: The updated sampler, and the problems's index in the buffer (-1 if it was not inserted)
        """
        if self.duplicate_check:
            idx = self.find(sampler, level, hrm)
            return jax.lax.cond(
                idx == -1,
                lambda: self._insert_new(sampler, level, hrm, score, extra),
                lambda: ({
                    **self.update(sampler, idx, score, extra),  # what happens to mutation rate here?
                    "timestamps": sampler["timestamps"].at[idx].set(sampler["episode_count"] + 1),
                    "episode_count": sampler["episode_count"] + 1
                }, idx),
            )
        return self._insert_new(sampler, level, hrm, score, extra)
    
    def insert_batch(self, sampler: Buffer, levels: Level, hrms: HRM, scores: chex.Array, problem_extras: dict=None) -> Tuple[Buffer, chex.Array]:
        """
        Inserts a batch of problems.

        Args:
            sampler (Sampler): The sampler object
            levels (Level): The levels to insert. This must be a `batched` level, in that it has an extra dimension at the front.
            hrms (HRM): The HRMs to insert. This must be a `batched` HRM, in that it has an extra dimension at the front.
            scores (float): The scores of each level
            problem_extras (dict, optional): The optional problem_extras. Defaults to None.
        """
        def _insert(sampler, step):
            level, hrm, score, problem_extra = step
            return self.insert(sampler, level, hrm, score, problem_extra)
        return jax.lax.scan(_insert, sampler, (levels, hrms, scores, problem_extras))
    
    def find(self, sampler: Buffer, level: Level, hrm: HRM) -> int:
        """
        Returns the index of problem in the buffer. If problem is not present, -1 is returned.

        Args:
            sampler (Sampler): The sampler object
            level (Level): The level to find
            hrm (HRM): The HRM to find

        Returns:
            int: index or -1 if not found.
        """
        eq_tree = jax.tree_map(lambda X, y: (X == y).reshape(self.capacity, -1).all(axis=-1), sampler["problems"], (level, hrm))
        eq_tree_flat, _ = jax.tree_util.tree_flatten(eq_tree)
        eq_mask = jnp.array(eq_tree_flat).all(axis=0) & (jnp.arange(self.capacity) < sampler["size"])
        return jax.lax.select(eq_mask.any(), eq_mask.argmax(), -1)
    
    def get_problems(self, sampler: Buffer, idx: int) -> Tuple[Level, HRM]:
        """
        Returns the level and HRM at a particular index.

        Args:
            sampler (Sampler): The sampler object
            level_idx (int): The index to return

        Returns:
            Level: 
            HRM:
        """
        return jax.tree_map(lambda x: x[idx], sampler["problems"])
    
    def get_problems_extra(self, sampler: Buffer, level_idx: int) -> dict:
        """
        Returns the extras associated with a particular index

        Args:
            sampler (Sampler): The sampler object
            level_idx (int): The index to return

        Returns:
            dict: 
        """
        return jax.tree_map(lambda x: x[level_idx], sampler["extra"])
    
    def update(self, sampler: Buffer, idx: int, score: float, problem_extra: dict=None) -> Buffer:
        """
        This updates the score and problem_extras of a problem (level + HRM).
        The update is performed only if the score is not -inf, i.e. if at least
        one episode was completed during the policy rollout.

        Args:
            sampler (Sampler): The sampler object
            idx (int): The index of the level
            score (float): The score of the level
            problem_extra (dict, optional): The associated. Defaults to None.

        Returns:
            Sampler: Updated Sampler
        """
        update_cond = jnp.all(score > -jnp.inf)

        def _replace():
            new_sampler = {
                **sampler,
                "scores": sampler["scores"].at[:, idx].set(score),
                "num_replays": sampler["num_replays"].at[idx].set(sampler["num_replays"][idx] + 1),
            }
            if problem_extra is not None:
                new_sampler["extra"] = jax.tree_map(lambda x, y: x.at[idx].set(y), new_sampler["extra"], problem_extra)
            return new_sampler

        return jax.lax.cond(update_cond, _replace, lambda: sampler,)
    
    def update_batch(self, sampler: Buffer, inds: chex.Array, scores: chex.Array, extras: dict=None) -> Buffer:
        """
        Updates the scores and extras of a batch of problems.

        Args:
            sampler (Sampler): The sampler object
            inds (chex.Array): Indices
            scores (chex.Array): Scores
            extras (dict, optional): . Defaults to None.

        Returns:
            Sampler: Updated Sampler
        """
        def _update(sampler, step):
            problem_idx, score, extra = step
            return self.update(sampler, problem_idx, score, extra), None
        return jax.lax.scan(_update, sampler, (inds, scores, extras))[0]
        
    def problem_weights(self, sampler: Buffer, prioritization: Prioritization=None, prioritization_params: dict=None) -> chex.Array:
        """
        Returns the weights for each problem, taking into account both staleness and score.

        Args:
            sampler (Sampler): The sampler
            prioritization (Prioritization, optional): Possibly overrides self.prioritization. Defaults to None.
            prioritization_params (dict, optional): Possibly overrides self.prioritization_params. Defaults to None.

        Returns:
            chex.Array: Weights, shape (self.capacity)
        """
        w_s = self.score_weights(sampler, prioritization, prioritization_params)
        w_c = self.staleness_weights(sampler)
        return (1 - self.staleness_coeff) * w_s + self.staleness_coeff * w_c
    
    def score_weights(self, sampler: Buffer, prioritization: Prioritization=None, prioritization_params: dict=None, score_coeffs: list=None) -> chex.Array:
        """
        Returns an array of shape (self.capacity) with the weights of each problem (for sampling purposes).

        Args:
            sampler (Sampler): 
            prioritization (Prioritization, optional): Possibly overrides self.prioritization. Defaults to None.
            prioritization_params (dict, optional): Possibly overrides self.prioritization_params. Defaults to None.

        Returns:
            chex.Array: Score weights, shape (self.capacity)
        """
        mask = jnp.arange(self.capacity) < sampler["size"]
        
        if prioritization is None:
            prioritization = self.prioritization
        if prioritization_params is None:
            prioritization_params = self.prioritization_params
        if score_coeffs is None:
            score_coeffs = self.score_coeffs
        
        if prioritization == "rank":
            ord = (-jnp.where(mask, sampler["scores"], -jnp.inf)).argsort()
            row_idx = jnp.arange(self.num_scores)[:, jnp.newaxis]
            ranks = jnp.empty_like(ord).at[row_idx, ord].set(jnp.arange(self.capacity) + 1)
            temperature = prioritization_params["temperature"]
            w_s = jnp.where(mask, 1 / ranks, 0) ** (1 / temperature)
            w_s = w_s / w_s.sum(axis=1, keepdims=True)
        elif prioritization == "topk":
            ord = (-jnp.where(mask, sampler["scores"], -jnp.inf)).argsort()
            k = prioritization_params["k"]
            row_idx = jnp.arange(self.num_scores)[:, jnp.newaxis]
            topk_mask = jnp.empty_like(ord).at[row_idx, ord].set(jnp.arange(self.capacity) < jnp.minimum(sampler["size"], k))
            w_s = jax.nn.softmax(sampler["scores"], where=topk_mask, initial=0)
        else:
            raise Exception(f"\"{self.prioritization}\" not a valid prioritization.")
        
        return jnp.sum(score_coeffs[:, jnp.newaxis] * w_s, axis=0)

    def staleness(self, sampler: Buffer) -> chex.Array:
        """
        Returns the staleness of the problems in the buffer.
        """
        return sampler["episode_count"] - sampler["timestamps"]

    def staleness_weights(self, sampler: Buffer) -> chex.Array:
        """
        Returns staleness weights for each problem.

        Args:
            sampler (Sampler): 

        Returns:
            chex.Array: shape (self.capacity)
        """
        mask = jnp.arange(self.capacity) < sampler["size"]
        staleness = self.staleness(sampler)
        w_c = jnp.where(mask, staleness, 0)
        w_c = w_c / w_c.max()  # normalize the staleness for numerical stability
        w_c = jax.lax.select(w_c.sum() > 0, w_c / w_c.sum(), mask / sampler["size"])
        return w_c
    
    def freshness_weights(self, sampler: Buffer) -> chex.Array:
        """
        Returns freshness weights for each problem.

        Args:
            sampler (Sampler): 

        Returns:
            chex.Array: shape (self.capacity)
        """
        mask = jnp.arange(self.capacity) < sampler["size"]
        earliest_timestamp = jnp.where(mask, sampler["timestamps"], jnp.iinfo(jnp.int32).max).min()
        freshness = sampler["timestamps"] - earliest_timestamp
        w_f = jnp.where(mask, freshness, 0)
        w_f = jax.lax.select(w_f.sum() > 0, w_f / w_f.sum(), mask / sampler["size"])
        return w_f
    
    def flush(self, sampler: Buffer) -> Buffer:
        """
        Flushes this sampler, putting it back to its empty state. 
        This does update it in place.

        Args:
            sampler (Sampler): 

        Returns:
            Sampler:
        """
        sampler["size"] = 0
        sampler["scores"] = jnp.full((self.num_scores, self.capacity), -jnp.inf, dtype=jnp.float32)
        return sampler
    
    def _insert_new(self, sampler: Buffer, level: Level, hrm: HRM, score: float, problem_extra: dict) -> Tuple[Buffer, int]:
        idx = self._get_next_idx(sampler)

        def _replace():
            new_sampler = {
                **sampler,
                "problems": jax.tree_map(lambda x, y: x.at[idx].set(y), sampler["problems"], (level, hrm)),
                "scores": sampler["scores"].at[:, idx].set(score),
                "init_timestamps": sampler["init_timestamps"].at[idx].set(sampler["episode_count"] + 1),
                "timestamps": sampler["timestamps"].at[idx].set(sampler["episode_count"] + 1),
                "size": jnp.minimum(sampler["size"] + 1, self.capacity),
                "num_replays": sampler["num_replays"].at[idx].set(1),  # overwriting a problem resets the replay count
            }
            if problem_extra is not None:
                new_sampler["extra"] = jax.tree_map(lambda x, y: x.at[idx].set(y), new_sampler["extra"], problem_extra)
            return new_sampler

        cmp = jnp.less_equal if self.use_leq_insertion else jnp.less
        if self.use_primary_score:
            replace_cond = cmp(sampler["scores"][0][idx], score[0])
            new_sampler = jax.lax.cond(replace_cond, _replace, lambda: sampler)
        else:
            new_sampler = _replace()
            score_probs = self.score_weights(sampler)
            score_probs_new = self.score_weights(new_sampler)
            replace_cond = jnp.logical_or(
                jnp.isnan(score_probs[idx]), cmp(score_probs[idx], score_probs_new[idx])
            )
            new_sampler = jax.lax.cond(replace_cond, lambda: new_sampler, lambda: sampler)

        new_sampler["episode_count"] += 1
        
        return new_sampler, jax.lax.select(replace_cond, idx, -1)
    
    def _proportion_filled(self, sampler: Buffer) -> float:
        return sampler["size"] / self.capacity
    
    def _get_next_idx(self, sampler: Buffer) -> int:
        def _f():
            """
            Returns the index of the problem with the lowest replay probability if no tie breaking;
            otherwise, one of the problems with the lowest probability is returned based on the
            number of replay times (one with the highest).
            """
            scores = self.problem_weights(sampler)
            if self.use_tie_breaking:
                return jnp.argmax(sampler["num_replays"] * jnp.isclose(scores, scores.min()))
            return scores.argmin()

        return jax.lax.select(
            sampler["size"] < self.capacity,
            sampler["size"],
            _f()
        )

    def get_highest_scored_problems(self, sampler: Buffer, num_problems: int) -> Tuple[Level, HRM]:
        return self._get_lowest_scored_problems_aux(sampler, num_problems, -self.score_weights(sampler))

    def get_lowest_scored_problems(self, sampler: Buffer, num_problems: int) -> Tuple[Level, HRM]:
        return self._get_lowest_scored_problems_aux(sampler, num_problems, self.score_weights(sampler))

    def get_highest_weighted_problems(self, sampler: Buffer, num_problems: int) -> Tuple[Level, HRM]:
        return self._get_lowest_scored_problems_aux(sampler, num_problems, -self.problem_weights(sampler))

    def get_lowest_weighted_problems(self, sampler: Buffer, num_problems: int) -> Tuple[Level, HRM]:
        return self._get_lowest_scored_problems_aux(sampler, num_problems, self.problem_weights(sampler))

    def _get_lowest_scored_problems_aux(self, sampler: Buffer, num_problems: int, scores: chex.Array) -> Tuple[Level, HRM]:
        # Discard unfilled positions in the buffer by setting the highest possible score
        mask = jnp.arange(self.capacity) >= sampler["size"]
        scores = jnp.where(mask, jnp.inf, scores)

        # Get the lowest scored problems
        indices = jnp.argpartition(scores, num_problems)[:num_problems]
        return self.get_problems(sampler, indices[jnp.argsort(scores[indices])])

    def get_weighted_score(self, sampler: Buffer) -> chex.Array:
        mask = jnp.arange(self.capacity) < sampler["size"]
        return jnp.where(mask, sampler["scores"] * self.problem_weights(sampler), 0).sum(axis=1)

    def get_score_aggregates(self, sampler: Buffer) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Returns the min, mean and max scores in the buffer.
        """
        mask = jnp.arange(self.capacity) < sampler["size"]

        # Extract the mean avoiding numerical overflow
        scores = sampler["scores"]
        norm_factor = jnp.max(jnp.where(mask, jnp.abs(scores), 0), axis=1, keepdims=True)
        norm_metric = scores / jnp.maximum(norm_factor, 1e-10)
        avg_norm_metric = jnp.sum(jnp.where(mask, norm_metric, 0), axis=1, keepdims=True) / jnp.maximum(sampler["size"], 1)
        avg_metric = avg_norm_metric * norm_factor

        return (
            jnp.min(jnp.where(mask, scores, jnp.inf), axis=1),  # min
            avg_metric.squeeze(axis=1),  # mean
            jnp.max(jnp.where(mask, scores, -jnp.inf), axis=1),  # max
        )

    def get_staleness_aggregates(self, sampler: Buffer) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Returns the min, mean and max staleness in the buffer.
        """
        mask = jnp.arange(self.capacity) < sampler["size"]

        # Extract the mean avoiding numerical overflow
        staleness = self.staleness(sampler)
        norm_factor = jnp.max(jnp.abs(mask * staleness))
        norm_metric = staleness / jnp.maximum(norm_factor, 1)
        avg_norm_metric = jnp.sum(mask * norm_metric) / jnp.maximum(sampler["size"], 1)
        avg_metric = avg_norm_metric * norm_factor

        return (
            jnp.min(mask * staleness + jnp.logical_not(mask) * jnp.inf),  # min
            avg_metric,  # mean
            jnp.max(mask * staleness),  # max
        )
