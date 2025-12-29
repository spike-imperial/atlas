import chex
import jax
import jax.numpy as jnp
from xminigrid.core.constants import NUM_TILES, NUM_COLORS

from ....hrm.ops import get_propositions
from ....problem_samplers.independent import IndependentProblemSampler


class XMinigridHRMConditionedProblemSampler(IndependentProblemSampler):
    """
    A problem sampler that samples levels based on sampled HRMs.
    The levels are sampled such that the objects in it are determined
    by the propositions in the HRM.

    Warning: Feasibility might not be guaranteed. Examples:
      - The HRM contains a door proposition but the level sampler
        cannot generate doors (it generates single rooms without doors).
      - The number of objects associated with the HRM goes beyond of what
        is allowed in the grid (a maximum number of objects or even the
        grid size).
    """
    def sample(self, rng: chex.PRNGKey):
        hrm_rng, obj_rng, level_rng = jax.random.split(rng, 3)

        # Sample the HRM on which the level generation is conditioned
        hrm = self._hrm_sampler.sample(hrm_rng)

        # Obtain the propositions from the HRM
        props = get_propositions(hrm, self._label_fn.get_alphabet_size())

        # Get the objects from each proposition
        obj_rngs = jax.random.split(obj_rng, len(props))
        xminigrid_objs = jax.vmap(
            lambda rng, prop: self._label_fn.sample_xminigrid_objs_from_prop(prop, rng)
        )(obj_rngs, props)

        # Get unique objects from list
        xminigrid_objs = self._get_unique_xminigrid_objs(xminigrid_objs)

        # Sample the level conditioned to the object set and return
        return self._level_sampler.sample(level_rng, {"objects": xminigrid_objs}), hrm

    def _get_unique_xminigrid_objs(self, xminigrid_obj_pairs: chex.Array) -> chex.Array:
        """
        Returns the list of unique XMinigrid objects extracted from the propositions.

        Important: If a given object appears twice in a pair (e.g., `next_ball_blue_ball_blue`),
        we add two of such objects. That's why the `jnp.unique` method across the 0th axis is
        not used.
        """
        def f(carry, xminigrid_obj_pair):
            _seen, _xminigrid_objs, _obj_count = carry

            obj1 = xminigrid_obj_pair[:2]
            obj2 = xminigrid_obj_pair[2:]

            obj1_t = tuple(obj1)
            obj2_t = tuple(obj2)

            # If the object has not been seen before, set it as seen and add it
            _seen, _xminigrid_objs, _obj_count = jax.lax.cond(
                pred=(obj1[0] > -1) & (_seen[obj1_t] == 0),
                true_fun=lambda: (_seen.at[obj1_t].add(1), _xminigrid_objs.at[_obj_count].set(obj1), _obj_count + 1),
                false_fun=lambda: (_seen, _xminigrid_objs, _obj_count)
            )

            # If the object has not been seen before, or it has been seen and it is equal than the first object,
            # add it
            _seen, _xminigrid_objs, _obj_count = jax.lax.cond(
                pred=(obj2[0] > -1) & ((jnp.all(obj1 == obj2) & (_seen[obj2_t] == 1)) | (_seen[obj2_t] == 0)),
                true_fun=lambda: (_seen.at[obj2_t].add(1), _xminigrid_objs.at[_obj_count].set(obj2), _obj_count + 1),
                false_fun=lambda: (_seen, _xminigrid_objs, _obj_count)
            )

            return (_seen, _xminigrid_objs, _obj_count), None

        # The carry contains:
        # - Whether a given object has already been checked.
        # - The final list of unique XMinigrid objects.
        # - The counter.
        # Note the maximum number of objects is 2 * ... because the same object can
        # be mentioned twice in each pair.
        init_carry = (
            jnp.zeros((NUM_TILES, NUM_COLORS), dtype=int),
            -jnp.ones(
                (2 * self._label_fn.get_num_obj_types() * self._label_fn.get_num_color_types(), 2),
                dtype=int
            ),
            0
        )
        carry, _ = jax.lax.scan(f, init=init_carry, xs=xminigrid_obj_pairs)
        return carry[1]
