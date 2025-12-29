import chex
import jax
import jax.numpy as jnp
from xminigrid.core.constants import Tiles, Colors

from ....problem_samplers.independent import IndependentProblemSampler


class XMinigridLevelConditionedProblemSampler(IndependentProblemSampler):
    """
    A problem sampler that samples HRMs based on sampled levels.
    The HRMs are sampled such that the edges can be labeled with
    propositions derived from the level's objects. The probability
    of each possible proposition is chosen uniformly at random.

    Warning: While it ensures the propositions are associated
    with objects, it does not guarantee the level-HRM pair is
    solvable. Examples:
      - A door cannot be unlocked because the key is missing
        or inaccessible.
      - If the door in the level is unlocked and the HRM involves
        a locked door.
      - If the grid contains one door, and the HRM mentions
        it should be closed or open, then locked (which is impossible,
        doors can only become unlocked).
      - If an object is required to put next to a locked door
        but the door requires unlocking to reach the object.
    """
    def sample(self, rng: chex.PRNGKey):
        level_rng, hrm_rng = jax.random.split(rng,2)

        # Sample a level on which the HRM generation is conditioned
        level = self._level_sampler.sample(level_rng)

        # Get the list of objects in the level and get the unique
        # objects, bounding the output to the maximum possible
        # value of generated object types.
        objs = jnp.unique(
            level.grid.reshape(-1, 2),
            axis=0,
            size=self._level_sampler.unwrapped().get_max_num_objects(),
            fill_value=jnp.array([Tiles.FLOOR, Colors.BLACK])  # a usual value for filling
        )

        # Obtain proposition masks
        mask = self._label_fn.get_prop_mask_from_objs(objs)

        # Get uniform probabilities for each proposition
        probs = mask / jnp.sum(mask)

        # Sample the HRM using the probabilities
        hrm = self._hrm_sampler.sample(hrm_rng, {"prop_probs": probs})

        return level, hrm
