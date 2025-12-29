import chex
import jax
from jax import numpy as jnp
from typing import Tuple

from flax import linen as nn


class RNNModel(nn.Module):
    """
    Adapted from https://github.com/DramaCow/jaxued/blob/62d155ae772aa2a0ecac9541d498df7ae88e6def/src/jaxued/linen.py#L9.
    """
    cell: nn.RNNCellBase

    @nn.compact
    def __call__(self, inputs: Tuple[chex.Array, chex.Array], init_hstate):
        # Determine hidden state to reset to when an episode is terminated
        batch_size = inputs[0].shape[0]
        reset_hstate = self.cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, 1))

        def scan_fn(cell: nn.RNNCellBase, hstate, input: Tuple[chex.Array, chex.Array]):
            obs, done = input
            hstate = jax.tree_map(
                lambda a, b: jnp.where(done[:, None], a, b), reset_hstate, hstate
            )
            return cell(hstate, obs)

        scan = nn.scan(
            scan_fn,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )

        # Returns new hidden state and output
        return scan(self.cell, init_hstate, inputs)
