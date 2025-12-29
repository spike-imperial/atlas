from typing import Optional

from flax import linen as nn
from flax.typing import Dtype
from jax import numpy as jnp
from xminigrid.core.constants import NUM_TILES, NUM_COLORS


class EmbeddingEncoder(nn.Module):
    """
    A module for encoding the observations from XLand-Minigrid.

    See https://github.com/dunnolab/xland-minigrid/blob/a46e78ce92f28bc90b8aac96d3b7b7792fb5bf3b/training/nn.py#L75.

    See also Section 2.2 ('Observation and action space') in the paper:
    https://arxiv.org/abs/2312.12044.
    """

    emb_dim: int = 16
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, img):
        entity_emb = nn.Embed(NUM_TILES, self.emb_dim, self.dtype, self.param_dtype)
        color_emb = nn.Embed(NUM_COLORS, self.emb_dim, self.dtype, self.param_dtype)

        # [..., channels]
        return jnp.concatenate(
            [
                entity_emb(img[..., 0]),
                color_emb(img[..., 1]),
            ],
            axis=-1,
        )
