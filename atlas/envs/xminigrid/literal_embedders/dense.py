import chex
import jax.numpy as jnp
from flax import linen as nn

from .utils import prop_to_vec
from ..labeling_function import XMinigridLabelingFunction
from ...common.literal_embedder import LiteralEmbedder


class XMinigridDenseLiteralEmbedder(LiteralEmbedder):
    """
    An embedder that leverages the semantics of the propositions used
    in XLand-Minigrid.

    At initialization, it creates basic embeddings encoding the location
    type, and for each possible object (at most two, because of the `next`
    location type) encode its type, color and state. Each property is
    encoded using 0s and 1s.

    At the time of building the final embeddings:
        1. Literals are transformed into propositions and their sign is
           obtained. The sign will determine the sign of the numbers in
           the final embedding.
        2. The propositions are used to recover the base embedding created
           at initialization time.
        3. The object parts of the propositions are passed through a dense
           layer and summed to capture order invariance.
        4. The result is appended to the location vector and passed altogether
           to another dense layer.
        5. The output is multiplied by the sign.

    Note that proposition 0 actually corresponds to the special condition
    'True', which has no sign (i.e. 0). If one wants to have a non-zero
    encoding for it, the flag `use_true` must be enabled, which will give
    it a sign of 1.
    """
    base_loc_v: chex.Array   # Base embeddings for the location part of each proposition
    base_obj1_v: chex.Array  # Base embeddings for the first object part of each proposition
    base_obj2_v: chex.Array  # Base embeddings for the second object part of each proposition

    d_feat: int              # Number of dimensions for the projections
    use_obj2: bool           # Whether object 2 is used
    use_true: bool = True    # Whether let it have a non-zero embedding

    def setup(self) -> None:
        assert self.d_feat is not None

        # Projection for the objects
        self.obj_proj = nn.Dense(self.d_feat)

        # Projection for the concatenation of obj. projections and location
        self.loc_objs_proj = nn.Dense(self.d_feat)

    def __call__(self, literal: chex.Array):
        # If the True literals (whose value is 0) are to be used,
        # their sign will be set to 1 so that the final embedding
        # is not 'zerofied'
        prop = jnp.abs(literal)
        sign = jnp.sign(literal) | (self.use_true & (prop == 0))

        # Get base embeddings for the propositions
        loc_v = self.base_loc_v[prop]
        obj1_v = self.base_obj1_v[prop]
        obj2_v = self.base_obj2_v[prop]

        # Obtain object embedding part
        obj1_v = self.obj_proj(obj1_v)
        if self.use_obj2:
            obj2_v = self.obj_proj(obj2_v)
            objs_v = obj1_v + obj2_v  # invariance order
        else:
            objs_v = obj1_v

        # Append the location (and transform)
        loc_objs_v = jnp.concat((loc_v, objs_v), axis=-1)
        loc_objs_v = self.loc_objs_proj(loc_objs_v)

        # Apply sign and return
        # Note: prop 0 (corresponding to True) is already all zeros
        return sign[..., jnp.newaxis] * loc_objs_v

    @staticmethod
    def init_embedder(
        label_fn: XMinigridLabelingFunction,
        d_feat: int,
        use_true: bool = True,
    ) -> "XMinigridDenseLiteralEmbedder":
        loc_v, obj1_v, obj2_v = zip(*[
            prop_to_vec(label_fn, prop_id)
            for prop_id in range(label_fn.get_alphabet_size())
        ])

        loc_true_v = jnp.zeros_like(loc_v[0])[jnp.newaxis, :]
        obj_true_v = jnp.zeros_like(obj1_v[0])[jnp.newaxis, :]

        return XMinigridDenseLiteralEmbedder(
            base_loc_v=jnp.concat((loc_true_v, jnp.asarray(loc_v))),
            base_obj1_v=jnp.concat((obj_true_v, jnp.asarray(obj1_v))),
            base_obj2_v=jnp.concat((obj_true_v, jnp.asarray(obj2_v))),
            d_feat=d_feat,
            use_obj2=label_fn.use_next_to_props,
            use_true=use_true,
        )
