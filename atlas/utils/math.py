from typing import Optional

import chex
import jax
import jax.numpy as jnp


def segment_softmax(
    logits: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: Optional[int] = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False
) -> chex.ArrayTree:
    """
    Computes a segment-wise softmax.
    Copied from: https://github.com/google-deepmind/jraph/blob/master/jraph/_src/utils.py.

    For a given tree of logits that can be divded into segments, computes a
    softmax over the segments.

    logits = jnp.ndarray([1.0, 2.0, 3.0, 1.0, 2.0])
    segment_ids = jnp.ndarray([0, 0, 0, 1, 1])
    segment_softmax(logits, segments)
    >> DeviceArray([0.09003057, 0.24472848, 0.66524094, 0.26894142, 0.7310586],
    >> dtype=float32)

    Args:
    logits: an array of logits to be segment softmaxed.
    segment_ids: an array with integer dtype that indicates the segments of
      `data` (along its leading axis) to be maxed over. Values can be repeated
      and need not be sorted. Values outside of the range [0, num_segments) are
      dropped and do not contribute to the result.
    num_segments: optional, an int with positive value indicating the number of
      segments. The default is ``jnp.maximum(jnp.max(segment_ids) + 1,
      jnp.max(-segment_ids))`` but since ``num_segments`` determines the size of
      the output, a static value must be provided to use ``segment_sum`` in a
      ``jit``-compiled function.
    indices_are_sorted: whether ``segment_ids`` is known to be sorted
    unique_indices: whether ``segment_ids`` is known to be free of duplicates

    Returns:
    The segment softmax-ed ``logits``.
    """
    # First, subtract the segment max for numerical stability
    maxs = jax.ops.segment_max(
      logits, segment_ids, num_segments, indices_are_sorted, unique_indices
    )
    logits = logits - maxs[segment_ids]

    # Then take the exp
    logits = jnp.exp(logits)

    # Then calculate the normalizers
    normalizers = jax.ops.segment_sum(
        logits, segment_ids, num_segments, indices_are_sorted, unique_indices
    )
    normalizers = normalizers[segment_ids]
    softmax = logits / normalizers
    return softmax
