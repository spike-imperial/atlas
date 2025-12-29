from typing import Tuple

import chex
import jax.numpy as jnp

from ..labeling_function import XMinigridLabelingFunction


def prop_to_vec(label_fn: XMinigridLabelingFunction, prop_id: int) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """
    Transforms a proposition into a vector form. Returns vectors
    for location information, object 1 and object 2 separately.
    """
    if label_fn.is_front_prop(prop_id):
        loc_idx = 0
    elif label_fn.is_carrying_prop(prop_id):
        loc_idx = 1 if label_fn.use_front_props else 0
    elif label_fn.is_next_to_prop(prop_id):
        if not label_fn.use_front_props and not label_fn.use_carrying_props:
            loc_idx = 0
        elif label_fn.use_front_props and label_fn.use_carrying_props:
            loc_idx = 2
        elif label_fn.use_front_props or label_fn.use_carrying_props:
            loc_idx = 1

    loc_v = jnp.zeros((label_fn.get_num_loc_types(),))
    loc_v = loc_v.at[loc_idx].set(1)

    def _get_obj_vec(obj_id: int, color_id: int, status_id: int) -> chex.Array:
        obj_v = jnp.zeros((label_fn.get_num_obj_types())).at[obj_id].set(1)

        color_v = jnp.zeros((label_fn.get_num_color_types()))
        if label_fn.is_non_empty_color(color_id):
            color_v = color_v.at[color_id].set(1)

        status_v = jnp.zeros((label_fn.get_num_status_types()))
        if label_fn.is_valid_non_empty_status(obj_id, status_id):
            status_v = status_v.at[status_id].set(1)

        return jnp.concat((obj_v, color_v, status_v))

    if label_fn.is_front_prop(prop_id):
        obj1_v = _get_obj_vec(*label_fn.get_front_obj_properties(prop_id))
        obj2_v = jnp.zeros_like(obj1_v)
    elif label_fn.is_carrying_prop(prop_id):
        obj1_v = _get_obj_vec(*label_fn.get_carrying_obj_properties(prop_id))
        obj2_v = jnp.zeros_like(obj1_v)
    elif label_fn.is_next_to_prop(prop_id):
        obj1, col1, status1, obj2, col2, status2 = label_fn.get_next_obj_properties(prop_id)
        obj1_v = _get_obj_vec(obj1, col1, status1)
        obj2_v = _get_obj_vec(obj2, col2, status2)

    return loc_v, obj1_v, obj2_v
