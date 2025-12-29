import jax
import jax
from jax import tree_util


def update_rngs(prev_key):
    # split each key in the dict
    next_keys = tree_util.tree_map(jax.random.split, prev_key)
    return prev_key, next_keys


def update_rngs(prev_key):
    # split each key in the dict
    next_keys = {}
    for key in prev_key.keys():
        prev_key[key], next_keys[key] = jax.random.split(prev_key[key])
    return prev_key, next_keys