from typing import Optional

import chex
import jax
from jax import numpy as jnp
from xminigrid.core.constants import Tiles
from xminigrid.core.grid import sample_coordinates

from ...common.level_sampling import LevelSampler


class XMinigridLevelSampler(LevelSampler):
    def get_max_num_objects(self):
        """
        Returns the maximum number of unique objects that can be placed.
        """
        num_tiles = len(self._env_params.non_door_obj_types) + len(self._env_params.door_obj_types)
        num_colored_tiles = num_tiles * len(self._env_params.color_types) + 2  # for walls (+1) and floor (+1)
        return min(num_colored_tiles, (self._env_params.height - 2) * (self._env_params.width - 2))

    def _pop_grid_with_all_non_door_obj_col(
        self, key: chex.PRNGKey, grid: chex.Array
    ) -> chex.Array:
        """
        Returns a grid where each object-color combination is placed on a random position.

        Args:
            key: the key used to generate the positions for each object-color combination.
            env_params: the parameters of the environment.
            grid: the grid where to place the objects.
        """
        num_objects = len(self._env_params.non_door_obj_types) * len(
            self._env_params.color_types
        )
        obj_pos = sample_coordinates(key, grid, num_objects)
        for i in range(len(self._env_params.non_door_obj_types)):
            for j in range(len(self._env_params.color_types)):
                idx = i * len(self._env_params.color_types) + j
                grid = grid.at[obj_pos[idx, 0], obj_pos[idx, 1]].set(
                    jnp.array(
                        [
                            self._env_params.non_door_obj_types[i],
                            self._env_params.color_types[j],
                        ],
                        dtype=jnp.uint8,
                    )
                )
        return grid

    def _pop_grid_from_rand_non_door_objs(
        self,
        key: chex.PRNGKey,
        grid: chex.Array,
        num_objects: int,
        max_objects: int,
        mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        """
        Populates a grid with number of random non-door objects placed at random locations.
        """
        obj_key, obj_pos_key = jax.random.split(key)
        objs = self._sample_objs(obj_key, max_objects)
        obj_pos = sample_coordinates(obj_pos_key, grid, max_objects, mask)

        def _f(grid, xs):
            count, obj, pos = xs
            return jax.lax.select(count < num_objects, grid.at[pos[0], pos[1]].set(obj), grid), None

        grid, _ = jax.lax.scan(_f, grid, (jnp.arange(max_objects), objs, obj_pos))
        return grid

    def _pop_grid_with_non_door_obj_from_list(
        self,
        key: chex.PRNGKey,
        grid: chex.Array,
        tgt_num_objects: int,
        max_objects: int,
        obj_list: chex.Array,
    ) -> chex.Array:
        """
        Populates a grid with objects from a list up to a maximum. If the
        list contains fewer objects than the maximum, the rest are randomly
        generated. All objects are placed at random locations.
        """
        obj_key, obj_pos_key = jax.random.split(key)

        # Sample the minimum number of objects
        extra_objs = self._sample_objs(obj_key, max_objects)

        # Ignore doors, invalid objects and objects that go over the maximum.
        # Compute number of actual objects we add from the list.
        tile_ids = obj_list[:, 0]
        obj_mask = (
            (tile_ids > -1) &
            (tile_ids != Tiles.DOOR_OPEN) &
            (tile_ids != Tiles.DOOR_CLOSED) &
            (tile_ids != Tiles.DOOR_LOCKED)
        )
        cumsum_objs = obj_mask.cumsum()
        obj_mask = obj_mask * (cumsum_objs <= tgt_num_objects)
        num_objs = cumsum_objs[-1]

        # Compute the number of extra objects
        num_extra_objs = jnp.maximum(0, tgt_num_objects - num_objs)

        # Compute mask for all the objects we aim to add and the final list of objects
        final_mask = jnp.concat((
            obj_mask,
            jnp.arange(max_objects) < num_extra_objs
        ))
        all_objs = jnp.concat((obj_list, extra_objs), axis=0)

        def pop_grid(carry, obj_mask):
            i, grid = carry
            obj, mask, obj_rng = obj_mask
            obj_pos = sample_coordinates(obj_rng, grid, num=1)[0]
            i, grid = jax.lax.cond(
                pred=mask,
                true_fun=lambda: (i + 1, grid.at[obj_pos[0], obj_pos[1]].set(
                    jnp.asarray(obj, dtype=jnp.uint8)
                )),
                false_fun=lambda: (i, grid)
            )
            return (i, grid), None

        # Go through each object and it
        init_carry = (jnp.asarray(0), grid)
        obj_pos_keys = jax.random.split(obj_pos_key, all_objs.shape[0])
        carry, _ = jax.lax.scan(pop_grid, init_carry, xs=(all_objs, final_mask, obj_pos_keys))
        return carry[1]

    def _sample_objs(self, key: chex.PRNGKey, num_objects: int) -> chex.Array:
        """
        Samples a random list of tile-color pairs.
        """
        obj_type_key, obj_color_key = jax.random.split(key)
        obj_types = jax.random.choice(
            obj_type_key,
            jnp.array(self._env_params.non_door_obj_types, dtype=jnp.uint8),
            (num_objects, 1),
        )
        obj_colors = jax.random.choice(
            obj_color_key,
            jnp.array(self._env_params.color_types, dtype=jnp.uint8),
            (num_objects, 1),
        )

        return jnp.hstack((obj_types, obj_colors))
