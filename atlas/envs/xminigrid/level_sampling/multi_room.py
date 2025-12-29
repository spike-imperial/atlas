from abc import abstractmethod
from typing import Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from xminigrid.core.constants import Colors, Tiles
from xminigrid.core.grid import sample_coordinates, sample_direction

from .base import XMinigridLevelSampler
from ..level import XMinigridLevel
from ..types import XMinigridEnvParams
from ..utils import pad_grid, sample_door_obj_type, sample_color_type


class XMinigridMultiRoomLevelSampler(XMinigridLevelSampler):
    """
    Base class for generating XLand-Minigrid levels/grids with multiple rooms.
    Objects (determined by the environment parameters) are placed randomly in the grid.
    The corridors between two rooms are determined by the subclasses, and may contain
    a door of a random color and state.

    Attributes:
        - env_params: The parameters used for the environment. The objects
            there will those be contained in the generated level.
        - min_objects: The minimum number of objects to place in the grid.
        - max_objects: The maximum number of objects to place in the grid.
        - height: the actual height of the grid.
        - width: the actual width of the grid.
    """
    def __init__(
        self,
        env_params: XMinigridEnvParams,
        min_objects: int,
        max_objects: int,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ):
        super().__init__(env_params)

        self._height = height if height else env_params.height
        self._width = width if width else env_params.width
        assert self._is_valid_size()

        self._min_objects = min_objects
        self._max_objects = max_objects
        self._use_doors = len(self._env_params.door_obj_types) > 0
        self._corridor_rows, self._corridor_cols = self._get_corridors()
        self._num_corridors = self._corridor_rows.shape[0]
        assert self._max_objects >= self._min_objects >= self._num_corridors

    def sample(self, key: chex.PRNGKey, extras: Optional[Dict] = None) -> XMinigridLevel:
        obj_key, obj_num_key, agent_pos_key, agent_dir_key = jax.random.split(key, 4)

        # Create base grid
        grid = self._make_grid()
        num_objects = jax.random.choice(obj_num_key, jnp.arange(self._min_objects, self._max_objects + 1))

        # Populate grid with objects (from any possible object, or the list in `extras`)
        if extras is not None:
            grid = self._pop_grid_from_list(obj_key, grid, num_objects, extras["objects"])
        else:
            grid = self._pop_grid_from_all(obj_key, grid, num_objects)

        # Generate the agent position and direction and return the level
        return XMinigridLevel(
            height=self._height,
            width=self._width,
            grid=pad_grid(grid, self._env_params.height, self._env_params.width),
            agent_pos=sample_coordinates(agent_pos_key, grid, 1)[0],
            agent_dir=sample_direction(agent_dir_key),
        )

    def _pop_grid_from_all(self, key: chex.PRNGKey, grid: chex.Array, num_objects: int) -> chex.Array:
        door_key, rest_key = jax.random.split(key)

        # Place non-door objects
        num_non_door_objects = num_objects - self._use_doors * self._num_corridors
        grid = jax.lax.select(
            num_non_door_objects > 0,
            self._pop_grid_from_rand_non_door_objs(rest_key, grid, num_non_door_objects, self._max_objects),
            grid
        )

        # Sample the doors for the corridors (or take standard floor tiles)
        if self._use_doors:
            door_status_key, door_col_key = jax.random.split(door_key, 2)
            door_colors = jax.random.choice(
                door_col_key,
                jnp.asarray(self._env_params.color_types, dtype=jnp.uint8),
                (self._num_corridors, 1),
            )
            door_status = jax.random.choice(
                door_status_key,
                jnp.asarray(self._env_params.door_obj_types, dtype=jnp.uint8),
                (self._num_corridors, 1),
            )
            corridor_objs = jnp.hstack((door_status, door_colors))
        else:
            corridor_objs = jnp.tile(
                jnp.array([Tiles.FLOOR, Colors.BLACK], dtype=jnp.uint8),
                (self._num_corridors, 1),
            )

        # Place the door/empty tiles and return
        return self._pop_corridors(grid, corridor_objs)

    def _pop_grid_from_list(self, key: chex.PRNGKey, grid: chex.Array, num_objects: int, obj_list: chex.Array) -> chex.Array:
        """
        Populates the grid from objects in a list.
        """
        door_key, rest_key = jax.random.split(key, 2)

        num_non_door_objects = num_objects - self._use_doors * self._num_corridors
        grid = jax.lax.select(
            num_non_door_objects > 0,
            self._pop_grid_with_non_door_obj_from_list(rest_key, grid, num_non_door_objects, self._max_objects, obj_list),
            grid
        )

        if self._use_doors:
            corridor_objs = self._sample_doors_from_list(door_key, obj_list)
        else:
            corridor_objs = jnp.tile(jnp.array([Tiles.FLOOR, Colors.BLACK]), (self._num_corridors, 1))

        return self._pop_corridors(grid, jnp.asarray(corridor_objs, dtype=jnp.uint8))

    def _sample_doors_from_list(self, key: chex.PRNGKey, obj_list: chex.Array) -> chex.Array:
        tile_ids = obj_list[:, 0]
        door_mask = (
            (tile_ids == Tiles.DOOR_OPEN) |
            (tile_ids == Tiles.DOOR_CLOSED) |
            (tile_ids == Tiles.DOOR_LOCKED)
        )

        def _sample_random_doors():
            tile_key, col_key = jax.random.split(key, 2)
            tiles = sample_door_obj_type(tile_key, self._env_params, self._num_corridors)
            colors = sample_color_type(col_key, self._env_params, self._num_corridors)
            return jnp.concat((tiles[:, jnp.newaxis], colors[:, jnp.newaxis]), axis=1)

        return jax.lax.cond(
            pred=door_mask.sum() > 0,
            true_fun=lambda: jax.random.choice(
                key, obj_list, shape=(self._num_corridors,), p=door_mask / door_mask.sum()
            ),
            false_fun=lambda: _sample_random_doors()
        )

    def _pop_corridors(self, grid: chex.Array, objs: chex.Array) -> chex.Array:
        return grid.at[self._corridor_rows, self._corridor_cols].set(objs)

    @abstractmethod
    def _make_grid(self):
        raise NotImplementedError

    @abstractmethod
    def _is_valid_size(self):
        raise NotImplementedError

    @abstractmethod
    def _get_corridors(self) -> Tuple[chex.Array, chex.Array]:
        raise NotImplementedError

    @abstractmethod
    def _get_num_available_cells(self) -> int:
        raise NotImplementedError

    def get_max_num_objects(self):
        num_colored_tiles = (
            len(self._env_params.non_door_obj_types) * len(self._env_params.color_types) +
            self._num_corridors * len(self._env_params.color_types) * (len(self._env_params.door_obj_types) > 0) +
            3  # for walls (+1), floor (+1) and empty/padding cells (+1) same for max objects below
        )
        return min(self._max_objects + 3, num_colored_tiles, self._get_num_available_cells())
