from typing import Dict, Optional

import chex
import jax
import jax.numpy as jnp
from xminigrid.core.grid import room, sample_coordinates, sample_direction

from .base import XMinigridLevelSampler
from ..level import XMinigridLevel
from ..types import XMinigridEnvParams
from ..utils import pad_grid


class XMinigridSingleRoomBaseLevelSampler(XMinigridLevelSampler):
    def __init__(self, env_params: XMinigridEnvParams, height: Optional[int] = None, width: Optional[int] = None):
        super().__init__(env_params)
        self._height = height if height else env_params.height
        self._width = width if width else env_params.width

    def get_max_num_objects(self):
        """
        Returns the maximum number of unique objects that can be placed.
        """
        num_colored_tiles = len(self._env_params.non_door_obj_types) * len(self._env_params.color_types) + 3  # for walls (+1), floor (+1), empty cells (+1)
        return min(num_colored_tiles, (self._height - 2) * (self._width - 2))


class XMinigridSingleRoomAllPairsLevelSampler(XMinigridSingleRoomBaseLevelSampler):
    """
    Generates a squared room containing all possible non_door_object-color pairs.

    Attributes (superclass):
        - env_params: The parameters used for the environment. The objects
            there will those be contained in the generated level. Make sure the
            room is big enough to let the agent move around.
        - height: The height of the grid (without the padding from `env_params`).
        - width: The width of the grid (without the padding from `env_params`).
    """
    def sample(self, key: chex.PRNGKey, extras: Optional[Dict] = None) -> XMinigridLevel:
        grid = room(self._height, self._width)
        num_objects = len(self._env_params.non_door_obj_types) * len(self._env_params.color_types)

        # Make sure the grid is big enough for the agent to move around
        assert 2 * self._height * self._width > num_objects

        agent_pos_key, agent_dir_key, obj_pos_key = jax.random.split(key, 3)

        # Assign an object-color pair to each non-door object
        grid = self._pop_grid_with_all_non_door_obj_col(obj_pos_key, grid)

        return XMinigridLevel(
            grid=pad_grid(grid, self._env_params.height, self._env_params.width),
            agent_pos=sample_coordinates(agent_pos_key, grid, 1)[0],
            agent_dir=sample_direction(agent_dir_key),
        )


class XMinigridSingleRoomLevelSampler(XMinigridSingleRoomBaseLevelSampler):
    """
    Generates a squared room containing random non-door objects. The
    number of objects is determined by a specified density and the
    size of the room.

    If a list of objects is specified, the objects in the list are
    added up to the maximum number of objects. If remaining slots
    remain, they are filled with random objects.

    Attributes (superclass):
        - env_params: The parameters used for the environment. The objects
            there will those be contained in the generated level. Make sure the
            room is big enough to let the agent move around.
        - min_objects: The minimum number of objects to place in the grid.
        - max_objects: The maximum number of objects to place in the grid.
        - height: The height of the grid (without the padding from `env_params`).
        - width: The width of the grid (without the padding from `env_params`).
    """
    def __init__(
        self,
        env_params: XMinigridEnvParams,
        min_objects: int = 5,
        max_objects: int = 5,
        height: Optional[int] = 7,
        width: Optional[int] = 7,
    ):
        assert max_objects >= min_objects >= 1
        super().__init__(env_params, height, width)
        self._min_objects = min_objects
        self._max_objects = max_objects

    def sample(self, key: chex.PRNGKey, extras: Optional[Dict] = None) -> XMinigridLevel:
        obj_key, obj_num_key, agent_pos_key, agent_dir_key = jax.random.split(key, 4)

        grid = room(self._height, self._width)
        num_objects = jax.random.choice(obj_num_key, jnp.arange(self._min_objects, self._max_objects + 1))

        if extras is not None:
            grid = self._pop_grid_with_non_door_obj_from_list(key, grid, num_objects, self._max_objects, extras["objects"])
        else:
            grid = self._pop_grid_from_rand_non_door_objs(obj_key, grid, num_objects, self._max_objects)

        return XMinigridLevel(
            height=self._height,
            width=self._width,
            grid=pad_grid(grid, self._env_params.height, self._env_params.width),
            agent_pos=sample_coordinates(agent_pos_key, grid, 1)[0],
            agent_dir=sample_direction(agent_dir_key),
        )

    def get_max_num_objects(self):
        """
        Returns the maximum number of unique objects that can be placed.
        """
        return min(super().get_max_num_objects(), self._max_objects + 3)  # account for floor, walls and empty cells (+3)
