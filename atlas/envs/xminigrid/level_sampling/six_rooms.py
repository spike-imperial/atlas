from typing import Optional, Tuple

import chex
import jax.numpy as jnp
from xminigrid.core.constants import Colors, Tiles, TILES_REGISTRY
from xminigrid.core.grid import (
    empty_world,
    horizontal_line,
    rectangle,
    vertical_line
)
from xminigrid.types import Tile

from .multi_room import XMinigridMultiRoomLevelSampler
from ..types import XMinigridEnvParams


class XMinigridSixRoomsLevelSampler(XMinigridMultiRoomLevelSampler):
    """
    Generates a six rooms XLand-Minigrid level (i.e., grid). Objects (determined
    by the environment parameters) are placed randomly in the grid. Adjacent rooms
    are connected through a one-cell corridor, which may contain a door of a random
    color and state. Corridors are placed in the middle of the walls separating two
    rooms.
    """
    def __init__(
        self,
        env_params: XMinigridEnvParams,
        min_objects: int = 20,
        max_objects: int = 20,
        height: Optional[int] = 13,
        width: Optional[int] = 19,
    ):
        super().__init__(env_params, min_objects, max_objects, height, width)

    def _make_grid(self):
        wall_tile: Tile = TILES_REGISTRY[Tiles.WALL, Colors.GREY]
        grid = empty_world(self._height, self._width)
        grid = rectangle(grid, 0, 0, self._height, self._width, tile=wall_tile)
        grid = vertical_line(grid, self._width // 3, 0, self._height, tile=wall_tile)
        grid = vertical_line(grid, 2 * self._width // 3, 0, self._height, tile=wall_tile)
        grid = horizontal_line(grid, 0, self._height // 2, self._width, tile=wall_tile)
        return grid

    def _is_valid_size(self):
        return (self._width > self._height) and (self._height % 2 == 1) and (self._width % 3 == 1)

    def _get_corridors(self) -> Tuple[chex.Array, chex.Array]:
        return (
            jnp.array([
                self._height // 2,
                self._height // 2,
                self._height // 2,
                self._height // 4,
                self._height // 4,
                3 * self._height // 4,
                3 * self._height // 4,
            ]),
            jnp.array([
                (self._width - 1) // 6,
                3 * (self._width - 1) // 6,
                5 * (self._width - 1) // 6,
                self._width // 3,
                2 * self._width // 3,
                self._width // 3,
                2 * self._width // 3,
            ])
        )

    def _get_num_available_cells(self):
        room_cells = (self._width // 3 - 1) * (self._height // 2 - 1)
        return 6 * room_cells + self._num_corridors
