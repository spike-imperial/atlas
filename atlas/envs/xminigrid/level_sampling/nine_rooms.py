from typing import Optional, Tuple

import chex
import jax.numpy as jnp
from xminigrid.core.grid import nine_rooms

from .multi_room import XMinigridMultiRoomLevelSampler
from ..types import XMinigridEnvParams


class XMinigridNineRoomsLevelSampler(XMinigridMultiRoomLevelSampler):
    """
    Generates a nine rooms XLand-Minigrid level (i.e., grid). Objects (determined
    by the environment parameters) are placed randomly in the grid. Adjacent rooms
    are connected through a one-cell corridor, which may contain a door of a random
    color and state. Corridors are placed in the middle of the walls separating two
    rooms.
    """
    def __init__(
        self,
        env_params: XMinigridEnvParams,
        min_objects: int = 25,
        max_objects: int = 25,
        height: Optional[int] = 19,
        width: Optional[int] = 19,
    ):
        super().__init__(env_params, min_objects, max_objects, height, width)

    def _make_grid(self):
        return nine_rooms(self._height, self._width)

    def _is_valid_size(self):
        return (self._width == self._height) and (self._width % 2 == 1)

    def _get_corridors(self) -> Tuple[chex.Array, chex.Array]:
        return (
            jnp.array([
                self._height // 6,
                self._height // 6,
                3 * self._height // 6,
                3 * self._height // 6,
                5 * self._height // 6,
                5 * self._height // 6,
                self._height // 3,
                self._height // 3,
                self._height // 3,
                2 * self._height // 3,
                2 * self._height // 3,
                2 * self._height // 3,
            ]),
            jnp.array([
                self._width // 3,
                2 * self._width // 3,
                self._width // 3,
                2 * self._width // 3,
                self._width // 3,
                2 * self._width // 3,
                self._width // 6,
                3 * self._width // 6,
                5 * self._width // 6,
                self._width // 6,
                3 * self._width // 6,
                5 * self._width // 6,
            ])
        )

    def _get_num_available_cells(self):
        room_cells = (self._height // 3 - 1) ** 2
        return 9 * room_cells + self._num_corridors
