from typing import Optional, Tuple

import chex
import jax.numpy as jnp
from xminigrid.core.grid import two_rooms

from .multi_room import XMinigridMultiRoomLevelSampler
from ..types import XMinigridEnvParams


class XMinigridTwoRoomsLevelSampler(XMinigridMultiRoomLevelSampler):
    """
    Generates a grid with a wall in the middle that constitutes two rooms. In the
    middle of the wall there is a door (closed, locked or open, chosen at random)
    or a hole if the environment parameters do not allow for doors. The number of
    non-door objects is specified through an object density argument.

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
        min_objects: int = 10,
        max_objects: int = 10,
        height: Optional[int] = 7,
        width: Optional[int] = 13,
    ):
        super().__init__(env_params, min_objects, max_objects, height, width)

    def _make_grid(self):
        return two_rooms(self._height, self._width)

    def _is_valid_size(self):
        return True

    def _get_corridors(self) -> Tuple[chex.Array, chex.Array]:
        return (
            jnp.array([self._height // 2]),
            jnp.array([self._width // 2])
        )

    def _get_num_available_cells(self):
        return (
            (self._height - 2) * (self._width - 2)
            - (self._height - 2)  # wall between rooms
            + 1  # corridor between rooms
        )
