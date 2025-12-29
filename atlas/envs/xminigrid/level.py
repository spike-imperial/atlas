from typing import Tuple

import chex
import jax.numpy as jnp
from xminigrid.core.constants import Colors, Tiles, TILES_REGISTRY
from xminigrid.types import State
import yaml

from .types import XMinigridEnvParams
from .utils import pad_grid
from ..common.level import Level

_LEVEL_LOADING_COLOURS = {
    "r": Colors.RED,
    "g": Colors.GREEN,
    "b": Colors.BLUE,
    "p": Colors.PURPLE,
    "y": Colors.YELLOW,
    "#": Colors.GREY,
    "n": Colors.BROWN,
    "o": Colors.ORANGE,
    "k": Colors.PINK,
    "w": Colors.WHITE,
}
_LEVEL_LOADING_COLOURS_INV = {v: k for k, v in _LEVEL_LOADING_COLOURS.items()}

_LEVEL_LOADING_OBJS = {
    "#": Tiles.WALL,
    "o": Tiles.BALL,
    "k": Tiles.KEY,
    "-": Tiles.DOOR_CLOSED,
    "+": Tiles.DOOR_OPEN,
    "x": Tiles.DOOR_LOCKED,
    "s": Tiles.SQUARE,
    "h": Tiles.HEX,
    "p": Tiles.PYRAMID,
    "*": Tiles.STAR,
}
_LEVEL_LOADING_OBJS_INV = {v: k for k, v in _LEVEL_LOADING_OBJS.items()}


class XMinigridLevel(Level):
    """
    Parameters characterizing an XLand-Minigrid instance.

    Args:
        height: The height of the grid.
        width: The width of the grid.
        grid: The XLand-Minigrid grid containing the objects (the grid is padded to
            comply with the height and width specified through the environment parameters)
        agent_pos: The position of the agent.
        agent_dir: The orientation of the agent.

    Inspired by:
    https://github.com/DramaCow/jaxued/blob/main/src/jaxued/environments/maze/level.py
    """

    height: int
    width: int
    grid: chex.Array  # (max_height, max_width, 2)
    agent_pos: chex.Array
    agent_dir: int
    agent_pocket: chex.Array = jnp.array([Tiles.EMPTY, Colors.EMPTY], dtype=jnp.uint8)

    @staticmethod
    def from_env_state(env_state: State, height: int, width: int) -> "XMinigridLevel":
        return XMinigridLevel(
            height=height,
            width=width,
            grid=env_state.grid,
            agent_pos=env_state.agent.position,
            agent_dir=env_state.agent.direction,
            agent_pocket=env_state.agent.pocket,
        )

    @staticmethod
    def from_file(path: str, env_params: XMinigridEnvParams) -> "XMinigridLevel":
        with open(path, "r") as f:
            env_config = yaml.safe_load(f)

        nrows, ncols, grid, agent_pos, agent_dir = XMinigridLevel._parse_grid(
            env_config["obj_grid"], env_config["col_grid"]
        )

        obj_pocket = env_config.get("obj_pocket")
        obj_pocket = _LEVEL_LOADING_OBJS[obj_pocket] if obj_pocket else Tiles.EMPTY

        col_pocket = env_config.get("col_pocket")
        col_pocket = _LEVEL_LOADING_COLOURS[col_pocket] if obj_pocket else Colors.EMPTY

        return XMinigridLevel(
            height=nrows,
            width=ncols,
            grid=pad_grid(grid, env_params.height, env_params.width),
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            agent_pocket=jnp.array([obj_pocket, col_pocket], dtype=jnp.uint8),
        )

    @staticmethod
    def _parse_grid(obj_grid: str, col_grid: str) -> Tuple[int, int, chex.Array, chex.Array, int]:
        obj_str = obj_grid.strip()
        rows = obj_str.split("\n")
        nrows = len(rows)
        assert all(
            len(row) == len(rows[0]) for row in rows
        ), "All rows must have same length"
        ncols = len(rows[0])

        colour_str = col_grid.strip()
        colour_rows = colour_str.split("\n")
        assert all(
            len(row) == len(colour_rows[0]) for row in colour_rows
        ), "All rows must have same length"

        grid = jnp.zeros((nrows, ncols, 2), dtype=jnp.uint8)
        grid = grid.at[:, :, 0:2].set(TILES_REGISTRY[Tiles.FLOOR, Colors.BLACK])
        agent_pos = None
        agent_dir = None

        for y, (row, colour_row) in enumerate(zip(rows, colour_rows)):
            for x, (cell, colour) in enumerate(zip(row, colour_row)):
                if cell in _LEVEL_LOADING_OBJS and colour in _LEVEL_LOADING_COLOURS:
                    grid = grid.at[y, x, 0:2].set(
                        TILES_REGISTRY[
                            _LEVEL_LOADING_OBJS[cell], _LEVEL_LOADING_COLOURS[colour]
                        ]
                    )
                elif cell == "^":
                    assert agent_pos is None, "Agent position can only be set once."
                    agent_pos = jnp.array([y, x])
                    agent_dir = 0
                elif cell == ">":
                    assert agent_pos is None, "Agent position can only be set once."
                    agent_pos = jnp.array([y, x])
                    agent_dir = 1
                elif cell == "v":
                    assert agent_pos is None, "Agent position can only be set once."
                    agent_pos = jnp.array([y, x])
                    agent_dir = 2
                elif cell == "<":
                    assert agent_pos is None, "Agent position can only be set once."
                    agent_pos = jnp.array([y, x])
                    agent_dir = 3

        return nrows, ncols, grid, agent_pos, agent_dir

    def to_file(self, path: str):
        with open(path, "w") as f:
            obj_str, col_str = self._grid_to_str()
            obj_pocket, col_pocket = int(self.agent_pocket[0]), int(self.agent_pocket[1])
            yaml.dump({
                "obj_grid": obj_str,
                "col_grid": col_str,
                "obj_pocket": _LEVEL_LOADING_OBJS_INV[obj_pocket] if obj_pocket != Tiles.EMPTY else None,
                "col_pocket": _LEVEL_LOADING_COLOURS_INV[col_pocket] if col_pocket != Colors.EMPTY else None,
            }, f, default_style="|")

    def _grid_to_str(self) -> Tuple[str, str]:
        obj_str = ""
        col_str = ""

        for y in range(self.height):
            for x in range(self.width):
                if jnp.array_equal(self.agent_pos, jnp.array([y, x])):
                    if self.agent_dir == 0:
                        obj_str += "^"
                    elif self.agent_dir == 1:
                        obj_str += ">"
                    elif self.agent_dir == 2:
                        obj_str += "v"
                    elif self.agent_dir == 3:
                        obj_str += "<"
                    col_str += "."
                else:
                    obj_id, col_id = int(self.grid[y, x, 0]), int(self.grid[y, x, 1])
                    if obj_id in _LEVEL_LOADING_OBJS_INV:
                        obj_str += _LEVEL_LOADING_OBJS_INV[obj_id]
                        if col_id in _LEVEL_LOADING_COLOURS_INV:
                            col_str += _LEVEL_LOADING_COLOURS_INV[col_id]
                        else:
                            col_str += "."
                    else:
                        obj_str += "."
                        col_str += "."
            obj_str += "\n"
            col_str += "\n"

        return obj_str, col_str

    def contains(self, xminigrid_obj: chex.Array) -> jnp.bool:
        """
        Returns True if the grid contains the object (tile, color) passed
        as a parameter, and False otherwise.
        """
        return jnp.any(jnp.all(self.grid == xminigrid_obj, axis=2))

    def num_of(self, xminigrid_obj: chex.Array) -> jnp.int_:
        """
        Returns the number of objects matching that passed as a parameter.
        """
        return jnp.sum(jnp.all(self.grid == xminigrid_obj, axis=2))

    def get_colors(self, xminigrid_tile: Tiles) -> chex.Array:
        """
        Returns the colors appearing for a specific tile type.
        """
        return self.grid[self.grid[..., 0] == xminigrid_tile][:, 1]

    def get_agent_mask(self, env_params: XMinigridEnvParams) -> chex.Array:
        """
        Returns a mask of the grid indicating where the agent is placed.
        """
        agent_mask = jnp.zeros((env_params.height, env_params.width), dtype=jnp.bool)
        return agent_mask.at[self.agent_pos[0], self.agent_pos[1]].set(True)

    def is_agent_in_door_pos(self) -> jnp.bool:
        """
        Returns True if the agent is in a position containing a door, and False otherwise.
        """
        return jnp.isin(
            self.grid[tuple(self.agent_pos)][0],
            jnp.array([Tiles.DOOR_CLOSED, Tiles.DOOR_LOCKED, Tiles.DOOR_OPEN], dtype=jnp.uint8)
        )

    def is_pocket_empty(self) -> jnp.bool:
        """
        Returns True if the agent's pocket is empty.
        """
        return jnp.any(self.agent_pocket == jnp.array([Tiles.EMPTY, Colors.EMPTY], dtype=jnp.uint8))


def get_level_sizes(levels: XMinigridLevel) -> chex.Array:
    return jnp.unique(
        jnp.concat((levels.height[:, jnp.newaxis], levels.width[:, jnp.newaxis]), axis=1),
        axis=0
    )


def get_num_objects(levels: XMinigridLevel) -> chex.Array:
    tiles = levels.grid[..., 0]
    return jnp.isin(
        tiles,
        jnp.array([
            Tiles.BALL, Tiles.SQUARE, Tiles.PYRAMID, Tiles.KEY, Tiles.HEX, Tiles.STAR,
            Tiles.DOOR_OPEN, Tiles.DOOR_LOCKED, Tiles.DOOR_CLOSED
        ])
    ).sum(axis=(1, 2))
