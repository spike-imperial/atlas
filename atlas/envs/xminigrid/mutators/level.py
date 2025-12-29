from abc import ABC
from typing import Optional, Tuple

import chex
import jax
from jax import numpy as jnp
from xminigrid.core.constants import Tiles, Colors
from xminigrid.core.grid import free_tiles_mask, sample_direction

from .common import Mutations
from ..level import XMinigridLevel
from ..types import XMinigridEnvParams
from ..utils import (
    clear_coord,
    get_door_obj_mask,
    get_nondoor_obj_mask,
    get_obj_mask,
    insert_obj,
    sample_color_type,
    sample_coordinates,
    sample_door_obj_type,
    sample_nondoor_obj_type,
)


def build_level_mutator(mutation_id: Mutations, env_params: XMinigridEnvParams, max_num_args: int):
    match mutation_id:
        case Mutations.ADD_NON_DOOR_OBJ:
            cls = AddNonDoorObjMutator
        case Mutations.RM_NON_DOOR_OBJ:
            cls = RemoveNonDoorObjMutator
        case Mutations.MOVE_NON_DOOR_OBJ:
            cls = MoveNonDoorObjMutator
        case Mutations.MOVE_AGENT:
            cls = MoveAgentMutator
        case Mutations.REPLACE_DOOR:
            cls = ReplaceDoorMutator
        case Mutations.REPLACE_NON_DOOR:
            cls = ReplaceNonDoorMutator
        case Mutations.ADD_ROOMS:
            cls = AddRoomsMutator
        case Mutations.RM_ROOMS:
            cls = RemoveRoomsMutator

    return cls(env_params, max_num_args)


class LevelMutator(ABC):
    ONE_ROOM_H, ONE_ROOM_W = 7, 7
    TWO_ROOMS_H, TWO_ROOMS_W = 7, 13
    FOUR_ROOMS_H, FOUR_ROOMS_W = 13, 13
    SIX_ROOMS_H, SIX_ROOMS_W = 13, 19

    ONE_ROOM_MAX_OBJS = 5
    TWO_ROOMS_MAX_OBJS = 10
    FOUR_ROOMS_MAX_OBJS = 15
    SIX_ROOMS_MAX_OBJS = 20

    ROOM_SIZES = jnp.array([
        ONE_ROOM_H * ONE_ROOM_W,
        TWO_ROOMS_H * TWO_ROOMS_W,
        FOUR_ROOMS_H * FOUR_ROOMS_W,
        SIX_ROOMS_H * SIX_ROOMS_H,
    ])

    NUM_MAX_OBJS = jnp.array([ONE_ROOM_MAX_OBJS, TWO_ROOMS_MAX_OBJS, FOUR_ROOMS_MAX_OBJS, SIX_ROOMS_MAX_OBJS])

    def __init__(self, env_params: XMinigridEnvParams, max_num_args: int = 2):
        self.env_params = env_params
        self.max_num_args = max_num_args

    def is_applicable(self, level: XMinigridLevel) -> bool:
        raise NotImplementedError

    def apply(self, rng: chex.PRNGKey, level: XMinigridLevel) -> Tuple[XMinigridLevel, chex.Array, chex.Array]:
        raise NotImplementedError

    def _is_one_room(self, level: XMinigridLevel) -> bool:
        return jnp.logical_and(level.height == self.ONE_ROOM_H, level.width == self.ONE_ROOM_W)

    def _is_two_rooms(self, level: XMinigridLevel) -> bool:
        return jnp.logical_and(level.height == self.TWO_ROOMS_H, level.width == self.TWO_ROOMS_W)

    def _is_four_rooms(self, level: XMinigridLevel) -> bool:
        return jnp.logical_and(level.height == self.FOUR_ROOMS_H, level.width == self.FOUR_ROOMS_W)

    def _is_six_rooms(self, level: XMinigridLevel) -> bool:
        return jnp.logical_and(level.height == self.SIX_ROOMS_H, level.width == self.SIX_ROOMS_W)


class AddNonDoorObjMutator(LevelMutator):
    """
    Creates a new level with a new randomly generated object occupying
    a free position in the grid.
    """
    def __init__(self, env_params: XMinigridEnvParams, max_num_args: int = 2):
        assert max_num_args >= 2
        super().__init__(env_params, max_num_args)

    def is_applicable(self, level: XMinigridLevel) -> bool:
        return jnp.any(jnp.logical_and(
            (level.width * level.height) == self.ROOM_SIZES,
            self._get_num_objects(level) < self.NUM_MAX_OBJS
        ))

    def apply(self, rng: chex.PRNGKey, level: XMinigridLevel) -> Tuple[XMinigridLevel, chex.Array, chex.Array]:
        type_rng, col_rng, coord_rng = jax.random.split(rng, 3)

        obj = jnp.array([
            sample_nondoor_obj_type(type_rng, self.env_params)[0],
            sample_color_type(col_rng, self.env_params)[0]
        ], dtype=jnp.uint8)

        coord = sample_coordinates(
            coord_rng, level.grid, num=1, mask=self._get_free_tile_mask(level)
        )[0]

        return (
            level.replace(grid=insert_obj(level.grid, obj, coord)),
            Mutations.ADD_NON_DOOR_OBJ,
            jnp.pad(obj.astype(jnp.int32), pad_width=(0, self.max_num_args - 2))
        )

    def _get_free_tile_mask(self, level: XMinigridLevel) -> chex.Array:
        return free_tiles_mask(level.grid) & jnp.logical_not(level.get_agent_mask(self.env_params))

    def _get_num_objects(self, level: XMinigridLevel) -> chex.Array:
        return jnp.sum(get_obj_mask(level.grid, self.env_params)) + jnp.logical_not(level.is_pocket_empty())


class RemoveNonDoorObjMutator(LevelMutator):
    """
    Creates a new level by removing one of the existing non-door objects in the grid.
    """
    def __init__(self, env_params: XMinigridEnvParams, max_num_args: int = 2):
        assert max_num_args >= 2
        super().__init__(env_params, max_num_args)

    def is_applicable(self, level: XMinigridLevel) -> bool:
        return jnp.sum(get_nondoor_obj_mask(level.grid, self.env_params)) > 0

    def apply(self, rng: chex.PRNGKey, level: XMinigridLevel) -> Tuple[XMinigridLevel, chex.Array, chex.Array]:
        coords = sample_coordinates(
            rng, level.grid, num=1, mask=get_nondoor_obj_mask(level.grid, self.env_params)
        )[0]
        rem_obj = level.grid[coords[0], coords[1]]
        return (
            level.replace(grid=clear_coord(level.grid, coords)),
            Mutations.RM_NON_DOOR_OBJ,
            jnp.pad(rem_obj.astype(jnp.int32), pad_width=(0, self.max_num_args - 2))
        )


class MoveNonDoorObjMutator(LevelMutator):
    """
    Creates a new level by moving a non-door object to a new location.
    """
    def __init__(self, env_params: XMinigridEnvParams, max_num_args: int = 2):
        assert max_num_args >= 2
        super().__init__(env_params, max_num_args)

    def is_applicable(self, level: XMinigridLevel) -> bool:
        return jnp.logical_and(
            jnp.sum(self._get_src_mask(level)) > 0,
            jnp.sum(self._get_dst_mask(level)) > 0
        )

    def apply(self, rng: chex.PRNGKey, level: XMinigridLevel) -> Tuple[XMinigridLevel, chex.Array, chex.Array]:
        obj_coords = sample_coordinates(
            rng, level.grid, num=1, mask=self._get_src_mask(level)
        )[0]
        moved_obj = level.grid[obj_coords[0], obj_coords[1]]
        dst_coords = sample_coordinates(
            rng, level.grid, num=1, mask=self._get_dst_mask(level)
        )[0]
        grid = clear_coord(level.grid, obj_coords)

        return (
            level.replace(grid=insert_obj(grid, moved_obj, dst_coords)),
            Mutations.MOVE_NON_DOOR_OBJ,
            jnp.pad(moved_obj.astype(jnp.int32), pad_width=(0, self.max_num_args - 2))
        )

    def _get_src_mask(self, level: XMinigridLevel) -> chex.Array:
        return get_nondoor_obj_mask(level.grid, self.env_params)

    def _get_dst_mask(self, level: XMinigridLevel) -> chex.Array:
        return free_tiles_mask(level.grid) & jnp.logical_not(level.get_agent_mask(self.env_params))


class MoveAgentMutator(LevelMutator):
    """
    Creates a new level by moving the agent to a new position and changing its orientation.
    """
    def is_applicable(self, level: XMinigridLevel) -> bool:
        return jnp.sum(self._get_dst_mask(level)) > 0

    def apply(self, rng: chex.PRNGKey, level: XMinigridLevel) -> Tuple[XMinigridLevel, chex.Array, chex.Array]:
        dir_rng, loc_rng = jax.random.split(rng, 2)
        return level.replace(
            agent_pos=sample_coordinates(loc_rng, level.grid, num=1, mask=self._get_dst_mask(level))[0],
            agent_dir=sample_direction(dir_rng),
        ), Mutations.MOVE_AGENT, jnp.zeros((self.max_num_args,), dtype=jnp.int32)

    def _get_dst_mask(self, level: XMinigridLevel) -> chex.Array:
        return free_tiles_mask(level.grid) & jnp.logical_not(level.get_agent_mask(self.env_params))


class ReplaceDoorMutator(LevelMutator):
    """
    Creates a new level by changing one of the existing doors (if any).
    """
    def __init__(self, env_params: XMinigridEnvParams, max_num_args: int = 2):
        assert max_num_args >= 2
        super().__init__(env_params, max_num_args)

    def is_applicable(self, level: XMinigridLevel) -> bool:
        return jnp.sum(get_door_obj_mask(level.grid, self.env_params)) > 0

    def apply(self, rng: chex.PRNGKey, level: XMinigridLevel) -> Tuple[XMinigridLevel, chex.Array, chex.Array]:
        type_rng, col_rng, coord_rng = jax.random.split(rng, 3)

        obj = jnp.array([
            sample_door_obj_type(type_rng, self.env_params)[0],
            sample_color_type(col_rng, self.env_params)[0],
        ], dtype=jnp.uint8)
        coords = sample_coordinates(
            coord_rng, level.grid, num=1, mask=get_door_obj_mask(level.grid, self.env_params)
        )[0]

        return (
            level.replace(grid=insert_obj(level.grid, obj, coords)),
            Mutations.REPLACE_DOOR,
            jnp.pad(obj.astype(jnp.int32), pad_width=(0, self.max_num_args - 2))
        )


class ReplaceNonDoorMutator(LevelMutator):
    """
    Creates a new level by changing one of the non-door objects.
    """
    def __init__(self, env_params: XMinigridEnvParams, max_num_args: int = 2):
        assert max_num_args >= 2
        super().__init__(env_params, max_num_args)

    def is_applicable(self, level: XMinigridLevel) -> bool:
        return jnp.sum(get_nondoor_obj_mask(level.grid, self.env_params)) > 0

    def apply(self, rng: chex.PRNGKey, level: XMinigridLevel) -> Tuple[XMinigridLevel, chex.Array]:
        type_rng, col_rng, coord_rng = jax.random.split(rng, 3)

        obj = jnp.array([
            sample_nondoor_obj_type(type_rng, self.env_params)[0],
            sample_color_type(col_rng, self.env_params)[0],
        ], dtype=jnp.uint8)
        coords = sample_coordinates(
            coord_rng, level.grid, num=1, mask=get_nondoor_obj_mask(level.grid, self.env_params)
        )[0]

        return (
            level.replace(grid=insert_obj(level.grid, obj, coords)),
            Mutations.REPLACE_NON_DOOR,
            jnp.pad(obj.astype(jnp.int32), pad_width=(0, self.max_num_args - 2))
        )


class AddRoomsMutator(LevelMutator):
    """
    Extends levels from one room to two rooms, two rooms to four rooms and
    four rooms to six rooms. The size of the input levels are fixed (for
    simplicity). Only new doors are added for introduced corridors. If the
    source problem was solvable, the new problem should be too since only "confounder"
    new paths are added.
    """
    def is_applicable(self, level: XMinigridLevel) -> bool:
        return self._is_one_room(level) | self._is_two_rooms(level) | self._is_four_rooms(level)

    def apply(self, rng: chex.PRNGKey, level: XMinigridLevel) -> Tuple[XMinigridLevel, chex.Array, chex.Array]:
        def _extend_one_room():
            def _add_left_room():
                # Move the grid, extend it with black tiles
                grid = jnp.roll(level.grid, self.ONE_ROOM_W - 1, axis=1)
                grid = grid.at[:self.ONE_ROOM_H, :self.ONE_ROOM_W].set(
                    jnp.array([Tiles.FLOOR, Colors.BLACK], dtype=jnp.uint8)
                )
                agent_pos = level.agent_pos + jnp.array([0, self.ONE_ROOM_W - 1], dtype=jnp.uint8)
                return agent_pos, grid

            def _add_right_room():
                # Extend the grid and add vertical wall
                grid = level.grid.at[:self.ONE_ROOM_H, self.ONE_ROOM_W:self.TWO_ROOMS_W].set(
                    jnp.array([Tiles.FLOOR, Colors.BLACK], dtype=jnp.uint8)
                )
                return level.agent_pos, grid

            side_rng, type_rng, col_rng = jax.random.split(rng, 3)

            # Add a room
            agent_pos, grid = jax.lax.cond(jax.random.bernoulli(side_rng), _add_left_room, _add_right_room)

            # Add the horizontal walls
            grid = grid.at[0, :self.TWO_ROOMS_W].set(jnp.array([Tiles.WALL, Colors.GREY], dtype=jnp.uint8))
            grid = grid.at[self.ONE_ROOM_H - 1, :self.TWO_ROOMS_W].set(jnp.array([Tiles.WALL, Colors.GREY], dtype=jnp.uint8))

            # Add the vertical walls
            grid = grid.at[:self.ONE_ROOM_H, (0, self.ONE_ROOM_W - 1, self.TWO_ROOMS_W - 1)].set(
                jnp.array([Tiles.WALL, Colors.GREY], dtype=jnp.uint8)
            )

            # Sample a door
            obj = jnp.array([
                sample_door_obj_type(type_rng, self.env_params)[0],
                sample_color_type(col_rng, self.env_params)[0],
            ], dtype=jnp.uint8)

            # Add the door
            grid = grid.at[self.ONE_ROOM_H // 2, self.ONE_ROOM_W - 1].set(obj)

            return level.replace(width=self.TWO_ROOMS_W, grid=grid, agent_pos=agent_pos)

        def _extend_two_rooms():
            def _add_top_rooms():
                # Move the grid, extend it with black tiles, add vertical wall
                grid = jnp.roll(level.grid, self.ONE_ROOM_H - 1, axis=0)
                grid = grid.at[:self.ONE_ROOM_H - 1, :self.TWO_ROOMS_W].set(
                    jnp.array([Tiles.FLOOR, Colors.BLACK], dtype=jnp.uint8)
                )
                grid = grid.at[:self.TWO_ROOMS_H, self.TWO_ROOMS_W // 2].set(
                    jnp.array([Tiles.WALL, Colors.GREY], dtype=jnp.uint8)
                )
                agent_pos = level.agent_pos + jnp.array([self.ONE_ROOM_H - 1, 0], dtype=jnp.uint8)
                door_row = self.FOUR_ROOMS_H // 4
                return agent_pos, grid, door_row

            def _add_bottom_rooms():
                # Extend the size of the grid
                grid = level.grid.at[self.TWO_ROOMS_H:self.FOUR_ROOMS_H, :self.TWO_ROOMS_W].set(
                    jnp.array([Tiles.FLOOR, Colors.BLACK], dtype=jnp.uint8)
                )
                grid = grid.at[self.TWO_ROOMS_H:self.FOUR_ROOMS_H, self.TWO_ROOMS_W // 2].set(
                    jnp.array([Tiles.WALL, Colors.GREY], dtype=jnp.uint8)
                )
                door_row = 3 * self.FOUR_ROOMS_H // 4
                return level.agent_pos, grid, door_row

            side_rng, type_rng, col_rng = jax.random.split(rng, 3)

            # Add rooms
            agent_pos, grid, door_row = jax.lax.cond(jax.random.bernoulli(side_rng), _add_top_rooms, _add_bottom_rooms)

            # Add the horizontal walls
            grid = grid.at[(0, self.FOUR_ROOMS_H - 1), :self.TWO_ROOMS_W].set(jnp.array([Tiles.WALL, Colors.GREY], dtype=jnp.uint8))

            # Add the common vertical walls
            grid = grid.at[:self.FOUR_ROOMS_H, (0, self.TWO_ROOMS_W - 1)].set(jnp.array([Tiles.WALL, Colors.GREY], dtype=jnp.uint8))

            # Sample doors
            objs = sample_door_obj_type(type_rng, self.env_params, n=3)
            colors = sample_color_type(col_rng, self.env_params, n=3)
            objs = jnp.concat((objs[:, jnp.newaxis], colors[:, jnp.newaxis]), axis=1).astype(jnp.uint8)

            # Add the doors
            door_rows = jnp.array([self.FOUR_ROOMS_H // 2, self.FOUR_ROOMS_H // 2, door_row])
            door_cols = jnp.array([self.TWO_ROOMS_W // 4, 3 * self.TWO_ROOMS_W // 4, self.TWO_ROOMS_W // 2])
            grid = grid.at[door_rows, door_cols].set(objs)

            return level.replace(height=self.FOUR_ROOMS_H, grid=grid, agent_pos=agent_pos)

        def _extend_four_rooms():
            def _add_left_rooms():
                grid = jnp.roll(level.grid, self.ONE_ROOM_W - 1, axis=1)
                grid = grid.at[:self.FOUR_ROOMS_H, :self.ONE_ROOM_W - 1].set(
                    jnp.array([Tiles.FLOOR, Colors.BLACK], dtype=jnp.uint8)
                )
                grid = grid.at[self.FOUR_ROOMS_H // 2, :self.ONE_ROOM_W].set(
                    jnp.array([Tiles.WALL, Colors.GREY], dtype=jnp.uint8)
                )
                agent_pos = level.agent_pos + jnp.array([0, self.ONE_ROOM_W - 1], dtype=jnp.uint8)
                door_cols = jnp.array([self.TWO_ROOMS_W // 4, self.ONE_ROOM_W - 1, self.ONE_ROOM_W - 1])
                return agent_pos, grid, door_cols

            def _add_right_rooms():
                grid = level.grid.at[:self.FOUR_ROOMS_H, self.FOUR_ROOMS_W:self.SIX_ROOMS_W].set(
                    jnp.array([Tiles.FLOOR, Colors.BLACK], dtype=jnp.uint8)
                )
                grid = grid.at[self.FOUR_ROOMS_H // 2, self.FOUR_ROOMS_W:self.SIX_ROOMS_W].set(
                    jnp.array([Tiles.WALL, Colors.GREY], dtype=jnp.uint8)
                )
                door_cols = jnp.array([5 * (self.SIX_ROOMS_W - 1) // 6, 2 * self.SIX_ROOMS_W // 3, 2 * self.SIX_ROOMS_W // 3])
                return level.agent_pos, grid, door_cols

            side_rng, type_rng, col_rng = jax.random.split(rng, 3)

            # Add rooms
            agent_pos, grid, door_cols = jax.lax.cond(jax.random.bernoulli(side_rng), _add_left_rooms, _add_right_rooms)

            # Add the common horizontal and vertical walls
            grid = grid.at[(0, self.FOUR_ROOMS_H - 1), :self.SIX_ROOMS_W].set(jnp.array([Tiles.WALL, Colors.GREY], dtype=jnp.uint8))
            grid = grid.at[:self.FOUR_ROOMS_H, (0, self.SIX_ROOMS_W - 1)].set(jnp.array([Tiles.WALL, Colors.GREY], dtype=jnp.uint8))

            # Sample doors
            objs = sample_door_obj_type(type_rng, self.env_params, n=3)
            colors = sample_color_type(col_rng, self.env_params, n=3)
            objs = jnp.concat((objs[:, jnp.newaxis], colors[:, jnp.newaxis]), axis=1).astype(jnp.uint8)

            # Add the doors
            door_rows = jnp.array([self.FOUR_ROOMS_H // 2, self.FOUR_ROOMS_H // 4, 3 * self.FOUR_ROOMS_H // 4])
            grid = grid.at[door_rows, door_cols].set(objs)

            return level.replace(width=self.SIX_ROOMS_W, grid=grid, agent_pos=agent_pos)

        out_level = jax.lax.switch(
            index=self._is_one_room(level) + 2 * self._is_two_rooms(level) + 3 * self._is_four_rooms(level),
            branches=[lambda: level, _extend_one_room, _extend_two_rooms, _extend_four_rooms]
        )

        return (
            out_level,
            Mutations.ADD_ROOMS,
            jnp.zeros((self.max_num_args,), dtype=jnp.int32)
        )


class RemoveRoomsMutator(LevelMutator):
    def is_applicable(self, level: XMinigridLevel) -> bool:
        """
        Returns true if the grid consists of 2, 4 or 6 rooms, and the agent is not in a door position
        (otherwise, the removal of rooms may lead to a layout without the agent).
        """
        is_applicable_room_num = self._is_two_rooms(level) | self._is_four_rooms(level) | self._is_six_rooms(level)
        return is_applicable_room_num & jnp.logical_not(level.is_agent_in_door_pos())

    def apply(self, rng: chex.PRNGKey, level: XMinigridLevel) -> Tuple[XMinigridLevel, chex.Array, chex.Array]:
        return jax.lax.switch(
            self._is_four_rooms(level) + 2 * self._is_six_rooms(level),
            [self._apply_two_rooms, self._apply_four_rooms, self._apply_six_rooms],
            rng, level
        ), Mutations.RM_ROOMS, jnp.zeros((self.max_num_args,), dtype=jnp.int32)

    def _rm_nondoor_objs(
        self, rng: chex.PRNGKey, grid: chex.Array, tgt_num: int, max_to_rm: int, num_pocket_items: int
    ) -> chex.Array:
        """
        Removes the non-door objects that exceed the target number of items in the grid.
        """
        def _f(_grid, xs):
            _rng, _remove = xs
            return jax.lax.cond(
                pred=_remove,
                true_fun=lambda: clear_coord(
                    _grid,
                    sample_coordinates(_rng, _grid, num=1, mask=get_nondoor_obj_mask(_grid, self.env_params))[0]
                ),
                false_fun=lambda: _grid
            ), None

        num_to_rm = jnp.sum(get_obj_mask(grid, self.env_params)) + num_pocket_items - tgt_num
        rngs = jax.random.split(rng, max_to_rm)
        return jax.lax.scan(_f, grid, xs=(rngs, num_to_rm > jnp.arange(max_to_rm)))[0]

    def _apply_two_rooms(self, rng: chex.PRNGKey, level: XMinigridLevel) -> XMinigridLevel:
        def _left_room():
            grid = level.grid[:self.TWO_ROOMS_H, :self.ONE_ROOM_W]
            grid = grid.at[:self.TWO_ROOMS_H, self.ONE_ROOM_W - 1].set(
                jnp.array([Tiles.WALL, Colors.GREY], dtype=jnp.uint8)
            )
            return grid, level.agent_pos

        def _right_room():
            grid = level.grid[:self.TWO_ROOMS_H, self.ONE_ROOM_W - 1:self.TWO_ROOMS_W]
            grid = grid.at[:self.TWO_ROOMS_H, 0].set(jnp.array([Tiles.WALL, Colors.GREY], dtype=jnp.uint8))
            agent_pos = level.agent_pos - jnp.array([0, self.ONE_ROOM_W - 1])
            return grid, agent_pos

        grid, agent_pos = jax.lax.cond(level.agent_pos[1] < self.ONE_ROOM_W, _left_room, _right_room)
        num_pocket_items = jnp.logical_not(level.is_pocket_empty()).astype(jnp.int32)
        grid = self._rm_nondoor_objs(
            rng, grid, self.ONE_ROOM_MAX_OBJS, self.TWO_ROOMS_MAX_OBJS - self.ONE_ROOM_MAX_OBJS, num_pocket_items
        )
        return level.replace(
            grid=self._pad_grid(grid, self.env_params.height - self.ONE_ROOM_H, self.env_params.width - self.ONE_ROOM_W),
            width=self.ONE_ROOM_W,
            agent_pos=agent_pos,
        )

    def _apply_four_rooms(self, rng: chex.PRNGKey, level: XMinigridLevel) -> XMinigridLevel:
        def _top_rooms():
            grid = level.grid[:self.TWO_ROOMS_H, :self.TWO_ROOMS_W]
            grid = grid.at[self.TWO_ROOMS_H - 1, :self.TWO_ROOMS_W].set(
                jnp.array([Tiles.WALL, Colors.GREY], dtype=jnp.uint8)
            )
            return grid, level.agent_pos

        def _bot_rooms():
            grid = level.grid[self.TWO_ROOMS_H - 1: self.FOUR_ROOMS_H, :self.TWO_ROOMS_W]
            grid = grid.at[0, :self.TWO_ROOMS_W].set(jnp.array([Tiles.WALL, Colors.GREY], dtype=jnp.uint8))
            agent_pos = level.agent_pos - jnp.array([self.TWO_ROOMS_H - 1, 0])
            return grid, agent_pos

        grid, agent_pos = jax.lax.cond(level.agent_pos[0] < self.TWO_ROOMS_H, _top_rooms, _bot_rooms)
        num_pocket_items = jnp.logical_not(level.is_pocket_empty()).astype(jnp.int32)
        grid = self._rm_nondoor_objs(
            rng, grid, self.TWO_ROOMS_MAX_OBJS, self.FOUR_ROOMS_MAX_OBJS - self.TWO_ROOMS_MAX_OBJS, num_pocket_items
        )
        return level.replace(
            grid=self._pad_grid(grid, self.env_params.height - self.TWO_ROOMS_H, self.env_params.width - self.TWO_ROOMS_W),
            height=self.TWO_ROOMS_H,
            agent_pos=agent_pos
        )

    def _apply_six_rooms(self, rng: chex.PRNGKey, level: XMinigridLevel) -> XMinigridLevel:
        def _left_center():
            grid = level.grid[:self.SIX_ROOMS_H, :self.FOUR_ROOMS_W]
            grid = grid.at[:self.FOUR_ROOMS_H, self.FOUR_ROOMS_W - 1].set(jnp.array([Tiles.WALL, Colors.GREY], dtype=jnp.uint8))
            return grid, level.agent_pos

        def _right_center():
            grid = level.grid[:self.SIX_ROOMS_H, self.ONE_ROOM_W - 1:self.SIX_ROOMS_W]
            grid = grid.at[:self.FOUR_ROOMS_H, 0].set(jnp.array([Tiles.WALL, Colors.GREY], dtype=jnp.uint8))
            agent_pos = level.agent_pos - jnp.array([0, self.ONE_ROOM_W - 1])
            return grid, agent_pos

        room_rng, rm_rng = jax.random.split(rng)

        is_left_col = level.agent_pos[1] < self.ONE_ROOM_W
        is_right_col = level.agent_pos[1] >= self.FOUR_ROOMS_W
        use_left_cold_rand = jax.random.choice(room_rng, 2)

        grid, agent_pos = jax.lax.cond(
            is_left_col | (~is_left_col & ~is_right_col & (use_left_cold_rand == 0)),
            _left_center,
            _right_center
        )
        num_pocket_items = jnp.logical_not(level.is_pocket_empty()).astype(jnp.int32)
        grid = self._rm_nondoor_objs(
            rm_rng, grid, self.FOUR_ROOMS_MAX_OBJS, self.SIX_ROOMS_MAX_OBJS - self.FOUR_ROOMS_MAX_OBJS, num_pocket_items
        )
        return level.replace(
            grid=self._pad_grid(grid, self.env_params.height - self.FOUR_ROOMS_H, self.env_params.width - self.FOUR_ROOMS_W),
            width=self.FOUR_ROOMS_W,
            agent_pos=agent_pos,
        )

    def _pad_grid(self, grid: chex.Array, padding_h: int, padding_w: int) -> chex.Array:
        return jnp.pad(
            grid,
            pad_width=((0, padding_h), (0, padding_w), (0, 0)),
            constant_values=jnp.array([Tiles.EMPTY, Colors.EMPTY], dtype=jnp.uint8)
        )


class LevelSequenceMutator(LevelMutator):
    def __init__(
        self,
        num_edits: int,
        use_move_agent: bool,
        use_add_rm_objs: bool,
        use_add_rm_rooms: bool,
        env_params: XMinigridEnvParams,
        max_num_args: int = 2,
        max_num_edits: Optional[int] = None,
    ):
        super().__init__(env_params, max_num_args)

        self.num_edits = num_edits
        self.max_num_edits = max_num_edits

        mutation_ids = [Mutations.MOVE_NON_DOOR_OBJ, Mutations.REPLACE_DOOR, Mutations.REPLACE_NON_DOOR]
        if use_move_agent:
            mutation_ids.append(Mutations.MOVE_AGENT)
        if use_add_rm_objs:
            mutation_ids.extend([Mutations.ADD_NON_DOOR_OBJ, Mutations.RM_NON_DOOR_OBJ])
        if use_add_rm_rooms:
            mutation_ids.extend([Mutations.ADD_ROOMS, Mutations.RM_ROOMS])

        self.mutators = [
            build_level_mutator(mutation_id, env_params, max_num_args)
            for mutation_id in mutation_ids
        ]

    def is_applicable(self, level: XMinigridLevel) -> bool:
        return jnp.any(jnp.array([m.is_applicable(level) for m in self.mutators]))

    def apply(self, rng: chex.PRNGKey, level: XMinigridLevel) -> Tuple[XMinigridLevel, chex.Array, chex.Array]:
        def _mutate_aux(lvl: XMinigridLevel, _rng: chex.PRNGKey):
            idx_rng, mutate_rng = jax.random.split(_rng)
            cond_mask = jnp.array([f.is_applicable(lvl) for f in self.mutators])
            idx = jax.random.choice(idx_rng, len(self.mutators), p=cond_mask / cond_mask.sum())
            level, mutation_id, mutation_args = jax.lax.switch(
                idx, [f.apply for f in self.mutators], mutate_rng, lvl
            )
            return level, (mutation_id, mutation_args)

        next_level, (out_ids, out_args) = jax.lax.scan(
            _mutate_aux,
            init=level,
            xs=jnp.array(jax.random.split(rng, self.num_edits)),
        )

        if self.max_num_edits:
            out_ids = jnp.pad(out_ids, pad_width=(0, self.max_num_edits - self.num_edits), constant_values=-1)
            out_args = jnp.pad(out_args, pad_width=((0, self.max_num_edits - self.num_edits), (0, 0)))

        return next_level, out_ids, out_args
