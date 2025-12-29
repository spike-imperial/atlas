import itertools
from typing import Callable, List, Tuple

import chex
import jax
import jax.numpy as jnp
from xminigrid.core.constants import Tiles, Colors
from xminigrid.types import State, AgentState

from atlas.envs.common.types import Timestep
from atlas.envs.xminigrid.labeling_function import XMinigridLabelingFunction
from atlas.envs.xminigrid.types import XMinigridEnvParams


class TestLabelingFunction:
    LABELABLE_NON_DOOR_OBJS = [
        Tiles.BALL,
        Tiles.SQUARE,
        Tiles.KEY,
    ]
    LABELABLE_DOOR_OBJS = [Tiles.DOOR_LOCKED, Tiles.DOOR_CLOSED, Tiles.DOOR_OPEN]
    LABELABLE_OBJS = LABELABLE_NON_DOOR_OBJS + LABELABLE_DOOR_OBJS
    LABELABLE_COLORS = [
        Colors.RED,
        Colors.GREEN,
        Colors.BLUE,
        Colors.PURPLE,
        Colors.YELLOW,
        Colors.GREY,
    ]
    ENV_PARAMS = XMinigridEnvParams(
        non_door_obj_types=tuple(LABELABLE_NON_DOOR_OBJS),
        door_obj_types=tuple(LABELABLE_DOOR_OBJS),
        color_types=tuple(LABELABLE_COLORS),
        view_size=3,
    )
    LABELING_FUNCTION = XMinigridLabelingFunction(ENV_PARAMS)

    TILES_TO_STR = {
        Tiles.BALL: "ball",
        Tiles.DOOR_CLOSED: "door",
        Tiles.DOOR_LOCKED: "door",
        Tiles.DOOR_OPEN: "door",
        Tiles.KEY: "key",
        Tiles.SQUARE: "square",
    }

    COLORS_TO_STR = {
        Colors.BLUE: "blue",
        Colors.GREEN: "green",
        Colors.GREY: "grey",
        Colors.PURPLE: "purple",
        Colors.RED: "red",
        Colors.YELLOW: "yellow",
    }

    STATUS_TO_STR = {
        Tiles.DOOR_CLOSED: "closed",
        Tiles.DOOR_LOCKED: "locked",
        Tiles.DOOR_OPEN: "open",
    }

    def _test_expected_prop_list_(self, label: jax.Array, expected_prop_list: List):
        prop_list = self.LABELING_FUNCTION.label_to_prop_list(label)
        assert sorted(prop_list) == sorted(expected_prop_list)

    def _test_label_eq_expected_prop_list(
        self, label_fn: Callable, obs: jax.Array, expected_prop_list: List
    ):
        label = label_fn(obs)
        self._test_expected_prop_list_(label, expected_prop_list)

    def _test_single_obj_prop(
        self,
        label_fn: Callable,
        xminigrid_obj: Tiles,
        xminigrid_color: Colors,
        is_front_prop: bool,
        expected_prop_list: List,
    ):
        obs = jnp.ones(
            (self.ENV_PARAMS.view_size, self.ENV_PARAMS.view_size, 2), dtype=jnp.uint8
        ) * jnp.array([Tiles.EMPTY, Colors.EMPTY])
        agent_position = jnp.array([self.ENV_PARAMS.view_size - 1, self.ENV_PARAMS.view_size // 2])

        if is_front_prop:
            obj_position = (self.ENV_PARAMS.view_size - 2, self.ENV_PARAMS.view_size // 2)
            obs = obs.at[obj_position].set([xminigrid_obj, xminigrid_color])
            pocket = jnp.array([Tiles.EMPTY, Tiles.EMPTY])
        else:
            pocket = jnp.array([xminigrid_obj, xminigrid_color])

        timestep = Timestep(
            key=None,
            state=State(
                key=None,
                step_num=None,
                grid=obs,
                agent=AgentState(position=agent_position, pocket=pocket),
                goal_encoding=None,
                rule_encoding=None,
                carry=None
            ),
            step_type=None,
            reward=None,
            discount=None,
            observation={"ego": obs},
            num_steps=None,
        )

        self._test_label_eq_expected_prop_list(label_fn, timestep, expected_prop_list)

    def _test_front_prop(
        self,
        label_fn: Callable,
        xminigrid_obj: Tiles,
        xminigrid_color: Colors,
        expected_prop_list: List,
    ):
        self._test_single_obj_prop(
            label_fn,
            xminigrid_obj,
            xminigrid_color,
            True,
            expected_prop_list,
        )

    def test_front_prop(self, subtests):
        label_fn = jax.jit(
            chex.assert_max_traces(self.LABELING_FUNCTION.get_label, n=1)
        )

        for obj in self.LABELABLE_OBJS:
            obj_str = self.TILES_TO_STR[obj]
            for color in self.LABELABLE_COLORS:
                color_str = self.COLORS_TO_STR[color]
                status_str = self.STATUS_TO_STR.get(obj, None)

                test_msg = f"{obj_str}_{color_str}"
                if status_str:
                    test_msg += f"_{status_str}"

                with subtests.test(msg=test_msg):
                    expected_prop_list = [
                        f"front_{obj_str}",
                        f"front_{obj_str}_{color_str}",
                    ]

                    if status_str:
                        expected_prop_list.extend(
                            [
                                f"front_{obj_str}_{status_str}",
                                f"front_{obj_str}_{color_str}_{status_str}",
                            ]
                        )

                    self._test_front_prop(label_fn, obj, color, expected_prop_list)

    def _test_carrying_prop(
        self,
        label_fn: Callable,
        xminigrid_obj: Tiles,
        xminigrid_color: Colors,
        expected_prop_list: List,
    ):
        self._test_single_obj_prop(
            label_fn,
            xminigrid_obj,
            xminigrid_color,
            False,
            expected_prop_list,
        )

    def test_carrying_prop(self, subtests):
        label_fn = jax.jit(
            chex.assert_max_traces(self.LABELING_FUNCTION.get_label, n=1)
        )

        for obj in self.LABELABLE_NON_DOOR_OBJS:
            obj_str = self.TILES_TO_STR[obj]
            for color in self.LABELABLE_COLORS:
                color_str = self.COLORS_TO_STR[color]
                with subtests.test(msg=f"{obj_str}_{color_str}"):
                    self._test_carrying_prop(
                        label_fn,
                        obj,
                        color,
                        [
                            f"carrying_{obj_str}",
                            f"carrying_{obj_str}_{color_str}",
                        ],
                    )

    def _test_next_to_prop_symmetry(
        self,
        label_fn: Callable,
        xminigrid_obj1: Tiles,
        xminigrid_col1: Colors,
        xminigrid_obj2: Tiles,
        xminigrid_col2: Colors,
        expected_prop_list: List,
    ):
        obs = jnp.ones(
            (self.ENV_PARAMS.view_size, self.ENV_PARAMS.view_size, 2), dtype=jnp.uint8
        ) * jnp.array([Tiles.EMPTY, Colors.EMPTY])
        obs = (
            obs.at[0, 0]
            .set([xminigrid_obj1, xminigrid_col1])
            .at[0, 1]
            .set([xminigrid_obj2, xminigrid_col2])
        )
        agent_position = jnp.array([self.ENV_PARAMS.view_size - 1, self.ENV_PARAMS.view_size // 2])

        timestep = Timestep(
            key=None,
            state=State(
                key=None,
                step_num=None,
                grid=obs,
                agent=AgentState(position=agent_position),
                goal_encoding=None,
                rule_encoding=None,
                carry=None
            ),
            step_type=None,
            reward=None,
            discount=None,
            observation={"ego": obs},
            num_steps=None,
        )

        self._test_label_eq_expected_prop_list(label_fn, timestep, expected_prop_list)

    def test_next_to_prop_symmetry(self, subtests):
        label_fn = jax.jit(
            chex.assert_max_traces(self.LABELING_FUNCTION.get_label, n=1)
        )

        for obj1_idx, col1_idx, obj2_idx, col2_idx in itertools.product(
            range(len(self.LABELABLE_OBJS)),
            range(len(self.LABELABLE_COLORS)),
            range(len(self.LABELABLE_OBJS)),
            range(len(self.LABELABLE_COLORS)),
        ):
            obj1_str = self.TILES_TO_STR[self.LABELABLE_OBJS[obj1_idx]]
            col1_str = self.COLORS_TO_STR[self.LABELABLE_COLORS[col1_idx]]
            status1_str = self.STATUS_TO_STR.get(self.LABELABLE_OBJS[obj1_idx], "")
            status1_str_msg = f"_{status1_str}" if len(status1_str) > 0 else status1_str

            obj2_str = self.TILES_TO_STR[self.LABELABLE_OBJS[obj2_idx]]
            col2_str = self.COLORS_TO_STR[self.LABELABLE_COLORS[col2_idx]]
            status2_str = self.STATUS_TO_STR.get(self.LABELABLE_OBJS[obj2_idx], "")
            status2_str_msg = f"_{status2_str}" if len(status2_str) > 0 else status2_str

            if obj1_str == "door" and obj2_str == "door":
                break  # it cannot occur

            with subtests.test(
                msg=f"{obj1_str}_{col1_str}{status1_str_msg}_{obj2_str}_{col2_str}{status2_str_msg}"
            ):
                expected_prop_list = set()

                # TODO: By now, assuming only one of the objects will have a status (i.e., is a door).
                for c1, s1, c2, s2 in itertools.product(
                    ["", col1_str], ["", status1_str], ["", col2_str], ["", status2_str]
                ):
                    col1_n_idx = col1_idx if len(c1) > 0 else len(self.LABELABLE_COLORS)
                    col2_n_idx = col2_idx if len(c2) > 0 else len(self.LABELABLE_COLORS)

                    if obj1_idx == obj2_idx:
                        min_obj = max_obj = obj1_str
                        if col1_n_idx < col2_n_idx:
                            min_col, max_col = c1, c2
                            min_status, max_status = s1, s2
                        elif col1_n_idx == col2_n_idx:
                            min_col = max_col = c1
                            min_status = max_status = s1
                        else:
                            min_col, max_col = c2, c1
                            min_status, max_status = s2, s1
                    else:
                        if obj1_idx < obj2_idx:
                            min_obj, max_obj = obj1_str, obj2_str
                            min_col, max_col = c1, c2
                            min_status, max_status = s1, s2
                        elif obj1_idx > obj2_idx:
                            min_obj, max_obj = obj2_str, obj1_str
                            min_col, max_col = c2, c1
                            min_status, max_status = s2, s1

                    proposition = f"next_{min_obj}"
                    if len(min_col) > 0:
                        proposition += f"_{min_col}"
                    if len(min_status) > 0:
                        proposition += f"_{min_status}"
                    proposition += f"_{max_obj}"
                    if len(max_col) > 0:
                        proposition += f"_{max_col}"
                    if len(max_status) > 0:
                        proposition += f"_{max_status}"
                    expected_prop_list.add(proposition)

                self._test_next_to_prop_symmetry(
                    label_fn,
                    self.LABELABLE_OBJS[obj1_idx],
                    self.LABELABLE_COLORS[col1_idx],
                    self.LABELABLE_OBJS[obj2_idx],
                    self.LABELABLE_COLORS[col2_idx],
                    expected_prop_list,
                )

    def test_all_dense_grid(self):
        label_fn = jax.jit(self.LABELING_FUNCTION.get_label)
        obs = jnp.array(
            [
                [
                    [Tiles.EMPTY, Colors.EMPTY],
                    [Tiles.DOOR_LOCKED, Colors.GREEN],
                    [Tiles.EMPTY, Colors.EMPTY],
                ],
                [
                    [Tiles.KEY, Colors.BLUE],
                    [Tiles.SQUARE, Colors.BLUE],
                    [Tiles.KEY, Colors.PURPLE],
                ],
                [
                    [Tiles.BALL, Colors.RED],
                    [Tiles.BALL, Colors.RED],
                    [Tiles.SQUARE, Colors.YELLOW],
                ],
            ]
        )

        agent_position = jnp.array([self.ENV_PARAMS.view_size - 1, self.ENV_PARAMS.view_size // 2])

        timestep = Timestep(
            key=None,
            state=State(
                key=None,
                step_num=None,
                grid=obs,
                agent=AgentState(position=agent_position, pocket=jnp.array([Tiles.BALL, Colors.RED])),
                goal_encoding=None,
                rule_encoding=None,
                carry=None
            ),
            step_type=None,
            reward=None,
            discount=None,
            observation={"ego": obs},
            num_steps=None,
        )

        self._test_label_eq_expected_prop_list(
            label_fn,
            timestep,
            [
                "front_square",
                "front_square_blue",
                "carrying_ball",
                "carrying_ball_red",
                "next_square_blue_door_green_locked",
                "next_square_blue_door_green",
                "next_square_blue_door",
                "next_square_blue_door_locked",
                "next_square_door_green_locked",
                "next_square_door_green",
                "next_square_door_locked",
                "next_square_door",
                "next_square_blue_key_blue",
                "next_square_blue_key",
                "next_square_key_blue",
                "next_square_key",
                "next_ball_red_key_blue",
                "next_ball_red_key",
                "next_ball_key_blue",
                "next_ball_key",
                "next_square_blue_key_purple",
                "next_square_key_purple",
                "next_square_yellow_key_purple",
                "next_square_yellow_key",
            ],
        )

    def test_vmap(self):
        def _init_obs(obj: jax.Array) -> jax.Array:
            obs = jnp.ones(
                (self.ENV_PARAMS.view_size, self.ENV_PARAMS.view_size, 2),
                dtype=jnp.uint8,
            )
            obs *= jnp.array([Tiles.EMPTY, Colors.EMPTY])
            obs = obs.at[
                self.ENV_PARAMS.view_size - 2, self.ENV_PARAMS.view_size // 2
            ].set(obj)

            agent_position = jnp.array([self.ENV_PARAMS.view_size - 1, self.ENV_PARAMS.view_size // 2])

            return Timestep(
                key=None,
                state=State(
                    key=None,
                    step_num=None,
                    grid=obs,
                    agent=AgentState(position=agent_position),
                    goal_encoding=None,
                    rule_encoding=None,
                    carry=None
                ),
                step_type=None,
                reward=None,
                discount=None,
                observation={"ego": obs},
                num_steps=None,
            )

        observations = jax.vmap(lambda obj: _init_obs(obj), in_axes=(0,))(
            jnp.array(
                [
                    [Tiles.BALL, Colors.RED],
                    [Tiles.SQUARE, Colors.GREEN],
                    [Tiles.DOOR_LOCKED, Colors.BLUE],
                ]
            )
        )
        expected_prop_lists = [
            ["front_ball", "front_ball_red"],
            ["front_square", "front_square_green"],
            [
                "front_door",
                "front_door_blue",
                "front_door_locked",
                "front_door_blue_locked",
            ],
        ]

        label_fn = jax.vmap(jax.jit(self.LABELING_FUNCTION.get_label), in_axes=(0,))
        labels = label_fn(observations)
        for i, label in enumerate(labels):
            self._test_expected_prop_list_(label, expected_prop_lists[i])
