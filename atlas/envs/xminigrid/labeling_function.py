import functools
from itertools import product
from typing import List, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from xminigrid.core.constants import Colors, Tiles, DIRECTIONS, NUM_COLORS, NUM_TILES, PICKABLE

from .level import XMinigridLevel
from .types import XMinigridEnvParams
from .utils import is_xminigrid_door, xminigrid_obj_to_str, xminigrid_color_to_str, xminigrid_status_to_str
from ..common.labeling_function import LabelingFunction
from ..common.types import Timestep


class XMinigridLabelingFunction(LabelingFunction):
    """
    Implementation of the labeling function for XLand-Minigrid.

    Propositions are of the following form:
      - `front_{obj}_{color}_{status}` capture the object in front of the agent,
        where `obj` is the name of the object, `color` is the color of the object
        and `status` applies (by now) to doors (`locked`, `open`, `closed`).
      - `carrying_{obj}_{color}` capture the object being carried by the agent,
        where `obj` and `color` are as above. The status is not capture by now
        since it only applies to doors, which cannot be carried.
      - `next_{obj1}_{color1}_{status1}_{obj2}_{color2}_{status2}` capture two
        adjacent objects within the field of view of the agent.

    The `color` and `status` fields are optional (e.g., we can just express that
    there is a door in front of us regardless of its color and status using
    `front_door`).

    The `next` propositions are such that symmetries are broken, e.g. a proposition
    `next_ball_square` is semantically equivalent to `next_square_ball`. The symmetry
    breaking is enforced through the indices of the objects in the arrays
    (see constructor method). They also assume no two doors will ever be adjacent to
    each other, i.e. there is no proposition `next_door_door`.
    """

    def __init__(
        self,
        env_params: XMinigridEnvParams,
        use_front_props: bool = True,
        use_carrying_props: bool = True,
        use_next_to_props: bool = True,
        labelable_non_door_obj_types: Optional[Tuple] = None,
        labelable_door_obj_types: Optional[Tuple] = None,
        labelable_color_types: Optional[Tuple] = None,
    ):
        """
        Initializes the labeling function. The objects and colors within `env_params`
        will be captured by the propositions.

        Args:
            - env_params: environment parameters that determine the objects and colors
                that determine the propositions.
            - use_front_props: whether the `front` propositions are detected.
            - use_carrying_props: whether the `carrying` propositions are detected.
            - use_next_to_props: whether the `next to` propositions are detected.
            - labelable_non_door_obj_types: tuple containing the non-door objects that
                will be labeled (if None, those in `env_params` are used).
            - labelable_door_obj_types: tuple containing the door objects that
                will be labeled (if None, those in `env_params` are used).
            - labelable_color_types: tuple containing the colors that
                will be labeled (if None, those in `env_params` are used).
        """
        # The labels are extracted from the egocentric observation if it is used (even
        # if the full one is specified too!)
        self._use_ego_obs = env_params.use_ego_obs

        assert use_front_props or use_carrying_props or use_next_to_props

        self.use_front_props = use_front_props
        self.use_carrying_props = use_carrying_props
        self.use_next_to_props = use_next_to_props

        # The original Minigrid objects and colors are considered labelable
        self._labelable_non_door_obj_types: jax.Array = jnp.array(
            env_params.non_door_obj_types
            if labelable_non_door_obj_types is None
            else labelable_non_door_obj_types
        )
        self._labelable_door_obj_types: jax.Array = jnp.array(
            env_params.door_obj_types
            if labelable_door_obj_types is None
            else labelable_door_obj_types
        )
        self._labelable_color_types: jax.Array = jnp.array(
            env_params.color_types
            if labelable_color_types is None
            else labelable_color_types
        )

        # All XLand-Minigrid door objects will be mapped into the same object id and
        # different status ids
        self._num_labelable_objs: int = len(self._labelable_non_door_obj_types) + (
            len(self._labelable_door_obj_types) > 0
        )

        # Sum 1 to account for the case where color is not reported
        self._num_colors = len(self._labelable_color_types) + 1

        # Mappings from XLand-Minigrid object ids to our ids and their status,
        # and from XLand-Minigrid color ids to our ids
        self._object_ids: jax.Array = -jnp.ones((NUM_TILES,), dtype=jnp.int32)
        self._status_ids: jax.Array = jnp.zeros((NUM_TILES,), dtype=jnp.int32)
        self._color_ids: jax.Array = -jnp.ones((NUM_COLORS,), dtype=jnp.int32)

        for i, xminigrid_obj in enumerate(self._labelable_non_door_obj_types):
            self._object_ids = self._object_ids.at[xminigrid_obj].set(i)
        for i, xminigrid_obj in enumerate(self._labelable_door_obj_types):
            self._object_ids = self._object_ids.at[xminigrid_obj].set(
                self._num_labelable_objs - 1
            )
            self._status_ids = self._status_ids.at[xminigrid_obj].set(i)
        for i, xminigrid_color in enumerate(self._labelable_color_types):
            self._color_ids = self._color_ids.at[xminigrid_color].set(i)

        # Number of status for each of our objects (all objects plus a unified door type)
        self._num_status_per_obj = jnp.zeros(
            (self._num_labelable_objs,), dtype=jnp.int32
        )
        if len(self._labelable_door_obj_types) > 0:
            self._num_status_per_obj = self._num_status_per_obj.at[
                self._num_labelable_objs - 1
            ].set(len(self._labelable_door_obj_types))

        # Whether an object type can be carried
        self._pickable_mask = jnp.isin(
            self._labelable_non_door_obj_types, PICKABLE, assume_unique=True
        )
        if len(self._labelable_door_obj_types):
            self._pickable_mask = jnp.concat(
                (
                    self._pickable_mask,
                    jnp.isin(jnp.array([Tiles.DOOR_LOCKED]), PICKABLE),
                )
            )

        # A tensor mapping (object, color, status, object, color, status) tuples
        # into a proposition identifier + a dictionary performing the inverse operation.
        self._next_to_props, self._next_to_prop_to_obj = self._init_next_to_structs()

        # The number of propositions of each type
        self._num_front_props = int(self.use_front_props * jnp.sum(
            (self._num_status_per_obj + 1) * self._num_colors
        ))
        self._num_carrying_props = int(self.use_carrying_props * jnp.sum(
            (self._num_status_per_obj + 1) * self._num_colors * self._pickable_mask
        ))
        self._num_next_to_props = int(self.use_next_to_props * len(
            self._next_to_prop_to_obj
        ))

    def _init_next_to_structs(self):
        """
        Returns structures that establish a mapping between <object, color, status,
        object, color, status> and proposition identifiers (and vice versa). The
        structures capture symmetries between propositions, e.g. <ball, red, _,
        square, blue, _> and <square, blue, _, ball, red, _> are mapped into the
        same proposition identifier.
        """
        max_num_status = self._num_status_per_obj.max() + 1
        next_to_props = jnp.zeros(
            (
                self._num_labelable_objs,
                self._num_colors,
                max_num_status,
                self._num_labelable_objs,
                self._num_colors,
                max_num_status,
            )
        )
        prop_to_obj = []

        if not self.use_next_to_props:
            return next_to_props, prop_to_obj

        prop_id = 0
        for obj1 in range(len(self._labelable_non_door_obj_types)):
            for color1 in range(self._num_colors):
                for status1 in range(self._num_status_per_obj[obj1] + 1):
                    for obj2 in range(obj1, self._num_labelable_objs):
                        color2_start = color1 if obj1 == obj2 else 0
                        for color2 in range(color2_start, self._num_colors):
                            status2_start = (
                                status1 if (obj1, color1) == (obj2, color2) else 0
                            )
                            for status2 in range(
                                status2_start, self._num_status_per_obj[obj2] + 1
                            ):
                                next_to_props = next_to_props.at[
                                    obj1, color1, status1, obj2, color2, status2
                                ].set(prop_id)
                                next_to_props = next_to_props.at[
                                    obj2, color2, status2, obj1, color1, status1
                                ].set(prop_id)
                                prop_to_obj.append((
                                    obj1, color1, status1, obj2, color2, status2
                                ))
                                prop_id += 1

        return next_to_props, jnp.asarray(prop_to_obj)

    def get_alphabet_size(self) -> int:
        return self._num_front_props + self._num_carrying_props + self._num_next_to_props

    def get_label(self, timestep: Timestep) -> jax.Array:
        return jnp.concat(
            (
                self._get_front_label(timestep),
                self._get_carrying_label(timestep),
                self._get_next_to_label(timestep),
            )
        )

    def _is_labelable_object(self, xminigrid_obj: int) -> jnp.bool_:
        return self._object_ids[xminigrid_obj] >= 0

    def _get_front_base_label(self) -> jax.Array:
        """
        Returns the label associated with the front propositions where none of them
        is observed.
        """
        return -jnp.ones((self._num_front_props,), dtype=jnp.int32)

    def _get_front_label(self, timestep: Timestep) -> jax.Array:
        """
        Returns the label associated with the `front` propositions.
        """
        if not self.use_front_props:
            return jnp.array([])

        direction = jax.lax.dynamic_index_in_dim(DIRECTIONS, timestep.state.agent.direction, keepdims=False)
        front_pos = timestep.state.agent.position + direction
        front_obj, front_color = timestep.state.grid[tuple(front_pos)]

        return jax.lax.cond(
            pred=self._is_labelable_object(front_obj),
            true_fun=lambda: self._get_object_label(
                front_obj,
                front_color,
                self._get_front_base_label(),
                jnp.ones((self._num_labelable_objs,), dtype=jnp.bool_),
            ),
            false_fun=self._get_front_base_label,
        )

    def _is_carriable_object(self, xminigrid_obj: int) -> jnp.bool_:
        """
        Returns true if the object is labelable (i.e. considered for
        'propositionalization') and can be picked up according to XLand-Minigrid.
        """
        return (
            self._is_labelable_object(xminigrid_obj)
            & self._pickable_mask[self._object_ids[xminigrid_obj]]
        )

    def _get_carrying_base_label(self) -> jax.Array:
        """
        Returns the label associated with the `front` propositions where none of them
        is observed.
        """
        return -jnp.ones((self._num_carrying_props,), dtype=jnp.int32)

    def _get_carrying_label(self, timestep: Timestep) -> jax.Array:
        """
        Returns the label associated with the `carrying` propositions.
        """
        if not self.use_carrying_props:
            return jnp.array([])

        carried_obj, carried_color = tuple(timestep.state.agent.pocket)

        return jax.lax.cond(
            pred=self._is_carriable_object(carried_obj),
            true_fun=lambda: self._get_object_label(
                carried_obj,
                carried_color,
                self._get_carrying_base_label(),
                self._pickable_mask,
            ),
            false_fun=self._get_carrying_base_label,
        )

    def _get_object_label(
        self,
        xminigrid_obj: Tiles,
        xminigrid_color: Colors,
        base_label: jax.Array,
        obj_mask: jax.Array,
    ) -> jax.Array:
        """
        Returns the label associated with a single object (`front` and `carrying`
        propositions).
        """
        obj_id = self._object_ids[xminigrid_obj]
        color_id = self._color_ids[xminigrid_color]
        status_id = self._status_ids[xminigrid_obj]

        # Compute the index of the object whose propositions we are going to set
        mask = jnp.arange(0, self._num_labelable_objs) < obj_id
        base_idx = jnp.sum(
            mask * (self._num_status_per_obj + 1) * self._num_colors * obj_mask
        )

        # Propositions without the status (the second case is to account for the case
        # where the color is not reported)
        no_status_idx = base_idx + self._num_status_per_obj[obj_id] * self._num_colors
        label = base_label.at[no_status_idx + color_id].set(1)
        label = label.at[no_status_idx + self._num_colors - 1].set(1)

        # Propositions with the status (does the same as above if the object does not
        # have multiple status, i.e. it is not a door)
        status_idx = base_idx + status_id * self._num_colors
        label = label.at[status_idx + color_id].set(1)
        label = label.at[status_idx + self._num_colors - 1].set(1)

        return label

    def _get_next_to_base_label(self) -> jax.Array:
        """
        Returns the label associated with the `next to` propositions where none of them
        is observed.
        """
        return -jnp.ones((self._num_next_to_props,), dtype=jnp.int32)

    def _get_next_to_label_obj_pair(
        self,
        xminigrid_obj1: Tiles,
        xminigrid_col1: Colors,
        xminigrid_obj2: Tiles,
        xminigrid_col2: Colors,
    ) -> jax.Array:
        """
        Returns the label associated with a pair of adjacent objects.
        """
        obj1 = self._object_ids[xminigrid_obj1]
        obj2 = self._object_ids[xminigrid_obj2]

        def _get_proposition(obj1_color, obj1_status, obj2_color, obj2_status):
            return self._next_to_props[
                obj1, obj1_color, obj1_status, obj2, obj2_color, obj2_status
            ]

        # Compute the proposition for each possible combination of taking into
        # account the color of the objects and their status or not
        col1 = self._color_ids[xminigrid_col1]
        col2 = self._color_ids[xminigrid_col2]
        ncol = self._num_colors - 1

        status1 = self._status_ids[xminigrid_obj1]
        status2 = self._status_ids[xminigrid_obj2]
        nstatus1 = self._num_status_per_obj[obj1]
        nstatus2 = self._num_status_per_obj[obj2]

        propositions = jnp.array([
            _get_proposition(col1, status1, col2, status2),
            _get_proposition(col1, status1, col2, nstatus2),
            _get_proposition(col1, status1, ncol, status2),
            _get_proposition(col1, status1, ncol, nstatus2),
            _get_proposition(col1, nstatus1, col2, status2),
            _get_proposition(col1, nstatus1, col2, nstatus2),
            _get_proposition(col1, nstatus1, ncol, status2),
            _get_proposition(col1, nstatus1, ncol, nstatus2),
            _get_proposition(ncol, status1, col2, status2),
            _get_proposition(ncol, status1, col2, nstatus2),
            _get_proposition(ncol, status1, ncol, status2),
            _get_proposition(ncol, status1, ncol, nstatus2),
            _get_proposition(ncol, nstatus1, col2, status2),
            _get_proposition(ncol, nstatus1, col2, nstatus2),
            _get_proposition(ncol, nstatus1, ncol, status2),
            _get_proposition(ncol, nstatus1, ncol, nstatus2),
        ], dtype=jnp.int32)

        # Set the value of the propositions to 1
        return self._get_next_to_base_label().at[propositions].set(1)

    def _get_next_to_label(self, timestep: Timestep) -> jax.Array:
        """
        Returns the label associated with `next to` propositions.
        """
        if not self.use_next_to_props:
            return jnp.array([])

        observation = self._get_observation(timestep)

        # Get the agent position in the observation
        if self._use_ego_obs:
            agent_pos = jnp.array([observation.shape[0] - 1, observation.shape[1] // 2])
        else:
            agent_pos = timestep.state.agent.position

        def _get_next_to_label_pos_pair(p1: jax.Array, p2: jax.Array) -> jax.Array:
            """
            Returns the label associated with a pair of adjacent positions if the
            positions are valid and the objects there are labelable.
            """
            is_valid_p2 = (
                jnp.all(p2 < jnp.array([observation.shape[0], observation.shape[1]]))
                & ~jnp.array_equal(agent_pos, p1)
                & ~jnp.array_equal(agent_pos, p2)
            )

            xminigrid_obj1, xminigrid_col1 = observation[p1[0], p1[1]]
            xminigrid_obj2, xminigrid_col2 = observation[p2[0], p2[1]]

            return jax.lax.cond(
                pred=(
                    is_valid_p2
                    & self._is_labelable_object(xminigrid_obj1)
                    & self._is_labelable_object(xminigrid_obj2)
                ),
                true_fun=lambda: self._get_next_to_label_obj_pair(
                    xminigrid_obj1, xminigrid_col1, xminigrid_obj2, xminigrid_col2
                ),
                false_fun=self._get_next_to_base_label,
            )

        @functools.partial(jax.vmap, in_axes=(0, None))
        @functools.partial(jax.vmap, in_axes=(None, 0))
        def _get_next_to_label_from(row: jnp.int_, col: jnp.int_) -> jax.Array:
            """
            Obtains the label for the positions to the right and below, and puts
            them together by stacking them and taking the maximum value for each
            of the propositions; e.g., if proposition `i` appears positively (as 1)
            at least once, we want its value to be 1 when put together.
            """
            pos = jnp.array([row, col])
            label_right = _get_next_to_label_pos_pair(pos, pos + jnp.array([1, 0]))
            label_down = _get_next_to_label_pos_pair(pos, pos + jnp.array([0, 1]))
            return jnp.stack((label_right, label_down)).max(axis=0)

        # Obtain the label from each possible position in the grid
        labels = _get_next_to_label_from(
            jnp.arange(0, observation.shape[0]), jnp.arange(0, observation.shape[1])
        )

        # Take the maximum value for each proposition across the different sublabels
        return jnp.max(
            labels.reshape(
                (observation.shape[0] * observation.shape[1], -1),
            ),
            axis=0,
        )

    def prop_to_str(self, prop_id: int) -> str:
        if self.is_front_prop(prop_id):
            return self._get_front_prop_str(prop_id)
        elif self.is_carrying_prop(prop_id):
            return self._get_carrying_prop_str(prop_id)
        elif self.is_next_to_prop(prop_id):
            return self._get_next_to_prop_str(prop_id)

    def get_num_loc_types(self) -> int:
        """
        Returns the number of location types used in the propositions.
        """
        return self.use_front_props + self.use_carrying_props + self.use_next_to_props

    def get_num_obj_types(self) -> int:
        """
        Returns the number of object types used in the propositions.
        """
        return self._num_labelable_objs

    def get_num_color_types(self) -> int:
        """
        Returns the number of color types used in the propositions.
        """
        return len(self._labelable_color_types)

    def get_num_status_types(self) -> int:
        """
        Returns the number of status types used in the propositions.
        In this case, it depends on that of doors.
        """
        return len(self._labelable_door_obj_types)

    def is_non_empty_color(self, color_id: int) -> bool:
        """
        Returns True if `color_id` corresponds to an actual XMinigrid color
        (i.e., not the special empty case created for the propositions ), and
        False otherwise.
        """
        return color_id < len(self._labelable_color_types)

    def is_valid_non_empty_status(self, obj_id: int, status_id: int) -> bool:
        """
        Returns True if the `status_id` is a possible status for `obj_id` and
        corresponds to an actual XMinigrid status (i.e., not the special empty
        case created for the propositions), and False otherwise.
        """
        return status_id < self._num_status_per_obj[obj_id]

    def is_front_prop(self, prop_id: int) -> jnp.bool_:
        return jnp.logical_and(jnp.less_equal(0, prop_id), jnp.less(prop_id, self._num_front_props))

    def _get_front_prop_str(self, prop_id: int) -> str:
        """
        Returns the string representation of a `front` proposition.
        """
        return f"front_{self._get_obj_prop_str(*self.get_front_obj_properties(prop_id))}"

    def get_front_obj_properties(self, prop_id: int) -> Tuple[int, int, int]:
        return self._get_obj_properties(prop_id, jnp.ones((self._num_labelable_objs,), dtype=jnp.bool_))

    def is_carrying_prop(self, prop_id: int) -> jnp.bool_:
        return jnp.logical_and(
            jnp.less_equal(self._num_front_props, prop_id),
            jnp.less(prop_id, self._num_front_props + self._num_carrying_props)
        )

    def _get_carrying_prop_str(self, prop_id: int) -> str:
        """
        Returns the string representation of a `carrying` proposition.
        """
        return f"carrying_{self._get_obj_prop_str(*self.get_carrying_obj_properties(prop_id))}"

    def get_carrying_obj_properties(self, prop_id: int) -> Tuple[int, int, int]:
        return self._get_obj_properties(prop_id - self._num_front_props, self._pickable_mask)

    def get_next_obj_properties(self, prop_id: int) -> Tuple[int, int, int, int, int, int]:
        return self._next_to_prop_to_obj[
            prop_id - self._num_front_props - self._num_carrying_props
        ]

    def _get_obj_properties(self, prop_id: int, obj_mask: jax.Array) -> Tuple[int, int, int]:
        obj_end_idxs = jnp.cumsum(
            (self._num_status_per_obj + 1) * self._num_colors * obj_mask
        )
        obj_start_idxs = jnp.roll(obj_end_idxs, shift=1).at[0].set(0)

        obj_id = jnp.argmax(prop_id < obj_end_idxs)
        color_id = (prop_id - obj_start_idxs[obj_id]) % self._num_colors
        status_id = (prop_id - obj_start_idxs[obj_id]) // self._num_colors

        return obj_id, color_id, status_id

    def _get_obj_prop_str(self, obj_id: int, color_id: int, status_id: int) -> str:
        """
        Returns the string representation of a proposition involving a single
        object (`front` and `carrying`).
        """
        color_str = self._get_color_str(color_id)
        color_str = f"_{color_str}" if len(color_str) > 0 else color_str

        status_str = self._get_status_str(obj_id, status_id)
        status_str = f"_{status_str}" if len(status_str) > 0 else status_str

        return f"{self._get_obj_str(obj_id)}{color_str}{status_str}"

    def is_next_to_prop(self, prop_id: int) -> jnp.bool_:
        return jnp.greater_equal(prop_id, self._num_front_props + self._num_carrying_props)

    def _get_next_to_prop_str(self, prop_id: int) -> str:
        """
        Returns the string representation of a `next_to` proposition.
        """
        prop_id = prop_id - self._num_front_props - self._num_carrying_props
        obj1, col1, status1, obj2, col2, status2 = self._next_to_prop_to_obj[
            int(prop_id)
        ]

        col1_str = self._get_color_str(col1)
        col1_str = f"_{col1_str}" if len(col1_str) > 0 else col1_str

        col2_str = self._get_color_str(col2)
        col2_str = f"_{col2_str}" if len(col2_str) > 0 else col2_str

        status1_str = self._get_status_str(obj1, status1)
        status1_str = f"_{status1_str}" if len(status1_str) > 0 else status1_str

        status2_str = self._get_status_str(obj2, status2)
        status2_str = f"_{status2_str}" if len(status2_str) > 0 else status2_str

        return (
            "next_"
            f"{self._get_obj_str(obj1)}{col1_str}{status1_str}"
            f"_{self._get_obj_str(obj2)}{col2_str}{status2_str}"
        )

    def _get_xminigrid_obj(self, obj_id: int) -> Tiles:
        """
        Returns the XLand-Minigrid object id from our local object ids.
        """
        return jnp.nonzero(obj_id == self._object_ids)[0][0]

    def _get_obj_str(self, obj_id: jnp.int_) -> str:
        """
        Returns the string representation of a given color.
        """
        return xminigrid_obj_to_str(int(self._get_xminigrid_obj(obj_id)))

    def _get_color_str(self, color_id: jnp.int_) -> str:
        """
        Returns the string representation of a given color.
        """
        if color_id < self._num_colors - 1:
            return xminigrid_color_to_str(int(self._labelable_color_types[color_id]))
        return ""

    def _get_status_str(self, obj_id: jnp.int_, status_id: jnp.int_) -> str:
        """
        Returns the string representation of the status of a given object.
        """
        if status_id < self._num_status_per_obj[obj_id]:
            xminigrid_obj = self._get_xminigrid_obj(obj_id)
            if xminigrid_obj in self._labelable_door_obj_types:
                return xminigrid_status_to_str(
                    int(self._labelable_door_obj_types[status_id])
                )
        return ""

    def sample_xminigrid_objs_from_prop(self, prop_id: int, rng: chex.PRNGKey) -> chex.Array:
        """
        Returns the XMinigrid objects (tile, color) involved in a proposition.

        To make the method jittable, the size of the returned tuple is 4: a tile-color
        pair for each possible involved objects (a maximum of two from the `next`
        propositions).

        Since some propositions are not fully specified (i.e., color or status may
        be missing), we randomly generate an object instance for them.
        """
        obj1_rng, col1_rng, obj2_rng, col2_rng = jax.random.split(rng, 4)

        def _get_xminigrid_obj(obj_id: int, status_id: int, obj_rng: chex.PRNGKey):
            max_num_status = self.get_num_status_types()
            if max_num_status > 0:  # if there are doors
                return jax.lax.cond(
                    pred=self._num_status_per_obj[obj_id] > 0,  # if the object is a door
                    true_fun=lambda: jax.lax.cond(
                        pred=status_id >= max_num_status,  # if the status is not specified, randomly sample one
                        true_fun=lambda: self._labelable_door_obj_types[jax.random.choice(obj_rng, max_num_status)],
                        false_fun=lambda: self._labelable_door_obj_types[status_id],
                    ),
                    false_fun=lambda: self._labelable_non_door_obj_types[obj_id]
                )
            return self._labelable_non_door_obj_types[obj_id]

        def _get_xminigrid_col(color_id: int, col_rng: chex.PRNGKey):
            return jax.lax.cond(
                pred=color_id >= self.get_num_color_types(),  # if the color is not specified, randomly sample one
                true_fun=lambda: jax.random.choice(col_rng, self._labelable_color_types),
                false_fun=lambda: self._labelable_color_types[color_id],
            )

        def _get_xminigrid_front():
            obj_id, color_id, status_id = self.get_front_obj_properties(prop_id)
            return jnp.array([
                _get_xminigrid_obj(obj_id, status_id, obj1_rng),
                _get_xminigrid_col(color_id, col1_rng),
                -1,
                -1
            ])

        def _get_xminigrid_carrying():
            obj_id, color_id, status_id = self.get_carrying_obj_properties(prop_id)
            return jnp.array([
                _get_xminigrid_obj(obj_id, status_id, obj1_rng),
                _get_xminigrid_col(color_id, col1_rng),
                -1,
                -1
            ])

        def _get_xminigrid_next():
            obj1_id, color1_id, status1_id, obj2_id, color2_id, status2_id = self.get_next_obj_properties(prop_id)
            return jnp.array([
                _get_xminigrid_obj(obj1_id, status1_id, obj1_rng),
                _get_xminigrid_col(color1_id, col1_rng),
                _get_xminigrid_obj(obj2_id, status2_id, obj2_rng),
                _get_xminigrid_col(color2_id, col2_rng)
            ])

        branches = [lambda: -jnp.ones(shape=(4,), dtype=int)]  # an invalid object, return all -1s
        if self.use_front_props:
            branches.append(_get_xminigrid_front)
        if self.use_carrying_props:
            branches.append(_get_xminigrid_carrying)
        if self.use_next_to_props:
            branches.append(_get_xminigrid_next)

        return jax.lax.switch(
            index=self.is_front_prop(prop_id) + 2 * self.is_carrying_prop(prop_id) + 3 * self.is_next_to_prop(prop_id),
            branches=branches
        )

    def get_possible_xminigrid_objs(
        self,
        obj_id: jnp.int_,
        color_id: jnp.int_,
        status_id: jnp.int_,
        level: XMinigridLevel
    ):
        """
        Returns the possible XLand-Minigrid's <tile, color> combinations for a
        local <obj_id, color_id, status_id> triplet in a given level.

        The several combinations come from the fact that the color and the status
        may not be specified.

        In the case of doors, their status can change during an episode (locked,
        closed, open). If the passed status is `locked`, we check the level to
        verify the door can be unlocked with a matching key, and thus add `open`
        and `closed` as other feasible status. If the passed status is `open` or
        `closed`, then it cannot be locked.

        Example: If the object corresponds to `box` and the color is not specified,
        the possible combinations are those for any of the colors.

        Warning: This method is non-jittable.
        """
        return list(product(
            self._get_possible_xminigrid_tiles(obj_id, color_id, status_id, level),
            self._get_possible_xminigrid_colors(color_id)
        ))

    def _get_possible_xminigrid_tiles(
        self,
        obj_id: jnp.int_,
        color_id: jnp.int_,
        status_id: jnp.int_,
        level: XMinigridLevel
    ) -> List[Tiles]:
        num_status = self._num_status_per_obj[obj_id]

        # If the object is not a door, there are not multiple possibilities
        if num_status == 0:
            return [self._labelable_non_door_obj_types[obj_id]]

        # If the status is not specified, then any door type is valid
        if not self.is_valid_non_empty_status(obj_id, status_id):
            return [
                self._labelable_door_obj_types[status_id]
                for status_id in range(num_status)
            ]

        # If the status of the door is a specific one
        xminigrid_obj = self._labelable_door_obj_types[status_id]

        # If the door is not locked, then any other status is possible (open, closed)
        if xminigrid_obj != Tiles.DOOR_LOCKED:
            return [
                self._labelable_door_obj_types[status_id]
                for status_id in range(num_status)
                if self._labelable_door_obj_types[status_id] != Tiles.DOOR_LOCKED
            ]

        # If the status of the door is locked...
        if self.is_non_empty_color(color_id):
            # If there is a specific color, check whether there is a key
            # of the same color that can unlock it
            unlockable = level.contains(jnp.array([
                Tiles.KEY, self._labelable_color_types[color_id]
            ]))
        else:
            # If it can be of any color, check if there is a locked door
            # in the level that can be open with a key of a matching color
            locked_door_colors = set((level.get_colors(Tiles.DOOR_LOCKED)).tolist())
            key_colors = set((level.get_colors(Tiles.KEY)).tolist())
            unlockable = len(locked_door_colors.intersection(key_colors)) > 0

        if unlockable:
            # If the door can be unlocked, any available status is possible
            return [
                self._labelable_door_obj_types[i]
                for i in range(num_status)
            ]

        # The door can only remain locked
        return [xminigrid_obj]

    def _get_possible_xminigrid_colors(self, color_id: jnp.int_) -> List[Colors]:
        if self.is_non_empty_color(color_id):
            color_ids = [color_id]
        else:  # check any of the values
            color_ids = [i for i in range(self.get_num_color_types())]
        return [self._labelable_color_types[color] for color in color_ids]

    def get_prop_mask_from_objs(self, xminigrid_objs: chex.Array) -> chex.Array:
        """
        Returns a binary mask indicating the propositions associated with the
        passed list of objects.

        In the case of doors, the propositions associated to other states than
        the one being passed might be activated:
            - Closed and open doors can become open or closed respectively.
            - Locked doors can be also closed or open if there is a key with a
              matching color, or not if there is not such a key.

        Assumptions:
        - Two doors cannot be next to each other.
        """
        def _get_front_mask(xminigrid_tile: chex.Array, xminigrid_col: chex.Array) -> chex.Array:
            obj_label = jax.lax.cond(
                pred=self._is_labelable_object(xminigrid_tile),
                true_fun=lambda: self._get_object_label(
                    xminigrid_tile,
                    xminigrid_col,
                    self._get_front_base_label(),
                    jnp.ones((self._num_labelable_objs,), dtype=jnp.bool),
                ),
                false_fun=lambda: self._get_front_base_label(),
            )
            return obj_label == 1

        def _get_carrying_mask(xminigrid_tile: chex.Array, xminigrid_col: chex.Array) -> chex.Array:
            obj_label = jax.lax.cond(
                pred=self._is_carriable_object(xminigrid_tile),
                true_fun=lambda: self._get_object_label(
                    xminigrid_tile,
                    xminigrid_col,
                    self._get_carrying_base_label(),
                    self._pickable_mask,
                ),
                false_fun=lambda: self._get_carrying_base_label(),
            )
            return obj_label == 1

        def _get_next_to_mask(idx: int, xminigrid_tile: chex.Array, xminigrid_col: chex.Array) -> chex.Array:
            def _get_next_to_mask_aux(carry, x: Tuple[int, chex.Array]):
                idx_2, xminigrid_obj_2 = x
                xminigrid_tile_2, xminigrid_col_2 = xminigrid_obj_2[0], xminigrid_obj_2[1]
                label = jax.lax.cond(
                    pred=(
                        (idx != idx_2) &
                        self._is_labelable_object(xminigrid_tile) &
                        self._is_labelable_object(xminigrid_tile_2) &
                        jnp.logical_not(is_xminigrid_door(xminigrid_tile) & is_xminigrid_door(xminigrid_tile_2))
                    ),
                    true_fun=lambda: self._get_next_to_label_obj_pair(
                        xminigrid_tile, xminigrid_col, xminigrid_tile_2, xminigrid_col_2
                    ),
                    false_fun=lambda: self._get_next_to_base_label(),
                )
                return jnp.logical_or(carry, label == 1), None

            end_carry, _ = jax.lax.scan(
                _get_next_to_mask_aux,
                init=jnp.zeros((self._num_next_to_props,), dtype=jnp.bool),
                xs=(jnp.arange(len(xminigrid_objs)), xminigrid_objs)
            )

            return end_carry

        def _get_mask_tile_col(idx: int, xminigrid_tile: chex.Array, xminigrid_col: chex.Array):
            empty_mask = jnp.array([], dtype=jnp.bool)
            return jnp.concat((
                _get_front_mask(xminigrid_tile, xminigrid_col) if self.use_front_props else empty_mask,
                _get_carrying_mask(xminigrid_tile, xminigrid_col) if self.use_carrying_props else empty_mask,
                _get_next_to_mask(idx, xminigrid_tile, xminigrid_col) if self.use_next_to_props else empty_mask
            ))

        def _get_mask_door(idx: int, xminigrid_tile: chex.Array, xminigrid_col: chex.Array):
            # Get the proposition masks for the three possible door types
            masks = jnp.array([
                _get_mask_tile_col(idx, Tiles.DOOR_OPEN, xminigrid_col),
                _get_mask_tile_col(idx, Tiles.DOOR_CLOSED, xminigrid_col),
                _get_mask_tile_col(idx, Tiles.DOOR_LOCKED, xminigrid_col),
            ])

            # Determine whether each of the masks is applicable
            #  - The propositions for open and closed door appear if the doors is unlockable
            #    (it is originally closed, open or locked but there is matching key).
            #  - The propositions for locked always appear if the door is originally locked.
            exists_matching_key = jnp.any(jnp.all(xminigrid_objs == jnp.array([Tiles.KEY, xminigrid_col]), axis=1))
            is_unlockable = (
                ((xminigrid_tile == Tiles.DOOR_LOCKED) & exists_matching_key) |
                (xminigrid_tile == Tiles.DOOR_CLOSED) |
                (xminigrid_tile == Tiles.DOOR_OPEN)
            )
            applicability_mask = jnp.array([is_unlockable, is_unlockable, xminigrid_tile == Tiles.DOOR_LOCKED])

            # Select the applicable masks and merge them
            return jnp.any(masks * applicability_mask[:, jnp.newaxis], axis=0)

        def _get_mask_obj(idx: int, xminigrid_obj: chex.Array):
            tile, color = xminigrid_obj[0], xminigrid_obj[1]
            return jax.lax.cond(
                pred=is_xminigrid_door(tile),
                true_fun=lambda: _get_mask_door(idx, tile, color),
                false_fun=lambda: _get_mask_tile_col(idx, tile, color)
            )

        _, masks = jax.lax.scan(
            lambda _, xs: (None, _get_mask_obj(*xs)),
            None,
            (jnp.arange(len(xminigrid_objs)), xminigrid_objs)
        )

        return jnp.any(masks, axis=0)

    def _get_observation(self, timestep: Timestep):
        if self._use_ego_obs:
            return timestep.observation["ego"]
        
        # Take the observation (the :2 is to exclude the 'agent direction' matrix used in the `full_3d` mode)
        return timestep.observation["full"][:, :, :2]
