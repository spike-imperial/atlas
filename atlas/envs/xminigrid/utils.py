from itertools import product
import os

if os.environ.get("DISABLE_JAX_IMPORT"):
    # Added this to avoid importing JAX at the resolving stage
    # when using the `run_sweep` script
    TILE_IDS = {
        "ball": 3,
        "door_closed": 9,
        "door_locked": 8,
        "door_open": 10,
        "hex": 11,
        "key": 7,
        "pyramid": 5,
        "square": 4,
        "star": 12,
    }

    COLOR_IDS = {
        "black": 7,
        "blue": 3,
        "brown": 10,
        "green": 2,
        "grey": 6,
        "purple": 4,
        "red": 1,
        "orange": 8,
        "pink": 11,
        "white": 9,
        "yellow": 5,
    }

    STATUS_IDS = {
        "door_locked": 8,
        "door_closed": 9,
        "door_open": 10,
    }
else:
    import chex
    import jax
    import jax.numpy as jnp
    from xminigrid.core.constants import Colors, Tiles
    from xminigrid.types import GridState

    from .types import XMinigridEnvParams

    TILE_IDS = {
        "ball": Tiles.BALL,
        "door_closed": Tiles.DOOR_CLOSED,
        "door_locked": Tiles.DOOR_LOCKED,
        "door_open": Tiles.DOOR_OPEN,
        "hex": Tiles.HEX,
        "key": Tiles.KEY,
        "pyramid": Tiles.PYRAMID,
        "square": Tiles.SQUARE,
        "star": Tiles.STAR,
    }

    COLOR_IDS = {
        "black": Colors.BLACK,
        "blue": Colors.BLUE,
        "brown": Colors.BROWN,
        "green": Colors.GREEN,
        "grey": Colors.GREY,
        "purple": Colors.PURPLE,
        "red": Colors.RED,
        "orange": Colors.ORANGE,
        "pink": Colors.PINK,
        "white": Colors.WHITE,
        "yellow": Colors.YELLOW,
    }

    STATUS_IDS = {
        "door_locked": Tiles.DOOR_LOCKED,
        "door_closed": Tiles.DOOR_CLOSED,
        "door_open": Tiles.DOOR_OPEN,
    }

    def pad_grid(grid: chex.Array, max_height: int, max_width: int) -> chex.Array:
        """
        Pads the grid with empty tiles to reach the size imposed by
        the environment parameters. The empty tiles are chosen because
        it is what the agent sees beyond the limits of the grid (i.e.
        for consistency). The grid is of shape (height, width, 2).
        """
        h_pad = max_height - grid.shape[0]
        w_pad = max_width - grid.shape[1]

        return jnp.stack(
            arrays=(
                jnp.pad(grid[:, :, 0], pad_width=((0, h_pad), (0, w_pad)), constant_values=Tiles.EMPTY),
                jnp.pad(grid[:, :, 1], pad_width=((0, h_pad), (0, w_pad)), constant_values=Colors.EMPTY),
            ),
            axis=-1
        )

    def sample_coordinates(key: jax.Array, grid: GridState, num: int, mask: jax.Array | None = None) -> jax.Array:
        if mask is None:
            mask = jnp.ones((grid.shape[0], grid.shape[1]), dtype=jnp.bool_)

        coords = jax.random.choice(
            key=key,
            shape=(num,),
            a=jnp.arange(grid.shape[0] * grid.shape[1]),
            replace=False,
            p=mask.flatten(),
        )
        coords = jnp.divmod(coords, grid.shape[1])
        coords = jnp.concatenate((coords[0].reshape(-1, 1), coords[1].reshape(-1, 1)), axis=-1)
        return coords

    def sample_nondoor_obj_type(rng: chex.PRNGKey, env_params: XMinigridEnvParams, n: int = 1) -> chex.Array:
        return jax.random.choice(rng, jnp.array(env_params.non_door_obj_types), (n,))

    def sample_door_obj_type(rng: chex.PRNGKey, env_params: XMinigridEnvParams, n: int = 1) -> chex.Array:
        return jax.random.choice(rng, jnp.array(env_params.door_obj_types), (n,))

    def sample_color_type(rng: chex.PRNGKey, env_params: XMinigridEnvParams, n: int = 1) -> chex.Array:
        return jax.random.choice(rng, jnp.array(env_params.color_types), (n,))

    def get_nondoor_obj_mask(grid: GridState, env_params: XMinigridEnvParams) -> chex.Array:
        return jnp.isin(grid[..., 0], jnp.array(env_params.non_door_obj_types))

    def get_door_obj_mask(grid: GridState, env_params: XMinigridEnvParams) -> chex.Array:
        return jnp.isin(grid[..., 0], jnp.array(env_params.door_obj_types))

    def get_obj_mask(grid: GridState, env_params: XMinigridEnvParams) -> chex.Array:
        return jnp.logical_or(get_nondoor_obj_mask(grid, env_params), get_door_obj_mask(grid, env_params))

    def insert_obj(grid: GridState, obj: chex.Array, coord: chex.Array) -> GridState:
        return grid.at[coord[0], coord[1]].set(obj)

    def clear_coord(grid: GridState, coord: chex.Array) -> GridState:
        return grid.at[coord[0], coord[1]].set(jnp.array([Tiles.FLOOR, Colors.BLACK], dtype=jnp.uint8))


_OBJ_TO_STR = {
    TILE_IDS["ball"]: "ball",
    TILE_IDS["door_closed"]: "door",
    TILE_IDS["door_locked"]: "door",
    TILE_IDS["door_open"]: "door",
    TILE_IDS["hex"]: "hex",
    TILE_IDS["key"]: "key",
    TILE_IDS["pyramid"]: "pyramid",
    TILE_IDS["square"]: "square",
    TILE_IDS["star"]: "star",
}
_STR_TO_OBJ = {v: k for k, v in _OBJ_TO_STR.items() if k != "door"}

_COLOR_TO_STR = {
    COLOR_IDS["black"]: "black",
    COLOR_IDS["blue"]: "blue",
    COLOR_IDS["brown"]: "brown",
    COLOR_IDS["green"]: "green",
    COLOR_IDS["grey"]: "grey",
    COLOR_IDS["purple"]: "purple",
    COLOR_IDS["red"]: "red",
    COLOR_IDS["orange"]: "orange",
    COLOR_IDS["pink"]: "pink",
    COLOR_IDS["white"]: "white",
    COLOR_IDS["yellow"]: "yellow",
}
_STR_TO_COLOR = {v: k for k, v in _COLOR_TO_STR.items()}

_STATUS_TO_STR = {
    STATUS_IDS["door_locked"]: "locked",
    STATUS_IDS["door_closed"]: "closed",
    STATUS_IDS["door_open"]: "open",
}
_STR_TO_STATUS = {v: k for k, v in _STATUS_TO_STR.items()}


def xminigrid_obj_to_str(xminigrid_obj: int) -> str:
    return _OBJ_TO_STR[xminigrid_obj]


def xminigrid_obj_str_to_id(xminigrid_str: str) -> int:
    return _STR_TO_OBJ[xminigrid_str]


def xminigrid_color_to_str(xminigrid_color: int) -> str:
    return _COLOR_TO_STR[xminigrid_color]


def xminigrid_color_str_to_id(xminigrid_str: str) -> int:
    return _STR_TO_COLOR[xminigrid_str]


def xminigrid_status_to_str(xminigrid_status: int) -> str:
    return _STATUS_TO_STR[xminigrid_status]


def xminigrid_status_str_to_id(xminigrid_str: str) -> int:
    return _STR_TO_STATUS[xminigrid_str]


def is_xminigrid_door(xminigrid_obj: int) -> bool:
    return (
        (xminigrid_obj == TILE_IDS["door_closed"]) | 
        (xminigrid_obj == TILE_IDS["door_locked"]) | 
        (xminigrid_obj == TILE_IDS["door_open"])
    )


def is_observable_prop(prop_id: int, level, label_fn):
    """
    Returns True if the proposition is deemed observable by an agent interacting
    in the given level, and False otherwise.

    The generality of the propositions is taken into account, e.g. if the
    proposition is not associated with a specific color, we check whether
    the object appears with any of the possible labelable colors in the grid.
    See the method `get_possible_xminigrid_objs` implemented in the
    XMinigridLabelingFunction for more information.

    In the case of doors, we *assume* there will never be one door next to
    another, which is the case in our level samplers. Therefore, any `next`
    proposition involving two doors will always be set as not observable.
    The assumption is already made by the labeling function itself.
    """
    if label_fn.is_front_prop(prop_id):
        return _is_observable_single_obj(
            *label_fn.get_front_obj_properties(prop_id), level, label_fn
        )
    elif label_fn.is_carrying_prop(prop_id):
        return _is_observable_single_obj(
            *label_fn.get_carrying_obj_properties(prop_id), level, label_fn
        )
    elif label_fn.is_next_to_prop(prop_id):
        o1_id, c1_id, s1_id, o2_id, c2_id, s2_id = label_fn.get_next_obj_properties(prop_id)

        for xobj1, xobj2 in product(
            label_fn.get_possible_xminigrid_objs(o1_id, c1_id, s1_id, level),
            label_fn.get_possible_xminigrid_objs(o2_id, c2_id, s2_id, level)
        ):
            # *Assume* that two doors will never be next to each other
            # It should never be triggered since there is no proposition `next`
            # where both items are doors.
            if is_xminigrid_door(xobj1[0]) and is_xminigrid_door(xobj2[0]):
                continue

            if xobj1 == xobj2:
                # If the object is the same, check there are at least of them
                # for the proposition to appear
                exist = level.num_of(jnp.asarray(xobj1)) >= 2
            else:
                # Otherwise, check whether both exist
                exist = jnp.logical_and(
                    level.contains(jnp.asarray(xobj1)),
                    level.contains(jnp.asarray(xobj2))
                )

            if exist:
                return True

    return False


def _is_observable_single_obj(obj_id: int, color_id: int, status_id: int, level, label_fn):
    """
    Helper method for the method above (see its docstring).
    """
    for xminigrid_obj in label_fn.get_possible_xminigrid_objs(
        obj_id, color_id, status_id, level
    ):
        if level.contains(jnp.asarray(xminigrid_obj)):
            return True

    return False
