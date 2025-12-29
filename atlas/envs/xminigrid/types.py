from typing import Tuple

from flax import struct
from xminigrid.core.constants import Tiles, Colors
from xminigrid.environment import EnvParams as _XMinigridEnvParams

from ..common.types import EnvParams


class XMinigridEnvParams(_XMinigridEnvParams, EnvParams):
    """
    Parameters characterizing XLand-Minigrid environments. Some of the parameters
    are inherited from the original XLand-Minigrid class, and some from our high
    level abstraction.

    Args:
        use_ego_obs: whether to include the egocentric observation in the observation
                     dictionary.
        use_full_obs: whether to include the full grid observation in the observation
                      dictionary.
        non_door_obj_types: the objects different from doors with which the agent
                            interacts in the environment.
        door_obj_types: the door objects with which the agent interacts in the
                        environment.
        color_types: the possible colors of the objects above in the environment.
    """

    use_ego_obs: bool = struct.field(pytree_node=False, default=True)
    use_full_obs: bool = struct.field(pytree_node=False, default=False)

    non_door_obj_types: Tuple = struct.field(
        pytree_node=False, default=(Tiles.BALL, Tiles.SQUARE, Tiles.KEY),
    )
    door_obj_types: Tuple = struct.field(
        pytree_node=False,
        default=(Tiles.DOOR_CLOSED, Tiles.DOOR_LOCKED, Tiles.DOOR_OPEN),
    )
    color_types: Tuple = struct.field(
        pytree_node=False,
        default=(
            Colors.RED,
            Colors.GREEN,
            Colors.BLUE,
            Colors.PURPLE,
            Colors.YELLOW,
            Colors.GREY,
        ),
    )
