from omegaconf import OmegaConf

from ..envs.xminigrid.utils import (
    xminigrid_color_str_to_id,
    xminigrid_obj_str_to_id,
    xminigrid_status_str_to_id,
)


def register_resolvers():
    """Register custom OmegaConf resolvers."""
    # Resolver to determine the number of minibatches
    OmegaConf.register_new_resolver(
        "get_num_minibatches",
        lambda num_envs: num_envs // 32,
    )

    # Resolver to transform string representations of objects, colors and
    # status in XMinigrid into their actual integer representations
    OmegaConf.register_new_resolver(
        "xminigrid_obj",
        lambda x: list(map(xminigrid_obj_str_to_id, x)),
    )
    OmegaConf.register_new_resolver(
        "xminigrid_col",
        lambda x: list(map(xminigrid_color_str_to_id, x)),
    )
    OmegaConf.register_new_resolver(
        "xminigrid_status",
        lambda x: list(map(xminigrid_status_str_to_id, x)),
    )
