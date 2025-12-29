import chex
import jax.numpy as jnp
from xminigrid.core.constants import NUM_TILES, Tiles, Colors
from xminigrid.core.observation import transparent_field_of_view
from xminigrid.types import State

from .types import XMinigridEnvParams


def egocentric(env_params: XMinigridEnvParams, state: State) -> chex.Array:
    """
    Returns an egocentric view of the environment of shape [view_size, view_size, 2],
    where the first coordinate corresponds to the object id and the second corresponds
    to the color. The position of the agent contains a standard floor tile or the
    object being carried.
    """
    # Compute egocentric view
    observation = transparent_field_of_view(
        state.grid, state.agent, env_params.view_size, env_params.view_size
    )

    # Put content being carried on the agent's position
    agent_pos = (env_params.view_size - 1, env_params.view_size // 2)
    return observation.at[agent_pos].set(_get_agent_pos_obj(state))


def full_2d(state: State) -> chex.Array:
    """
    Returns the full view of the grid of shape [height, width, 2],
    where the first coordinate corresponds to the object id and the second
    corresponds to the color. The position of the agent contains a special
    id for the agent (represented as the number of tiles, which is an unused
    number) and the color is its direction. The object being picked is not
    represented, assuming that an RNN or similar will keep track of it,
    """
    return state.grid.at[tuple(state.agent.position)].set(
        jnp.array([NUM_TILES, state.agent.direction], dtype=jnp.uint8)
    )


def full_3d(env_params: XMinigridEnvParams, state: State) -> chex.Array:
    """
    Returns the full view of the grid of shape [height, width, 3],
    where the first coordinate corresponds to the object id, the second
    corresponds to the color, and the third is used to place the agent's
    direction. The first two dimensions represent a standard floor tile
    for the agent unless it is carrying something. Unlike the `full_2d`
    observation, an RNN shouldn't be needed.
    """
    agent_pos = tuple(state.agent.position)

    # Put object being carried in the agent position (or just a standard
    # floor tile if not object is carried)
    grid = state.grid.at[agent_pos].set(_get_agent_pos_obj(state))

    # Create matrix specifying the direction of the object
    agent_dir = jnp.zeros(
        (env_params.height, env_params.width),
        dtype=jnp.uint8
    ).at[agent_pos].set(jnp.asarray(state.agent.direction, dtype=jnp.uint8))

    # Concatenate the full grid view with the agent direction matrix
    return jnp.concat((grid, agent_dir[..., jnp.newaxis]), axis=2)


def _get_agent_pos_obj(state: State) -> chex.Array:
    """
    Returns the content to put in the agent position: a standard floor tile
    if carrying nothing or the object being carried otherwise.
    """
    pocket = state.agent.pocket
    mask = pocket != 0
    return (
        mask * pocket +
        jnp.logical_not(mask) * jnp.array([Tiles.FLOOR, Colors.BLACK], dtype=jnp.uint8)
    )
