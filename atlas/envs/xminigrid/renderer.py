from PIL import Image
from xminigrid.rendering.rgb_render import render
from xminigrid.types import AgentState, State

from .level import XMinigridLevel
from .types import XMinigridEnvParams
from ..common.renderer import EnvironmentRenderer


class XMinigridRenderer(EnvironmentRenderer):
    MINIGRID_TILE_SIZE = 64  # increased from 16 to 64 for higher resolution

    def render(self, state: State, env_params: XMinigridEnvParams) -> Image:
        return Image.fromarray(render(
            state.grid, state.agent, env_params.view_size, self.MINIGRID_TILE_SIZE
        ))

    def render_level(self, level: XMinigridLevel) -> Image:
        return Image.fromarray(render(
            level.grid[:level.height, :level.width, :],
            AgentState(position=level.agent_pos, direction=level.agent_dir),
            view_size=5,
            tile_size=self.MINIGRID_TILE_SIZE
        ))
