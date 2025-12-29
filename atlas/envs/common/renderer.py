from abc import ABC, abstractmethod

from PIL import Image

from .env import EnvParams
from .level import Level


class EnvironmentRenderer(ABC):
    @abstractmethod
    def render(self, state, env_params: EnvParams) -> Image:
        raise NotImplementedError

    @abstractmethod
    def render_level(self, level: Level) -> Image:
        raise NotImplementedError
