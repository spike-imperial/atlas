from abc import ABC, abstractmethod
from typing import Optional, Dict

import chex

from .level import Level
from .types import EnvParams


class LevelSampler(ABC):
    """
    Abstraction of an object for randomly sampling levels/instances.
    """

    def __init__(self, env_params: EnvParams):
        self._env_params = env_params

    def __call__(self, key: chex.PRNGKey, extras: Optional[Dict] = None) -> Level:
        return self.sample(key, extras)

    @abstractmethod
    def sample(self, key: chex.PRNGKey, extras: Optional[Dict] = None) -> Level:
        raise NotImplementedError

    def unwrapped(self) -> "LevelSampler":
        return self
