from abc import ABC, abstractmethod
from typing import Tuple

import chex

from ..envs.common.level import Level
from ..hrm.types import HRM


class ProblemSampler(ABC):
    """
    Samples a <level, HRM> pair.
    """
    @abstractmethod
    def sample(self, rng: chex.PRNGKey) -> Tuple[Level, HRM]:
        raise NotImplementedError

    def __call__(self, key: chex.PRNGKey) -> Tuple[Level, HRM]:
        return self.sample(key)
