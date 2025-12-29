from abc import ABC, abstractmethod
from typing import Dict, Optional

import chex

from .. import ops
from ..types import HRM


class HRMSampler(ABC):
    def __init__(
        self,
        max_num_rms: int,
        max_num_states: int,
        max_num_edges: int,
        max_num_literals: int,
        alphabet_size: int,
    ):
        self._max_num_rms = max_num_rms
        self._max_num_states = max_num_states
        self._max_num_edges = max_num_edges
        self._max_num_literals = max_num_literals
        self._alphabet_size = alphabet_size

    def __call__(self, key: chex.PRNGKey, extras: Optional[Dict] = None) -> HRM:
        return self.sample(key, extras)

    @abstractmethod
    def sample(self, key: chex.PRNGKey, extras: Optional[Dict] = None) -> HRM:
        """
        Returns a randomly generated HRM from the specified key.
        """
        raise NotImplementedError

    def unwrapped(self) -> "HRMSampler":
        return self

    def _init_hrm(self, root_id: int, extras: Optional[Dict] = None) -> HRM:
        return ops.init_hrm(
            root_id=root_id,
            max_num_rms=self._max_num_rms,
            max_num_states=self._max_num_states,
            max_num_edges=self._max_num_edges,
            max_num_literals=self._max_num_literals,
            extras=extras,
        )
