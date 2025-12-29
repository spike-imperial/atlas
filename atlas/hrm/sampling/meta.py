from typing import Dict, List, Optional, Union

import chex
from hydra.utils import instantiate
import jax
from omegaconf import ListConfig

from .common import HRMSampler
from ..types import HRM


class MetaHRMSampler(HRMSampler):
    """
    Samples uniformly at random from one of the samplers passed as an argument.
    """

    def __init__(
        self,
        samplers: Union[ListConfig, List[HRMSampler]],
        max_num_rms: int,
        max_num_states: int,
        max_num_edges: int,
        max_num_literals: int,
        alphabet_size: int,
        **kwargs: dict,
    ):
        super().__init__(max_num_rms, max_num_states, max_num_edges, max_num_literals, alphabet_size)
        if isinstance(samplers, ListConfig):
            self._samplers = [
                instantiate(
                    sampler,
                    max_num_rms=max_num_rms,
                    max_num_states=max_num_states,
                    max_num_edges=max_num_edges,
                    max_num_literals=max_num_literals,
                    alphabet_size=alphabet_size,
                )
                for sampler in samplers
            ]
        else:
            self._samplers = samplers

    def sample(self, key: chex.PRNGKey, extras: Optional[Dict] = None) -> HRM:
        meta_key, sampler_key = jax.random.split(key, 2)
        return jax.lax.switch(
            jax.random.randint(meta_key, (), 0, len(self._samplers)),
            self._samplers,
            sampler_key,
            extras,
        )
