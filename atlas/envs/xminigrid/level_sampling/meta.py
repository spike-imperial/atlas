from typing import Dict, List, Optional, Union

import chex
from hydra.utils import instantiate
import jax
from omegaconf import ListConfig

from .base import XMinigridLevelSampler
from ..level import Level
from ..types import XMinigridEnvParams


class XMinigridMetaLevelSampler(XMinigridLevelSampler):
    """
    Samples uniformly at random from one of the samplers passed as an argument.
    """

    def __init__(self, env_params: XMinigridEnvParams, samplers: Union[ListConfig, List[XMinigridLevelSampler]]):
        super().__init__(env_params)
        if isinstance(samplers, ListConfig):
            self._samplers = [
                instantiate(sampler, env_params=env_params)
                for sampler in samplers
            ]
        else:
            self._samplers = samplers

        self._max_num_objects = max([
            self._samplers[i].get_max_num_objects()
            for i in range(len(self._samplers))
        ])

    def sample(self, key: chex.PRNGKey, extras: Optional[Dict] = None) -> Level:
        meta_key, sampler_key = jax.random.split(key, 2)
        return jax.lax.switch(
            jax.random.randint(meta_key, (), 0, len(self._samplers)),
            self._samplers,
            sampler_key,
            extras,
        )

    def get_max_num_objects(self):
        return self._max_num_objects
