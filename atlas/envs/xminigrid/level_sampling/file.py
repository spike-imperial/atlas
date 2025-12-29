from typing import Dict, List, Optional

import chex
import jax
from jax import numpy as jnp

from .base import XMinigridLevelSampler
from ..level import XMinigridLevel
from ..types import XMinigridEnvParams
from ...common.level import Level


class XMinigridFileLevelSampler(XMinigridLevelSampler):
    def __init__(self, env_params: XMinigridEnvParams, files: List[str]):
        super().__init__(env_params)
        levels = [XMinigridLevel.from_file(f, env_params) for f in files]
        self._levels = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *levels)
        self._num_hrms = len(levels)

    def sample(self, key: chex.PRNGKey, extras: Optional[Dict] = None) -> Level:
        idx = jax.random.randint(key, shape=(), minval=0, maxval=self._num_hrms)
        return jax.tree_map(lambda x: x[idx], self._levels)
