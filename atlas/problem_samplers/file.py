from typing import Callable, Dict, List, Tuple

import chex
import jax
import jax.numpy as jnp

from .base import ProblemSampler
from ..envs.common.env import EnvParams
from ..envs.common.labeling_function import LabelingFunction
from ..envs.common.level import Level
from ..hrm import ops
from ..hrm.types import HRM


class FileProblemSampler(ProblemSampler):
    def __init__(
        self,
        files: List[Dict[str, str]],
        level_loading_fn: Callable,
        env_params: EnvParams,
        max_num_rms: int,
        max_num_states: int,
        max_num_edges: int,
        max_num_literals: int,
        label_fn: LabelingFunction,
        **kwargs,
    ):
        levels, hrms = [], []
        alphabet = label_fn.get_str_alphabet()
        for f in files:
            levels.append(level_loading_fn(f["level_path"], env_params))
            hrm = ops.init_hrm(
                root_id=0,
                max_num_rms=max_num_rms,
                max_num_states=max_num_states,
                max_num_edges=max_num_edges,
                max_num_literals=max_num_literals,
            )
            ops.load(hrm, f["hrm_path"], alphabet)
            hrms.append(hrm)

        self._num_problems = len(levels)
        self._levels = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *levels)
        self._hrms = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *hrms)

    def sample(self, rng: chex.PRNGKey) -> Tuple[Level, HRM]:
        idx = jax.random.randint(rng, shape=(), minval=0, maxval=self._num_problems)
        return jax.tree_util.tree_map(
            lambda x: x[idx], (self._levels, self._hrms)
        )
