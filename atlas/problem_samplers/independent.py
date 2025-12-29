from typing import Tuple

import chex
from hydra.utils import instantiate
import jax

from .base import ProblemSampler
from ..envs.common.labeling_function import LabelingFunction
from ..envs.common.level import Level
from ..envs.common.level_sampling import LevelSampler
from ..hrm.sampling.common import HRMSampler
from ..hrm.types import HRM


class IndependentProblemSampler(ProblemSampler):
    """
    Samples an HRM and a level independently of each other.
    """
    def __init__(self, level_sampler: LevelSampler, hrm_sampler: HRMSampler, label_fn: LabelingFunction):
        self._level_sampler = level_sampler
        self._hrm_sampler = hrm_sampler
        self._label_fn = label_fn

    def sample(self, rng: chex.PRNGKey) -> Tuple[Level, HRM]:
        level_rng, hrm_rng = jax.random.split(rng, 2)
        return self._level_sampler.sample(level_rng), self._hrm_sampler.sample(hrm_rng)

    @classmethod
    def build(cls, level_sampler, hrm_sampler, hrm_sampler_wrapper, env_params, label_fn, gamma):
        return cls(
            instantiate(level_sampler, env_params=env_params),
            instantiate(
                hrm_sampler_wrapper,
                sampler=instantiate(
                    hrm_sampler,
                    alphabet_size=label_fn.get_alphabet_size(),
                    alphabet=label_fn.get_str_alphabet(),
                ),
                gamma=gamma
            ),
            label_fn
        )
