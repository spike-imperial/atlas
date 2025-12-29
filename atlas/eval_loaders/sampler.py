import math

import jax

from .types import EvaluationProblem, EvaluationSetLoader
from ..problem_samplers.base import ProblemSampler


class SampledEvaluationSetLoader(EvaluationSetLoader):
    """
    Samples evaluation sets using specified level and HRM samplers.
    """

    def __init__(
        self,
        seed: int,
        problem_sampler: ProblemSampler,
        num_problems: int,
        **kwargs: dict,
    ):
        self._rng = jax.random.PRNGKey(seed)
        self._problem_sampler = problem_sampler
        self._num_problems = num_problems

    def load(self) -> EvaluationProblem:
        levels, hrms = jax.vmap(self._problem_sampler)(
            jax.random.split(self._rng, self._num_problems)
        )
        return EvaluationProblem(levels, hrms)

    def get_num_problems(self) -> int:
        return self._num_problems

    def get_problem_name(self, idx: int) -> str:
        return str(idx).zfill(int(math.log10(self._num_problems)) + 1)
