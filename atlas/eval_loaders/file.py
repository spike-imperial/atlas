import functools
import math
import os
from typing import Callable, Dict, List, Union

import jax
import jax.numpy as jnp
from tqdm import tqdm

from .types import EvaluationProblem, EvaluationSetLoader
from ..envs.common.types import EnvParams
from ..hrm.ops import init_hrm, load


class FileEvaluationSetLoader(EvaluationSetLoader):
    """
    Samples evaluation sets specified in files.
    """

    def __init__(
        self,
        files: Union[List[Dict[str, str]], str],
        level_loading_fn: Callable,
        env_params: EnvParams,
        max_num_rms: int,
        max_num_states: int,
        max_num_edges: int,
        max_num_literals: int,
        alphabet: List[str],
        **kwargs: dict,
    ):
        if isinstance(files, str):
            levels = sorted([x for x in os.listdir(os.path.join(files, "levels")) if x.endswith(".yaml")])
            hrms = sorted([x for x in os.listdir(os.path.join(files, "hrms")) if x.endswith(".yaml")])
            assert levels == hrms, "Error: File names must be the same"
            self._files = [
                dict(
                    level_path=os.path.join(files, "levels", name),
                    hrm_path=os.path.join(files, "hrms", name),
                    name=name[:-len(".yaml")]
                )
                for name in levels
            ]
        else:
            self._files = files

        self._num_problems = len(self._files)

        # Level loading
        self._level_loading_fn = functools.partial(level_loading_fn, env_params=env_params)

        # HRM loading
        self._init_hrm_fn = lambda: init_hrm(
            root_id=0,
            max_num_rms=max_num_rms,
            max_num_states=max_num_states,
            max_num_edges=max_num_edges,
            max_num_literals=max_num_literals,
        )
        self._alphabet = alphabet

    def load(self) -> EvaluationProblem:
        levels = []
        hrms = []

        for i in tqdm(range(self._num_problems), desc="Loading problems..."):
            level = self._level_loading_fn(self._files[i]["level_path"])
            levels.append(level)

            hrm = self._init_hrm_fn()
            load(hrm, self._files[i]["hrm_path"], self._alphabet)
            hrms.append(hrm)

        return EvaluationProblem(
            level=jax.tree_util.tree_map(lambda *x: jnp.stack(x), *levels),
            hrm=jax.tree_util.tree_map(lambda *x: jnp.stack(x), *hrms),
        )

    def get_num_problems(self) -> int:
        return self._num_problems

    def get_problem_name(self, idx: int) -> str:
        name = self._files[idx]["name"]
        if name:
            return name
        return str(idx).zfill(int(math.log10(self._num_problems)) + 1)
