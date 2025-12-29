from abc import abstractmethod, ABC
from typing import List

from flax import struct

from ..envs.common.level import Level
from ..hrm.types import HRM


class EvaluationProblem(struct.PyTreeNode):
    level: Level
    hrm: HRM


class EvaluationSetLoader(ABC):
    """
    Abstraction class for loading evaluation problems.
    """

    @abstractmethod
    def load(self) -> EvaluationProblem:
        raise NotImplementedError

    @abstractmethod
    def get_num_problems(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_problem_name(self, problem_id: int) -> str:
        raise NotImplementedError

    def get_problem_names(self) -> List[str]:
        return [
            self.get_problem_name(problem_id)
            for problem_id in range(self.get_num_problems())
        ]
