from abc import abstractmethod, ABC
from typing import Generic, Tuple, TypeVar

import chex
import numpy as np

from .level import Level
from .types import Timestep

EnvParams = TypeVar("EnvParams")
_LevelT = TypeVar("_LevelT", bound=Level)


class Environment(ABC, Generic[EnvParams, _LevelT]):
    """
    Abstraction of the environments used in the framework.
    """

    @abstractmethod
    def default_params(self, **kwargs) -> EnvParams:
        raise NotImplementedError

    @abstractmethod
    def num_actions(self, env_params: EnvParams) -> int:
        raise NotImplementedError

    @abstractmethod
    def reset(
        self, key: chex.PRNGKey, env_params: EnvParams, level: _LevelT, **kwargs,
    ) -> Timestep:
        """
        Resets the environment to a given level (i.e., grid layout/morphology).

        Warning: The Timestep structure is very similar to XLand-Minigrid's as of now.
        The purpose of the abstraction is to make it uniform across any future
        environments we may deal with.

        Inspiration:
        https://github.com/DramaCow/jaxued/blob/main/src/jaxued/environments/underspecified_env.py

        Args:
            - key: the random key that will be used to determine the initial state
                of the environment.
            - env_params: the parameters of the environment.
            - level: the level to which the environment is reset.

        Returns:
            - timestep: A `Timestep` containing all the initial info (state, observation, ...).
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self, env_params: EnvParams, timestep: Timestep, action: chex.Array
    ) -> Timestep:
        """
        Performs a step in the environment.

        Warning: The Timestep structure is very similar to XLand-Minigrid's as of now.
        The purpose of the abstraction is to make it uniform across any future
        environments we may deal with.

        Args:
            - env_params: the parameters of the environment.
            - timestep: the current timestep information (state, observation, ...).
            - action: the action to be performed.

        Returns:
            - timestep: the next timestep information.
        """
        raise NotImplementedError

    @abstractmethod
    def render(
        self, env_params: EnvParams, timestep: Timestep
    ) -> np.ndarray | str:
        raise NotImplementedError
