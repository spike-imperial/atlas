from typing import Any, TypeVar

import chex
from flax import struct
import jax.numpy as jnp


class EnvParams(struct.PyTreeNode):
    """
    Abstraction of the parameters that characterize an environment. This class
    should be extended for any environment we deal with -- it is important to
    define default values to ensure reusability.

    Warning: It is not feasible to `vmap` over this class; thus, the fields
    are fixed for any environment we initialize. Variability in the grids
    should be delegated to the Levels.
    """
    pass


_StateT = TypeVar("_StateT")


class StepType(jnp.uint8):
    """
    Abstraction of step types following XLand-Minigrid and Jumanji.
    """
    FIRST: chex.Array = jnp.asarray(0, dtype=jnp.uint8)
    MID: chex.Array = jnp.asarray(1, dtype=jnp.uint8)
    LAST: chex.Array = jnp.asarray(2, dtype=jnp.uint8)


class TimestepExtras(struct.PyTreeNode):
    """
    Abstraction of the extras for each timestep.
    """
    pass


class Timestep(struct.PyTreeNode):
    """
    Abstraction of the information received after each step or when the environment
    is reset. 
    
    The structure currently follows that of XLand-Minigrid and could easily capture 
    that of suites like Jumanji, which decouples the state from the rest of the
    information.
    """

    key: chex.PRNGKey
    state: _StateT
    step_type: StepType
    reward: chex.Array
    discount: chex.Array
    observation: Any
    num_steps: chex.Array
    extras: TimestepExtras = TimestepExtras()

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST
