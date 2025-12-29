from abc import ABC, abstractmethod
from typing import Tuple

import chex
from chex import dataclass
from flax import linen as nn
from flax import struct

from ..envs.common.labeling_function import LabelingFunction
from ..hrm.types import HRM, HRMState


@dataclass
class ConditionerState:
    """
    Encodes the state of the conditioner (e.g., if it employs some kind of
    internal memory).
    """

    pass


class ConditionerOutput(struct.PyTreeNode):
    """
    Encodes the output produced by the conditioner.

    Args:
        - conditioning_vector: the vector that will be used to condition the
            policy on.
    """

    conditioning_vector: chex.Array


class Conditioner(ABC, nn.Module):
    """
    The module responsible for producing HRM representations to condition policies on.
    """
    label_fn: LabelingFunction        # Number of propositions in the alphabet

    @abstractmethod
    def __call__(
        self,
        c_state: ConditionerState,
        hrm: HRM,
        hrm_state: HRMState,
        *args,
        **kwargs,
    ) -> Tuple[ConditionerState, ConditionerOutput]:
        """
        Args:
            c_state: state of the conditioner.
            hrm: the HRM to which the policy is conditioned, whose leaves are of shape
                [B,T,...], where B is the batch size, and T is the number of timesteps.
            hrm_state: the state within the HRM to condition to, whose shape is as above.
        """
        raise NotImplementedError

    def initialize_state(self, batch_size: int, rng: chex.PRNGKey, **kwargs) -> ConditionerState:
        """
        Initializes the state of the conditioner, which is mandatory
        before making a call to the module.
        """
        return ConditionerState()
