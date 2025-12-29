import jax.numpy as jnp

from .types import Conditioner, ConditionerOutput, ConditionerState
from ..hrm.types import HRM, HRMState


class DummyConditioner(Conditioner):
    """
    Conditioner that has a dummy state and whose output conditioning
    vector is empty.
    """

    def __call__(
        self, c_state: ConditionerState, hrm: HRM, hrm_state: HRMState, *args, **kwargs
    ):
        batch_size, sequence_length, *_ = hrm_state.rm_id.shape
        return ConditionerState(), ConditionerOutput(
            conditioning_vector=jnp.zeros((batch_size, sequence_length, 0)),
        )
