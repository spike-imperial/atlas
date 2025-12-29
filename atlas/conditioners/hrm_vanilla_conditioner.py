from typing import Tuple

import jax
import jax.numpy as jnp

from .types import Conditioner, ConditionerOutput, ConditionerState
from ..hrm.ops import get_max_num_machines, get_max_num_states_per_machine
from ..hrm.types import HRMState, HRM


class VanillaHRMConditioner(Conditioner):
    """
    Produces a conditioning on the concatenation of the one-hot vector
    representations of the current RM id and the RM state id of the HRM.
    The conditioning reward is that obtained in the root of the hierarchy.
    """

    def __call__(
        self, c_state: ConditionerState, hrm: HRM, hrm_state: HRMState, *args, **kwargs
    ) -> Tuple[ConditionerState, ConditionerOutput]:
        def _get_c_output(hrm: HRM, hrm_state: HRMState) -> ConditionerOutput:
            return ConditionerOutput(
                conditioning_vector=jnp.concat(
                    (
                        jnp.zeros((get_max_num_machines(hrm),))
                        .at[hrm_state.rm_id]
                        .set(1),
                        jnp.zeros((get_max_num_states_per_machine(hrm),))
                        .at[hrm_state.state_id]
                        .set(1),
                    )
                ),
            )

        # Since inputs have shape [B,T,...] we need to vmap over B and T
        c_outputs = jax.vmap(jax.vmap(_get_c_output))(
            hrm,
            hrm_state,
        )
        return ConditionerState(), c_outputs
