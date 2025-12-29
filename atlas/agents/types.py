from typing import NamedTuple

import chex

from ..conditioners.types import ConditionerState


class ConditionedAgentState(NamedTuple):
    c_state: ConditionerState    # state of the conditioner
    a_state: chex.Array  # state of the agent (the network)

