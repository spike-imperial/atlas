from enum import IntEnum
from typing import Dict, Optional

import chex
from chex import dataclass

"""
An array representing a conjunction of literals. The i-th position can be one of the following:
    +1 if the i-th proposition appears as a positive literal,
     0 if the i-th proposition does not appear, or
    -1 if the i-th proposition appears as a negative literal.
"""
Formula = chex.Array

"""
An array representing the truth assignment of propositions in the HRM's alphabet.
The i-th position is +1 if the i-th proposition is observed, and -1 otherwise.
"""
Label = chex.Array


@dataclass
class HRM:
    """
    root_id: the identifier of the root RM in the hierarchy.
    calls: identifier of the RMs associated with each edge.
    formulas: formulas labeling each edge in the HRM (each expressing a conjunction of literals).
    rewards: rewards associated with each pair of states.
    extras: a dictionary with additional information (e.g., sampling information)
    """

    root_id: chex.Numeric  # scalar
    calls: chex.Array  # (max_num_rms, max_num_states, max_num_states, max_num_edges) default=-1
    formulas: (
        chex.Array
    )  # (max_num_rms, max_num_states, max_num_states, max_num_edges, max_num_literals) defualt=0
    num_literals: chex.Array  # (max_num_rms, max_num_states, max_num_states, max_num_edges)
    rewards: chex.Array  # (max_num_rms, max_num_states, max_num_states)
    extras: Optional[Dict] = None


@dataclass
class HRMState:
    """
    rm_id: identifier of the RM (0 <= rm_id < max_num_rms).
    state_id: identifier of the state within the RM rm_id (0 <= state_id < max_num_states).
    stack: the call stack containing RMs and states to which control must be returned.
    stack_size: size of the stack (0 <= stack_size < max_num_rms).
    """

    rm_id: chex.Numeric  # scalar
    state_id: chex.Numeric  # scalar
    stack: chex.Array  # (max_num_rms, num_stack_fields)
    stack_size: chex.Numeric  # scalar


class StackFields(IntEnum):
    """
    The indices of each of the components in an item of the stack.

    CALLING_RM: The RM from which the call is made.
    CALLED_RM: The RM being called.
    NEXT_STATE_CALLING_RM: The next state in the CALLING_RM when the accepting state in
        the CALLED_RM is reached.

    TODO: The `stack` field of the HRMState dataclass could be of a custom dataclass
     type `CallStack`, each of whose members correspond to the fields below. An early
     test showed that the time/step is slightly higher. Since it is syntactic sugar we
     have left it undone by now.
    """

    CALLING_RM = 0
    CALLED_RM = 1
    SRC_STATE_CALLING_RM = 2
    DST_STATE_CALLING_RM = 3


@dataclass
class HRMReward:
    """
    scalar: the reward obtained in each RM after a step.
    mask: the RMs involved in the last step.
    src_id: the source state of an RM involved in the step.
    dst_id: the destination state of an RM involved in the step.
    """

    scalar: chex.Array  # (max_num_rms,)
    mask: chex.Array  # (max_num_rms,)
    src_id: chex.Array  # (max_num_rms,)
    dst_id: chex.Array  # (max_num_rms,)


@dataclass
class SatTransition:
    """
    src_id: the source state of the transition.
    dst_id: the destination state of the transition.
    called_rm_id: the identifier of the RM called in the transition.
    is_satisfied: whether the formula associated with the transition is satisfied.
    """

    src_id: chex.Numeric
    dst_id: chex.Numeric
    called_rm_id: chex.Numeric
    is_satisfied: chex.Numeric
