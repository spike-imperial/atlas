import functools
from typing import Dict, List, Optional, Tuple

import chex
from chex import dataclass
import jax
import jax.numpy as jnp
import yaml

from .types import (
    Formula,
    HRM,
    HRMReward,
    HRMState,
    Label,
    SatTransition,
    StackFields,
)


def init_hrm(
    root_id: int,
    max_num_rms: int,
    max_num_states: int,
    max_num_edges: int,
    max_num_literals: int,
    extras: Optional[Dict] = None,
    **kwargs,
) -> HRM:
    """
    Initializes an empty HRM.

    Args:
        root_id: Identifier of the root of the hierarchy.
        max_num_rms: The maximum number of RMs in the hierarchy.
        max_num_states: The maximum number of states per RM.
        max_num_edges: The maximum number of edges from one state to another.
        max_num_literals: The maximum number of literals per edge.

    Returns:
        hrm: A `dummy` HRM.

    Note:
        Choosing appropriate values for `max_num_rms`, `max_num_states`,
        `max_num_edges`, and `max_num_literals` is important for performance. Large
        values can lead to increased memory usage and slower computation. Consider the
        specific requirements of your use case and adjust these parameters accordingly.
    """
    return HRM(
        root_id=jnp.array(root_id),
        calls=-jnp.ones(
            (max_num_rms, max_num_states, max_num_states, max_num_edges),
            dtype=jnp.int32,
        ),
        formulas=jnp.zeros(
            (
                max_num_rms,
                max_num_states,
                max_num_states,
                max_num_edges,
                max_num_literals,
            ),
            dtype=jnp.int32,  # CHANGE PRECISION!
        ),
        num_literals=jnp.zeros((max_num_rms, max_num_states, max_num_states, max_num_edges), dtype=jnp.int32),
        rewards=jnp.zeros((max_num_rms, max_num_states, max_num_states)),
        extras=extras,
    )


def hrm_like(hrm: HRM) -> HRM:
    """
    Returns an initial HRM with the same bounds as the input HRM.
    """
    return init_hrm(
        hrm.root_id,
        get_max_num_machines(hrm),
        get_max_num_states_per_machine(hrm),
        get_max_num_edges_per_state_pair(hrm),
        get_max_num_literals(hrm),
        extras=jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), hrm.extras)
    )


def add_call(
    hrm: HRM,
    rm_id: int,
    src_id: int,
    dst_id: int,
    edge_id: int,
    called_rm_id: int,
) -> None:
    """
    Adds a call to an HRM.

    Args:
        hrm: The HRM to modify.
        rm_id: The RM for which a call is going to be added.
        src_id: The state from which the call is made.
        dst_id: The next state after the call is satisfied.
        edge_id: The identifier of the edge between `src_id` and `dst_id` associated with the call.
        called_rm_id: The identifier of the called RM.
    """
    hrm.calls = hrm.calls.at[rm_id, src_id, dst_id, edge_id].set(called_rm_id)


def add_leaf_call(
    hrm: HRM,
    rm_id: int,
    src_id: int,
    dst_id: int,
    edge_id: int,
) -> None:
    """
    Adds a call to the leaf RM in an HRM.

    Args:
        hrm: The HRM to modify.
        rm_id: The RM for which a call is going to be added.
        src_id: The state from which the call is made.
        dst_id: The next state after the call is satisfied.
        edge_id: The identifier of the edge between `src_id` and `dst_id` associated with the call.
    """
    add_call(hrm, rm_id, src_id, dst_id, edge_id, get_max_num_machines(hrm))


def add_condition(
    hrm: HRM,
    rm_id: int,
    src_id: int,
    dst_id: int,
    edge_id: int,
    proposition: int,
    is_positive: bool,
) -> None:
    """
    Adds a literal to an edge between two states.

    Args:
        hrm: The HRM to modify.
        rm_id: The RM where the literal is going to be added.
        src_id: The source of the edge where the literal is added.
        dst_id: The destination of the edge where the literal is added.
        edge_id: The identifier of the edge between `src_id` and `dst_id`.
        proposition: The proposition to be included in the edge.
        is_positive: Whether the proposition appears positively.
    """
    num_literals = hrm.num_literals[rm_id, src_id, dst_id, edge_id]
    literal = (2 * is_positive - 1) * (proposition + 1)  # +1 is to make sure it is not 0...
    hrm.formulas = hrm.formulas.at[rm_id, src_id, dst_id, edge_id, num_literals].set(literal)
    hrm.num_literals = hrm.num_literals.at[rm_id, src_id, dst_id, edge_id].set(num_literals + 1)


def add_reward(
    hrm: HRM,
    rm_id: int,
    src_id: int,
    dst_id: int,
    reward: float,
) -> None:
    """
    Associates a state transition with a reward scalar.

    Args:
        hrm: The HRM to modify.
        rm_id: The RM where the reward is specified.
        src_id: The source of the transition associated with the reward.
        dst_id: The destination of the transition associated with the reward.
        reward: The reward scalar.
    """
    hrm.rewards = hrm.rewards.at[rm_id, src_id, dst_id].set(reward)


def get_initial_hrm_state(hrm: HRM) -> HRMState:
    """
    Returns the initial HRM state in the given HRM.
    """
    return HRMState(
        rm_id=hrm.root_id,
        state_id=jnp.array(0),
        stack=-jnp.ones((get_max_num_machines(hrm), len(StackFields)), dtype=jnp.int32),
        stack_size=jnp.array(0),
    )


def get_initial_hrm_reward(hrm: HRM) -> HRMReward:
    """
    Returns the initial HRM reward. Since no transitions have yet been
    taken, everything is set to 0 and the mask indicates that no reward
    has been produced by any of the RMs in the hierarchy.
    """
    max_num_rms = get_max_num_machines(hrm)
    return HRMReward(
        scalar=jnp.zeros((max_num_rms,)),
        mask=jnp.zeros((max_num_rms,), dtype=jnp.bool_),
        src_id=jnp.zeros((max_num_rms,), dtype=jnp.int32),
        dst_id=jnp.zeros((max_num_rms,), dtype=jnp.int32),
    )


def get_max_num_machines(hrm: HRM) -> int:
    """
    Returns the maximum number of RMs contained in the given HRM.
    """
    return hrm.formulas.shape[0]


def get_max_num_states_per_machine(hrm: HRM) -> int:
    """
    Returns the maximum number of states per RM in the given HRM.
    """
    return hrm.formulas.shape[1]


def get_max_num_edges_per_state_pair(hrm: HRM) -> int:
    """
    Returns the maximum number of edges from one state to another in the given HRM.
    """
    return hrm.formulas.shape[3]


def _get_max_transitions_from_state(hrm):
    """
    Returns the maximum number of transitions from a state in an HRM.
    """
    return get_max_num_states_per_machine(hrm) * get_max_num_edges_per_state_pair(hrm)


def get_max_num_literals(hrm: HRM) -> int:
    """
    Returns the maximum number of literals allowed in each transition.
    """
    return hrm.formulas.shape[4]


def is_root_rm(hrm: HRM, rm_id: int) -> bool:
    """
    Returns True if the specified RM is the root of the HRM, and False otherwise.
    """
    return rm_id == hrm.root_id


def is_leaf_rm(hrm: HRM, rm_id: int) -> bool:
    """
    Returns True if the specified RM is the leaf of the HRM, and False otherwise.
    """
    return rm_id == get_max_num_machines(hrm)


def is_null_rm(hrm: HRM, rm_id: int) -> bool:
    """
    Returns True if the specified RM identifier does not represent an actual RM in the HRM,
    and False otherwise.
    """
    return (rm_id < 0) | (rm_id > get_max_num_machines(hrm))


def is_empty_rm(hrm: HRM, rm_id: int) -> bool:
    """
    Returns True if the specified RM does not contain any transitions, and False otherwise.
    """
    return jnp.sum(hrm.calls[rm_id]) == -(
        get_max_num_states_per_machine(hrm) ** 2
    ) * get_max_num_edges_per_state_pair(hrm)


def is_initial_state(state_id: int) -> bool:
    """
    Returns True if the specified state is the initial state of *any* RM, and False otherwise.
    """
    return state_id == get_initial_state_id()


def get_initial_state_id() -> int:
    """
    Returns the id of the initial state of *any* RM.
    """
    return 0


def is_accepting_state(hrm: HRM, state_id: int) -> bool:
    """
    Returns True if the specified state is the accepting state of *any* RM, and False otherwise.
    """
    return state_id == get_accepting_state_id(hrm)


def get_accepting_state_id(hrm: HRM) -> int:
    """
    Returns the id of the accepting state of *any* RM in the hierarchy.
    """
    return get_max_num_states_per_machine(hrm) - 1


def is_terminal_state(hrm: HRM, rm_id: int, state_id: int) -> bool:
    """
    Returns True if the specified state does not have any outgoing transitions,
    and False otherwise.
    """
    return jnp.sum(hrm.calls[rm_id, state_id]) == -get_max_num_states_per_machine(
        hrm
    ) * get_max_num_edges_per_state_pair(hrm)


def step(hrm: HRM, hrm_state: HRMState, label: Label) -> Tuple[HRMState, HRMReward]:
    """
    Performs a step in the HRM.

    Args:
        hrm: The HRM where the step is performed.
        hrm_state: The HRM state from which the step is performed.
        label: A truth assignment of propositions in the HRM's alphabet.

    Returns:
        next_hrm_state: The next HRM state.
        reward: The rewards obtained at the different transitions taken throughout the HRM.
    """
    hrm_state, hrm_reward = _step_top_down(hrm, hrm_state, label)
    return _step_bottom_up(hrm, hrm_state, hrm_reward)


def traverse(hrm: HRM, label_trace: chex.Array) -> Tuple[HRMState, chex.Array]:
    """
    Returns the sequence of HRM states and rewards induced by performing a step for
    each label in the input label trace starting from the initial HRM state.
    """

    def _step_aux(
        hrm_state: HRMState, label: Label
    ) -> Tuple[HRMState, Tuple[HRMState, float]]:
        next_hrm_state, hrm_rewards = step(hrm, hrm_state, label)
        return next_hrm_state, (next_hrm_state, hrm_rewards)

    _, (hrm_states, hrm_rewards) = jax.lax.scan(
        f=_step_aux,
        init=get_initial_hrm_state(hrm),
        xs=label_trace,
    )

    return hrm_states, hrm_rewards


def _step_top_down(
    hrm: HRM, hrm_state: HRMState, label: Label
) -> Tuple[HRMState, HRMReward]:
    """
    Returns the HRM state that results from applying a step using a label *but*
    without returning control to the machines in the call stack.

    Implementation:
        - Get the satisfiability state of the transitions from the current RM
          state and from the initial state for other RMs. The entries of the
         `sat_transitions` have shape (max_num_rms, max_num_states * max_num_edges).
        - Loop through the entries of `sat_transitions` using iterators: one
          keeps track of the RM being checked (starting from the one in the
          input HRM state), and one for each RM that goes through each entry
          in `sat_transitions`.
        - If the HRM state has not changed (i.e., the reward mask sums zero),
          the reward given is that corresponding to a loop in the current
          state of the current RM.
    """
    sat_transitions = _get_sat_transitions(hrm, hrm_state, label)

    @dataclass
    class _LoopState:
        """
        hrm_state: the current HRM state in the loop.
        it_current: identifier of the RM we currently loop through.
        it_counters: the count for each of the RMs in the hierarchy (i.e., what entry
            of sat_transitions is currently being examined for that RM).
        """

        hrm_state: HRMState
        hrm_reward: HRMReward
        it_current: chex.Numeric
        it_counters: chex.Array  # (max_num_rms,) initialized to zeros

    def _cond_fun(loop_state: _LoopState):
        """
        The loop will go on until we have gone through the entries for the *current* RM
        (that in the HRM state) or we get to the leaf RM. Note that the loop goes on if
        we have gone through all the entries of the current RM but we haven't still
        fully examined the called RMs (hrm_state.rm_id != loop_state.it_current).
        """
        max_counter_value = _get_max_transitions_from_state(hrm)
        is_under_max_value = loop_state.it_counters[hrm_state.rm_id] < max_counter_value
        return (
            is_under_max_value | (hrm_state.rm_id != loop_state.it_current)
        ) & ~is_leaf_rm(hrm, loop_state.it_current)

    def _body_fun(loop_state: _LoopState) -> _LoopState:
        """
        Returns the new loop state after checking whether the current transition is
        satisfied and the iterator across the transitions has a valid value (it could
        have been increased above the limit upon calling a potential RM and remain invalid,
        and JAX does not check whether it is out of bounds!).
        The loop state iterators change differently depending on the result (see called
        functions for details).
        """
        is_valid_it_value = loop_state.it_counters[
            loop_state.it_current
        ] < _get_max_transitions_from_state(hrm)
        return jax.lax.cond(
            pred=sat_transitions.is_satisfied[
                loop_state.it_current, loop_state.it_counters[loop_state.it_current]
            ]
            & is_valid_it_value,
            true_fun=_on_sat_transition,
            false_fun=_on_unsat_transition,
            operand=loop_state,
        )

    def _on_sat_transition(loop_state: _LoopState) -> _LoopState:
        """
        Returns the next loop state when the transition is satisfied. The next loop
        state depends on the called RM:
            - If it is the leaf, the state within the current RM is updated. The
              pointer is assigned to the leaf, and the loop will terminate in the
              next iteration (see `cond_fun`).
            - If it is a non-leaf RM, the call is pushed onto the stack and its size
              is updated (+1). The next RM checked we will iterate over is the one
              being called, whose associated iterator is reset to 0.
        In both cases, the iterator for the current RM is increased by 1.
        """
        current_rm = loop_state.it_current
        counter_value = loop_state.it_counters[current_rm]

        called_rm = sat_transitions.called_rm_id[current_rm, counter_value]
        src_state = sat_transitions.src_id[current_rm, counter_value]
        dst_state = sat_transitions.dst_id[current_rm, counter_value]
        next_it_counters = loop_state.it_counters.at[current_rm].set(counter_value + 1)

        return jax.lax.cond(
            pred=is_leaf_rm(hrm, called_rm),
            true_fun=lambda: _LoopState(
                hrm_state=HRMState(
                    rm_id=current_rm,
                    state_id=dst_state,
                    stack=loop_state.hrm_state.stack,
                    stack_size=loop_state.hrm_state.stack_size,
                ),
                hrm_reward=HRMReward(
                    scalar=loop_state.hrm_reward.scalar.at[current_rm].set(
                        hrm.rewards[current_rm, src_state, dst_state]
                    ),
                    mask=loop_state.hrm_reward.mask.at[current_rm].set(True),
                    src_id=loop_state.hrm_reward.src_id.at[current_rm].set(src_state),
                    dst_id=loop_state.hrm_reward.dst_id.at[current_rm].set(dst_state),
                ),
                it_current=called_rm,
                it_counters=next_it_counters,
            ),
            false_fun=lambda: _LoopState(
                hrm_state=HRMState(
                    rm_id=loop_state.hrm_state.rm_id,
                    state_id=loop_state.hrm_state.state_id,
                    stack=loop_state.hrm_state.stack.at[
                        loop_state.hrm_state.stack_size
                    ].set(jnp.array([current_rm, called_rm, src_state, dst_state])),
                    stack_size=loop_state.hrm_state.stack_size + 1,
                ),
                hrm_reward=loop_state.hrm_reward,
                it_current=called_rm,
                it_counters=next_it_counters.at[called_rm].set(0),
            ),
        )

    def _on_unsat_transition(loop_state: _LoopState) -> _LoopState:
        """
        Returns the next loop state when the transition is unsatisfied:
            - If the current RM is not the current HRM state's and we went through
              all the entries for it, it means this RM is not involved in a satisfied
              transition; hence, we remove it from the stack (simply decrease the
              stack size) and set the pointer to the calling rm.
            - Otherwise, we will check the next entry for the current RM without
              changing the HRM state.
        """
        current_rm = loop_state.it_current
        counter_value = loop_state.it_counters[current_rm]
        next_it_counters = loop_state.it_counters.at[current_rm].set(counter_value + 1)

        return jax.lax.cond(
            pred=(
                (current_rm != hrm_state.rm_id)
                & (counter_value >= _get_max_transitions_from_state(hrm))
            ),
            true_fun=lambda: _LoopState(
                hrm_state=HRMState(
                    rm_id=loop_state.hrm_state.rm_id,
                    state_id=loop_state.hrm_state.state_id,
                    stack=loop_state.hrm_state.stack,
                    stack_size=loop_state.hrm_state.stack_size - 1,
                ),
                hrm_reward=loop_state.hrm_reward,
                it_current=loop_state.hrm_state.stack[
                    loop_state.hrm_state.stack_size - 1, StackFields.CALLING_RM
                ],
                it_counters=next_it_counters,
            ),
            false_fun=lambda: _LoopState(
                hrm_state=loop_state.hrm_state,
                hrm_reward=loop_state.hrm_reward,
                it_current=current_rm,
                it_counters=next_it_counters,
            ),
        )

    end_loop_state = jax.lax.while_loop(
        init_val=_LoopState(
            hrm_state=hrm_state,
            hrm_reward=HRMReward(
                scalar=jnp.zeros((get_max_num_machines(hrm),)),
                mask=jnp.zeros((get_max_num_machines(hrm),), dtype=jnp.bool_),
                src_id=jnp.zeros((get_max_num_machines(hrm),), dtype=jnp.int32),
                dst_id=jnp.zeros((get_max_num_machines(hrm),), dtype=jnp.int32),
            ),
            it_current=hrm_state.rm_id,
            it_counters=jnp.zeros((get_max_num_machines(hrm),), dtype=jnp.int32),
        ),
        cond_fun=_cond_fun,
        body_fun=_body_fun,
    )

    hrm_state, hrm_reward = end_loop_state.hrm_state, end_loop_state.hrm_reward

    # If no transition has been taken, the reward in the current RM corresponds
    # to that in the self-loop to the current state.
    return hrm_state, jax.lax.cond(
        pred=jnp.sum(hrm_reward.mask) > 0,
        true_fun=lambda: hrm_reward,
        false_fun=lambda: HRMReward(
            scalar=hrm_reward.scalar.at[hrm_state.rm_id].set(
                hrm.rewards[hrm_state.rm_id, hrm_state.state_id, hrm_state.state_id],
            ),
            mask=hrm_reward.mask.at[hrm_state.rm_id].set(True),
            src_id=hrm_reward.src_id.at[hrm_state.rm_id].set(hrm_state.state_id),
            dst_id=hrm_reward.dst_id.at[hrm_state.rm_id].set(hrm_state.state_id),
        ),
    )


def _step_bottom_up(
    hrm: HRM, hrm_state: HRMState, hrm_reward: HRMReward
) -> Tuple[HRMState, HRMReward]:
    """
    Returns the HRM state that results by returning control to the calling RMs
    when the accepting state of the called RMs is reached.

    Implementation:
        - For jit-efficiency (though counterintuitive), it seems better to start the
          scan from the maximum stack level (i.e., the maximum number of RMs in the
          HRM) since it is not a quantity dependent on the HRM state at hand.
        - The HRM state *only changes* when control needs to be returned, i.e. the
          iterator matches the stack size and the current state is an accepting state.
          The stack does not need to change: we only change its size (the integer).
        - The HRM reward changes both when control is returned since a transition is
          taken on the calling RM, and when it is not since we need to account for
          the self-loops in the states of the stack.
    """

    @dataclass
    class _LoopState:
        hrm_state: HRMState
        hrm_reward: HRMReward

    def _on_return_control(loop_state: _LoopState, stack_level: int) -> _LoopState:
        calling_rm = loop_state.hrm_state.stack[stack_level - 1, StackFields.CALLING_RM]
        src_state = loop_state.hrm_state.stack[
            stack_level - 1, StackFields.SRC_STATE_CALLING_RM
        ]
        dst_state = loop_state.hrm_state.stack[
            stack_level - 1, StackFields.DST_STATE_CALLING_RM
        ]

        return _LoopState(
            hrm_state=HRMState(
                rm_id=calling_rm,
                state_id=dst_state,
                stack=loop_state.hrm_state.stack,
                stack_size=jnp.clip(loop_state.hrm_state.stack_size - 1, a_min=0),
            ),
            hrm_reward=HRMReward(
                scalar=loop_state.hrm_reward.scalar.at[calling_rm].set(
                    hrm.rewards[calling_rm, src_state, dst_state]
                ),
                mask=loop_state.hrm_reward.mask.at[calling_rm].set(True),
                src_id=loop_state.hrm_reward.src_id.at[calling_rm].set(src_state),
                dst_id=loop_state.hrm_reward.dst_id.at[calling_rm].set(dst_state),
            ),
        )

    def _on_self_loop(loop_state: _LoopState, stack_level: int) -> _LoopState:
        calling_rm = loop_state.hrm_state.stack[stack_level - 1, StackFields.CALLING_RM]
        src_state = loop_state.hrm_state.stack[
            stack_level - 1, StackFields.SRC_STATE_CALLING_RM
        ]

        return _LoopState(
            hrm_state=loop_state.hrm_state,
            hrm_reward=HRMReward(
                scalar=loop_state.hrm_reward.scalar.at[calling_rm].set(
                    hrm.rewards[calling_rm, src_state, src_state]
                ),
                mask=loop_state.hrm_reward.mask.at[calling_rm].set(True),
                src_id=loop_state.hrm_reward.src_id.at[calling_rm].set(src_state),
                dst_id=loop_state.hrm_reward.dst_id.at[calling_rm].set(src_state),
            ),
        )

    end_loop_state, _ = jax.lax.scan(
        init=_LoopState(hrm_state=hrm_state, hrm_reward=hrm_reward),
        xs=jnp.arange(start=get_max_num_machines(hrm), stop=0, step=-1),
        f=lambda loop_state, stack_level: jax.lax.cond(
            pred=(
                (stack_level == loop_state.hrm_state.stack_size)
                & is_accepting_state(hrm, loop_state.hrm_state.state_id)
            ),
            true_fun=lambda: (_on_return_control(loop_state, stack_level), 0),
            false_fun=lambda: jax.lax.cond(
                pred=stack_level
                <= loop_state.hrm_state.stack_size,  # if we are considering valid stack levels
                true_fun=lambda: (_on_self_loop(loop_state, stack_level), 0),
                false_fun=lambda: (loop_state, 0),
            ),
        ),
    )

    return end_loop_state.hrm_state, end_loop_state.hrm_reward


def _get_sat_transitions(hrm: HRM, hrm_state: HRMState, label: Label) -> SatTransition:
    """
    Returns a structure containing for each transition from the current RM state and
    each transition from the initial state of other RMs: the next state, the called RM,
    and whether the formula associated with the transitions is satisfied. The shape of
    each field is (max_num_rms, max_num_states * max_num_edges).
    """

    def _is_condition_satisfied(formula: Formula, num_literals: int, called_rm: int):
        """
        Returns True if the condition, constituted by a formula and the called RM, is
        satisfied by the label. Otherwise, it returns False.
        """
        # The formula is satisfied if the number of literals in the formula matches the
        # number of satisfied literals.
        formula_props = jnp.abs(formula) - 1
        formula_signs = jnp.sign(formula)
        mask = jnp.arange(get_max_num_literals(hrm)) < num_literals

        sat_literals = label[formula_props] * formula_signs * mask
        num_sat_literals = sat_literals.sum()
        is_formula_satisfied = num_sat_literals == num_literals

        # Three cases:
        #   0: if the called RM is not a leaf or null, check if the formula is satisfied.
        #   1: if the called RM is a leaf, check if the formula is satisfied and if the
        #      number of literals is greater than 0. That is, we do not allow unconditional
        #      transitions if no actual RM is called; in other words, only calls to non-leaf
        #      and non-null RMs can satisfy formulas that are always satisfiable (expressed as
        #      full zero arrays).
        #   2: if the called RM is null, the condition is never satisfied.
        return jax.lax.switch(
            index=is_leaf_rm(hrm, called_rm) + 2 * is_null_rm(hrm, called_rm),
            branches=[
                lambda: is_formula_satisfied,
                lambda: is_formula_satisfied & (num_sat_literals > 0),
                lambda: False,
            ],
        )

    @functools.partial(jax.vmap, in_axes=(0, 0, None, None))
    @functools.partial(jax.vmap, in_axes=(None, None, 0, None))
    @functools.partial(jax.vmap, in_axes=(None, None, None, 0))
    def _is_transition_satisfied(rm_id, src_id, dst_id, edge_id):
        formula = hrm.formulas[rm_id, src_id, dst_id, edge_id]
        num_literals = hrm.num_literals[rm_id, src_id, dst_id, edge_id]
        called_rm_id = hrm.calls[rm_id, src_id, dst_id, edge_id]
        return SatTransition(
            src_id=src_id,
            dst_id=dst_id,
            called_rm_id=called_rm_id,
            is_satisfied=_is_condition_satisfied(formula, num_literals, called_rm_id),
        )

    # Get the satisfied transitions from the initial state of every RM except for
    # the current one (we check the satisfied transitions from the current state in
    # that case). The `vmap` checks the transitions from these states to all possible
    # states in the RM using all possible edges.
    # The shape of each field is (max_num_rms, max_num_states * max_num_edges).
    return jax.tree_util.tree_map(
        lambda x: jnp.reshape(
            x,
            (
                get_max_num_machines(hrm),
                get_max_num_states_per_machine(hrm)
                * get_max_num_edges_per_state_pair(hrm),
            ),
        ),
        _is_transition_satisfied(
            jnp.arange(0, get_max_num_machines(hrm)),
            jnp.zeros((get_max_num_machines(hrm),), dtype=jnp.int32)
            .at[hrm_state.rm_id]
            .set(hrm_state.state_id),
            jnp.arange(0, get_max_num_states_per_machine(hrm)),
            jnp.arange(0, get_max_num_edges_per_state_pair(hrm)),
        ),
    )


def get_num_machines(hrm: HRM) -> int:
    """
    Returns the number of actual RMs in the hierarchy estimated
    as the number of RMs that call and get called (i.e., no
    reachability is tested).
    """
    # A machine is used if it makes calls and gets called
    return jnp.sum(get_machine_mask(hrm))


def get_machine_mask(hrm: HRM) -> chex.Array:
    """
    Returns a Boolean array indicating which RMs are active
    in the HRM, i.e. make AND receive calls (root included
    even it cannot receive calls).
    """
    return get_calling_rms(hrm) * get_called_rms(hrm)


def get_calling_rms(hrm: HRM) -> chex.Array:
    """
    Returns a Boolean array indicating which RMs make any calls.
    """
    calls = jnp.reshape(hrm.calls, (get_max_num_machines(hrm), -1))
    return jnp.any(calls > -1, axis=1)


def get_called_rms(hrm: HRM) -> chex.Array:
    """
    Returns a Boolean array indicating which RMs are called.
    The root is marked as called even though it cannot be called
    by any RM.
    """
    calls = jnp.reshape(hrm.calls, (get_max_num_machines(hrm), -1))

    # Count the number of times each RM is called
    num_called = jax.ops.segment_sum(
        jnp.ones(((get_max_num_states_per_machine(hrm) ** 2) * get_max_num_edges_per_state_pair(hrm),)),
        calls,
        num_segments=get_max_num_machines(hrm),
    )

    # A machine is called if it gets called >0 times
    called_rms = num_called > 0
    return called_rms.at[hrm.root_id].set(True)


def get_num_rm_states(hrm: HRM, rm_id: int) -> chex.Array:
    max_num_states = get_max_num_states_per_machine(hrm)
    max_num_edges = get_max_num_edges_per_state_pair(hrm)

    def _get_src_states(rm_calls: chex.Array) -> chex.Array:
        # Consider the accepting state as a source too
        rm_calls = jnp.reshape(rm_calls, (max_num_states, -1))
        src_states = jnp.any(rm_calls > -1, axis=1)
        return src_states.at[-1].set(True)

    def _get_dst_states(rm_calls: chex.Array) -> chex.Array:
        called_states = jnp.ones_like(rm_calls) * jnp.arange(max_num_states)[:, jnp.newaxis]
        mask = rm_calls > -1
        called_states = mask * called_states - jnp.logical_not(mask)
        called_states = jnp.reshape(called_states, (max_num_states, -1))
        num_dst = jax.ops.segment_sum(
            jnp.ones((max_num_states * max_num_edges,)),
            called_states,
            num_segments=max_num_states,
        )
        is_dst = num_dst > 0
        return is_dst.at[0].set(True)

    rm_calls = hrm.calls[rm_id]
    return jnp.sum(_get_src_states(rm_calls) * _get_dst_states(rm_calls))


def get_num_states(hrm: HRM) -> chex.Array:
    """
    Returns the number of states of each constituent RM in the hierarchy.
    The number of states is determined as the number of states that have
    incoming and outgoing edges (not through reachability).
    """
    def _get_num_rm_states(rm_id: int, mask: bool) -> int:
        return mask * get_num_rm_states(hrm, rm_id)

    return jax.vmap(_get_num_rm_states)(
        jnp.arange(get_max_num_machines(hrm)),
        get_calling_rms(hrm) * get_called_rms(hrm)
    )


def get_proposition_frequency(hrm: HRM, alphabet_size: int) -> chex.Array:
    """
    Returns the number of times each proposition appears in the hierarchy
    (positively or negatively). Only those for valid calls are checked
    (but it is not checked whether the edge they label is actually reachable).
    """
    calls_mask = (hrm.calls > -1)[..., jnp.newaxis]
    formulas = jnp.abs(calls_mask * hrm.formulas) - 1  # remove symbol (abs) and index from 0
    formulas = formulas.flatten()
    return jax.ops.segment_sum(jnp.ones_like(formulas), formulas, alphabet_size)


def get_propositions(hrm: HRM, alphabet_size: int) -> chex.Array:
    """
    Returns the propositions appearing in the HRM. The vector has size
    `alphabet_size + 1`: there are at most `alphabet_size` propositions
    and the `+1` accounts for the edges labeled without propositions.
    The fixed size makes the method jittable.
    """
    # Get the propositions from the literals
    propositions = jnp.abs(hrm.formulas) - 1

    return jnp.unique(propositions, size=alphabet_size + 1, fill_value=-1)


def load(hrm: HRM, path: str, alphabet: Optional[List[str]] = None) -> None:
    """
    Loads an HRM from a .yaml file into the given HRM.

    Args:
        hrm: an empty HRM instance.
        path: the path to the .yaml file.
        alphabet: alphabet overriding the loaded one.

    Example:
        ```
        alphabet:
          - a
          - b
        root: m0
        transitions:
          m0:
            u0:
              uA:
                reward: 0
                edges:
                  - formula: [a, -b]
                    call: leaf
        ```
    """

    def _get_rm_id(rm_name: str) -> int:
        if rm_name == "leaf":
            return get_max_num_machines(hrm)
        rm_id = int(rm_name[1:])
        if rm_id < 0 or rm_id >= get_max_num_machines(hrm):
            raise RuntimeError(
                "Reward machine identifier out of range "
                f"(0 <= id < {get_max_num_machines(hrm)})."
            )
        return rm_id

    def _get_state_id(state_name: str) -> int:
        if state_name == "uA":
            return get_accepting_state_id(hrm)
        state_id = int(state_name[1:])
        if state_id < 0 or state_id > get_accepting_state_id(hrm):
            raise RuntimeError(
                "State identifier out of range "
                f"(0 <= id < {get_accepting_state_id(hrm)})."
            )
        return state_id

    def _get_literal_info(literal: str, alphabet: list) -> Tuple[int, bool]:
        is_positive = True
        if literal.startswith("-"):
            is_positive = False
            literal = literal[1:]
        return alphabet.index(literal), is_positive

    # Read the file
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Parse the alphabet
    alphabet = alphabet if alphabet else config["alphabet"]

    # Parse the root
    root = config["root"]
    if root not in config["transitions"]:
        raise RuntimeError("The transitions for the root are unspecified.")
    hrm.root_id = _get_rm_id(root)

    # Parse the transitions
    for rm, transitions in config["transitions"].items():
        rm_id = _get_rm_id(rm)
        for src_state in transitions:
            src_state_id = _get_state_id(src_state)
            for dst_state in transitions[src_state]:
                dst_state_id = _get_state_id(dst_state)
                add_reward(
                    hrm,
                    rm_id,
                    src_state_id,
                    dst_state_id,
                    transitions[src_state][dst_state]["reward"],
                )
                for edge_id in range(len(transitions[src_state][dst_state]["edges"])):
                    edge = transitions[src_state][dst_state]["edges"][edge_id]
                    add_call(
                        hrm,
                        rm_id,
                        src_state_id,
                        dst_state_id,
                        edge_id,
                        _get_rm_id(edge["call"]),
                    )
                    for literal in edge["formula"]:
                        proposition_id, is_positive = _get_literal_info(
                            literal, alphabet
                        )
                        add_condition(
                            hrm,
                            rm_id,
                            src_state_id,
                            dst_state_id,
                            edge_id,
                            proposition_id,
                            is_positive,
                        )


def dump(hrm: HRM, path: str, alphabet: Optional[List[str]] = None, include_alphabet: bool = False) -> None:
    """
    Dumps an HRM into YAML format (see `load` for more info).

    Args:
        hrm: an empty HRM instance.
        path: the path to the .yaml file.
        alphabet: alphabet (if used the propositions are dumped with their string value), o.w. just an integer.
        include_alphabet: whether to dump the alphabet in the YAML file.
    """

    def _get_state_name(state_id: int) -> str:
        return "uA" if is_accepting_state(hrm, state_id) else f"u{state_id}"

    # Get transitions
    transitions = {}
    for rm_id in range(get_max_num_machines(hrm)):
        if is_empty_rm(hrm, rm_id):
            continue

        rm_transitions = {}
        for src_state in range(get_max_num_states_per_machine(hrm)):
            dst_transitions = {}
            for dst_state in range(get_max_num_states_per_machine(hrm)):
                edges = []
                for edge_id in range(get_max_num_edges_per_state_pair(hrm)):
                    called_rm = hrm.calls[rm_id, src_state, dst_state, edge_id]
                    if is_null_rm(hrm, called_rm):
                        continue

                    called_rm = "leaf" if is_leaf_rm(hrm, called_rm) else f"m{called_rm}"
                    formula = []
                    for i in range(hrm.num_literals[rm_id, src_state, dst_state, edge_id]):
                        literal = hrm.formulas[rm_id, src_state, dst_state, edge_id, i]
                        prop_id = jnp.abs(literal) - 1
                        symbol = "" if literal > 0 else "-"
                        if alphabet is not None:
                            prop_id = alphabet[int(prop_id)]
                        formula.append(f"{symbol}{prop_id}")

                    edges.append({"formula": formula, "call": called_rm})

                if len(edges) > 0:
                    dst_transitions[_get_state_name(dst_state)] = {
                        "reward": float(hrm.rewards[rm_id, src_state, dst_state]),
                        "edges": edges
                    }

            if len(dst_transitions) > 0:
                rm_transitions[_get_state_name(src_state)] = dst_transitions

        transitions[f"m{rm_id}"] = rm_transitions

    # Get alphabet
    if include_alphabet and alphabet is not None:
        alphabet_dict = {"alphabet": alphabet}
    else:
        alphabet_dict = {}

    # Build the dictionary with all info
    with open(path, 'w') as f:
        yaml.safe_dump({
            **alphabet_dict,
            "root": f"m{hrm.root_id}",
            "transitions": transitions,
        }, f)


def split(hrm: HRM, hrm_state: HRMState) -> Tuple[HRM, HRM]:
    """
    Returns HRMs for the predecessor states and the successor states.
    The input HRM is assumed to be:
      (i) the HRM is flat,
      (ii) there is a single edge between states, and
      (iii) it is sequential [a.k.a. single-path] and the states are sequentially indexed.
    The states are indexed sequentially from 0 in the resulting HRM. If the state is
    the initial state or the accepting state, the split is not produced and a copy
    is returned. The reward is all 0 except for the transition to the accepting state (1.0).
    """
    assert get_max_num_machines(hrm) == 1 and get_max_num_edges_per_state_pair(hrm) == 1 and get_max_num_literals(hrm) == 1
    return split_predecessors(hrm, hrm_state), split_successors(hrm, hrm_state)


def split_predecessors(hrm: HRM, hrm_state: HRMState) -> HRM:
    """
    Returns the predecessor bit from an HRM given an HRM state.
    See documentation of the `split` method.
    """
    assert get_max_num_machines(hrm) == 1 and get_max_num_edges_per_state_pair(hrm) == 1 and get_max_num_literals(hrm) == 1

    max_num_rms = get_max_num_machines(hrm)
    max_num_states = get_max_num_states_per_machine(hrm)
    max_num_edges = get_max_num_edges_per_state_pair(hrm)
    max_num_literals = get_max_num_literals(hrm)

    def _dummy_predecessor():
        pred_hrm = init_hrm(
            hrm.root_id,
            get_max_num_machines(hrm),
            get_max_num_states_per_machine(hrm),
            get_max_num_edges_per_state_pair(hrm),
            get_max_num_literals(hrm),
            hrm.extras,
        )
        add_leaf_call(pred_hrm, 0, get_initial_state_id(), get_accepting_state_id(pred_hrm), 0)
        return pred_hrm

    def _predecessor():
        state_id = hrm_state.state_id
        leaf_rm = get_max_num_machines(hrm)
        u_acc = get_accepting_state_id(hrm)

        mask = (
            jnp.arange(max_num_rms * max_num_states * max_num_states * max_num_edges) <
            (state_id - 1) * max_num_states * max_num_edges
        ).reshape((max_num_rms, max_num_states, max_num_states, max_num_edges))

        # Nullify calls from the predecessor state and onwards, and
        # add edge to accepting state
        calls = mask * hrm.calls - jnp.logical_not(mask)
        calls = calls.at[0, state_id - 1, u_acc].set(leaf_rm)

        # Remove the literals from the state and succeeding states, and
        # add the original number of literals required to the accepting state
        num_literals = mask * hrm.num_literals
        num_literals = num_literals.at[0, state_id - 1, u_acc].set(1)

        # Remove rewards from the state and succeeding states, and
        # add a reward of 1 to the accepting state from the state
        rewards = mask.squeeze(axis=-1) * hrm.rewards
        rewards = rewards.at[0, state_id - 1, u_acc].set(1.0)

        # Remove formulas from the state and succeeding states, and
        # add the formula from the state to the accepting state
        fmask = (
            jnp.arange(max_num_rms * max_num_states * max_num_states * max_num_edges * max_num_literals) <
            (state_id - 1) * max_num_states * max_num_edges * max_num_literals
        ).reshape((max_num_rms, max_num_states, max_num_states, max_num_edges, max_num_literals))
        formulas = fmask * hrm.formulas
        formulas = formulas.at[0, state_id - 1, u_acc].set(hrm.formulas[0, state_id - 1, state_id])

        return HRM(root_id=0, calls=calls, formulas=formulas, num_literals=num_literals, rewards=rewards, extras=hrm.extras)

    return jax.lax.switch(
        index=is_initial_state(hrm_state.state_id) + 2 * is_accepting_state(hrm, hrm_state.state_id),
        branches=[
            _predecessor,
            _dummy_predecessor,
            lambda: jax.tree_util.tree_map(lambda x: jnp.copy(x), hrm),  # clone
        ]
    )


def split_successors(hrm: HRM, hrm_state: HRMState) -> HRM:
    """
    Returns the predecessor bit from an HRM given an HRM state.
    See documentation of the `split` method.
    """
    max_num_rms = get_max_num_machines(hrm)
    max_num_states = get_max_num_states_per_machine(hrm)
    max_num_edges = get_max_num_edges_per_state_pair(hrm)
    max_num_literals = get_max_num_literals(hrm)

    def _dummy_successor():
        succ_hrm = init_hrm(
            hrm.root_id,
            get_max_num_machines(hrm),
            get_max_num_states_per_machine(hrm),
            get_max_num_edges_per_state_pair(hrm),
            get_max_num_literals(hrm),
            hrm.extras,
        )
        add_leaf_call(succ_hrm, 0, get_initial_state_id(), get_accepting_state_id(succ_hrm), 0)
        return succ_hrm

    def _successor():
        state_id = hrm_state.state_id
        leaf_rm = get_max_num_machines(hrm)
        u_acc = get_accepting_state_id(hrm)
        num_states = get_num_rm_states(hrm, rm_id=0)
        num_rem_states = num_states - state_id

        mask = (
            jnp.arange(max_num_rms * max_num_states * max_num_states * max_num_edges) <
            (num_rem_states - 2) * max_num_states * max_num_edges
        ).reshape((max_num_rms, max_num_states, max_num_states, max_num_edges))

        # The original calls structure is reused given its sequentiality (u0 -> u1, u1 -> u2, ...)
        # and flatness: only remove edges for the states that are left unused
        # and add an edge from the new penultimate state to the accepting state.
        calls = mask * hrm.calls - jnp.logical_not(mask)
        calls = calls.at[0, num_rem_states - 2, u_acc].set(leaf_rm)

        # Similar modification (requirement for one literal could be easily lifted)
        num_literals = mask * hrm.num_literals
        num_literals = num_literals.at[0, num_rem_states - 2, u_acc].set(1)

        # Similar modification to the one above (assuming reward 1 for transition to accepting state)
        rewards = mask.squeeze(axis=-1) * hrm.rewards
        rewards = rewards.at[0, num_rem_states - 2, u_acc].set(1.0)

        # Move the formulas to make the current state id become the initial state and rewire connections,
        # copy the formula from the original, and
        # set the rest of the formulas to 0.
        fmask = (
            jnp.arange(max_num_rms * max_num_states * max_num_states * max_num_edges * max_num_literals) <
            (num_rem_states - 1) * max_num_states * max_num_edges * max_num_literals
        ).reshape((max_num_rms, max_num_states, max_num_states, max_num_edges, max_num_literals))
        formulas = jnp.roll(hrm.formulas, -state_id, axis=[1, 2])
        formulas = formulas.at[0, num_rem_states - 2].set(hrm.formulas[0, num_states - 2])
        formulas = fmask * formulas

        return HRM(root_id=0, calls=calls, formulas=formulas, num_literals=num_literals, rewards=rewards, extras=hrm.extras)

    return jax.lax.switch(
        index=is_initial_state(hrm_state.state_id) + 2 * is_accepting_state(hrm, hrm_state.state_id),
        branches=[
            _successor,
            lambda: jax.tree_util.tree_map(lambda x: jnp.copy(x), hrm),  # clone
            _dummy_successor,
        ]
    )


def get_hrm_completion(hrm: HRM, hrm_state: HRMState) -> float:
    """
    Returns the percent of HRM completed given an HRM state.

    Warning: It assumes the HRM is sequential and flat, with increasing
    sequential state identifiers.
    """
    max_state_id = get_num_rm_states(hrm, hrm.root_id) - 1
    state_id = jnp.minimum(hrm_state.state_id, max_state_id)
    return state_id / max_state_id


def topsort(hrm: HRM) -> Tuple[chex.Array, int]:
    """
    Performs a topological sort of the root RM in the HRM passed as an argument.
    Returns the topological sort and the number of states in the sorted array (note
    it is padded with -1).
    Warning: Assumes the graph is acyclic.
    """
    # Convert calls in an easy to handle adjacency matrix
    adj = jnp.any(hrm.calls[0] > -1, axis=-1)

    # Create the initial sorted array and set the initial state to be the first element
    top_sort = -jnp.ones((get_max_num_states_per_machine(hrm),), dtype=jnp.int32)
    top_sort = top_sort.at[0].set(get_initial_state_id())
    top_sort_size = 1

    # A mask for valid states (states that have incoming or outgoing edges)
    valid = jnp.logical_or(jnp.any(adj, axis=0), jnp.any(adj, axis=1))

    # A mask for the states that have already been selected for the top sort
    selected = jnp.zeros((get_max_num_states_per_machine(hrm),), dtype=jnp.bool)
    selected = selected.at[get_initial_state_id()].set(True)

    # Remove edges from the initial state in the topological sort
    adj = adj.at[get_initial_state_id()].set(False)

    # While there is a valid state without incoming edges that hasn't been selected
    def _f(carry, _):
        _top_sort, _size, _adj, _selected = carry

        wo_in_edges = ~jnp.any(_adj, axis=0)
        cond = wo_in_edges & valid & ~_selected
        states = jnp.where(cond, size=get_max_num_states_per_machine(hrm), fill_value=-1)[0]

        return jax.lax.cond(
            pred=jnp.any(states > -1),
            true_fun=lambda: (
                (
                    _top_sort.at[_size].set(states[0]),
                    _size + 1,
                    _adj.at[states[0]].set(False),
                    _selected.at[states[0]].set(True)
                ),
                None
            ),
            false_fun=lambda: ((_top_sort, _size, _adj, _selected), None)
        )

    (top_sort, top_sort_size, *_), _ = jax.lax.scan(
        _f,
    (top_sort, top_sort_size, adj, selected),
        length=get_max_num_states_per_machine(hrm)
    )

    return top_sort, top_sort_size


def path_info(hrm: HRM) -> Tuple[chex.Array, chex.Array]:
    """
    Returns the number of paths and the average path length in an HRM
    that is flat and acyclic.
    """
    # Get topological sorting
    top_sort, top_sort_size = topsort(hrm)

    # Convert calls in an easy to handle adjacency matrix
    adj = jnp.any(hrm.calls[0] > -1, axis=-1)

    def _f(carry, state):
        path_length_sum, num_paths = carry

        return jax.lax.switch(
            index=is_initial_state(state) + 2 * (state > get_initial_state_id()),
            branches=[
                lambda: ((path_length_sum, num_paths), None),
                lambda: ((path_length_sum, num_paths.at[state].set(1)), None),
                lambda: ((
                    path_length_sum.at[state].set(
                        jnp.sum(adj[:, state] * (path_length_sum + adj[:, state] * num_paths))
                    ),
                    num_paths.at[state].set(jnp.sum(adj[:, state] * num_paths))
                ), None)
            ]
        )

    (path_length_sum, num_paths), _ = jax.lax.scan(
        _f,
        (
            jnp.zeros((get_max_num_states_per_machine(hrm)), dtype=jnp.int32),  # path_length_sum
            jnp.zeros((get_max_num_states_per_machine(hrm)), dtype=jnp.int32)   # number of paths
        ),
        top_sort
    )

    avg_path_length = path_length_sum / num_paths
    return num_paths[get_accepting_state_id(hrm)], avg_path_length[get_accepting_state_id(hrm)],
