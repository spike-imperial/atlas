from pathlib import Path
import sys

import chex
import jax
from jax import numpy as jnp

from atlas.hrm import ops
from atlas.hrm.types import HRM, HRMReward, HRMState, StackFields

PROJECT_DIR = Path(sys.argv[1]).parent


def _test_traversal_step_fn(
    hrm: HRM,
    label_trace: chex.Array,
    expected_hrm_states: HRMState,
    expected_hrm_rewards: HRMReward,
):
    chex.clear_trace_counter()
    step_fn = jax.jit(chex.assert_max_traces(ops.step, n=1))

    hrm_state = ops.get_initial_hrm_state(hrm)
    for i in range(len(label_trace)):
        hrm_state, hrm_reward = step_fn(hrm, hrm_state, label_trace[i])
        assert hrm_state.rm_id == expected_hrm_states.rm_id[i]
        assert hrm_state.state_id == expected_hrm_states.state_id[i]
        assert hrm_state.stack_size == expected_hrm_states.stack_size[i]
        if hrm_state.stack_size > 0:
            assert jnp.array_equal(
                hrm_state.stack[0 : hrm_state.stack_size],
                expected_hrm_states.stack[i, 0 : hrm_state.stack_size],
            )
        assert jnp.array_equal(hrm_reward.scalar, expected_hrm_rewards.scalar[i])
        assert jnp.array_equal(hrm_reward.mask, expected_hrm_rewards.mask[i])
        assert jnp.array_equal(hrm_reward.src_id, expected_hrm_rewards.src_id[i])
        assert jnp.array_equal(hrm_reward.dst_id, expected_hrm_rewards.dst_id[i])


def _test_traversal_traverse_fn(
    hrm: HRM,
    label_trace: chex.Array,
    expected_hrm_states: HRMState,
    expected_hrm_rewards: HRMReward,
):
    traverse_fn = jax.jit(ops.traverse)
    hrm_states, hrm_rewards = traverse_fn(hrm, label_trace)

    assert jnp.array_equal(hrm_states.rm_id, expected_hrm_states.rm_id)
    assert jnp.array_equal(hrm_states.state_id, expected_hrm_states.state_id)
    assert jnp.array_equal(hrm_states.stack_size, expected_hrm_states.stack_size)
    for i in range(0, len(hrm_states.stack)):
        stack_size = hrm_states.stack_size[i]
        if stack_size > 0:
            assert jnp.array_equal(
                hrm_states.stack[i, 0:stack_size],
                expected_hrm_states.stack[i, 0:stack_size],
            )
    assert jnp.array_equal(hrm_rewards.scalar, expected_hrm_rewards.scalar)
    assert jnp.array_equal(hrm_rewards.mask, expected_hrm_rewards.mask)
    assert jnp.array_equal(hrm_rewards.src_id, expected_hrm_rewards.src_id)
    assert jnp.array_equal(hrm_rewards.dst_id, expected_hrm_rewards.dst_id)


def _test_traversal(
    hrm: HRM,
    label_trace: chex.Array,
    expected_hrm_states: HRMState,
    expected_hrm_rewards: HRMReward,
):
    _test_traversal_step_fn(
        hrm,
        label_trace,
        expected_hrm_states,
        expected_hrm_rewards,
    )
    _test_traversal_traverse_fn(
        hrm,
        label_trace,
        expected_hrm_states,
        expected_hrm_rewards,
    )


class TestSimpleFlatHRM:
    def test_initial_hrm_state(self):
        initial_state = ops.get_initial_hrm_state(self.init_hrm())
        assert initial_state.rm_id == 0
        assert initial_state.state_id == 0

    def test_step(self):
        label_trace = jnp.array([[1, -1], [-1, -1], [-1, 1]])

        _test_traversal(
            hrm=self.init_hrm(),
            label_trace=label_trace,
            expected_hrm_states=HRMState(
                rm_id=jnp.array([0, 0, 0], dtype=jnp.int32),
                state_id=jnp.array([1, 1, 2], dtype=jnp.int32),
                stack=-jnp.ones(
                    (label_trace.shape[0], 1, len(StackFields)), dtype=jnp.int32
                ),
                stack_size=jnp.zeros((label_trace.shape[0],), dtype=jnp.int32),
            ),
            expected_hrm_rewards=HRMReward(
                scalar=jnp.array([[0], [0], [1]]),
                mask=jnp.ones((label_trace.shape[0], 1), dtype=jnp.bool_),
                src_id=jnp.array([[0], [1], [1]]),
                dst_id=jnp.array([[1], [1], [2]]),
            ),
        )

    def test_step_multiple_instances(self):
        """
        Checks that the step function is only compiled once even if different HRM
        instances are used. For the check to work, all HRM instances must have the
        same internal shapes (i.e., same maximum number of RMs, maximum number of
        states, maximum number of edges and maximum alphabet size).
        """
        step_fn = jax.jit(chex.assert_max_traces(ops.step, n=1))
        label = jnp.array([[1, -1]])

        hrm1 = self.init_hrm()
        step_fn(hrm1, ops.get_initial_hrm_state(hrm1), label)

        hrm2 = self.init_hrm()
        step_fn(hrm2, ops.get_initial_hrm_state(hrm2), label)

    def test_root_rm(self):
        hrm = self.init_hrm()
        assert ops.is_root_rm(hrm, 0)

    def test_root_initial_state(self):
        assert ops.is_initial_state(0)
        assert not (ops.is_initial_state(1) or ops.is_initial_state(2))

    def test_root_accepting_state(self):
        hrm = self.init_hrm()
        assert not (ops.is_accepting_state(hrm, 0) or ops.is_accepting_state(hrm, 1))
        assert ops.is_accepting_state(hrm, 2)

    def test_leaf_rm_id(self):
        hrm = self.init_hrm()
        assert not ops.is_leaf_rm(hrm, 0)
        assert ops.is_leaf_rm(hrm, 1)

    def init_hrm(self) -> HRM:
        """
        Generates a flat HRM (i.e., with a single RM) for tasks like `observe 'a' then 'b'`.
        """
        hrm = ops.init_hrm(
            root_id=0,
            max_num_rms=1,
            max_num_states=3,
            max_num_edges=1,
            max_num_literals=2,
        )
        ops.load(hrm, PROJECT_DIR / "data/simple_flat_hrm.yaml")
        return hrm


class TestDisjunctiveFlatHRM:
    def test_step(self):
        hrm = self.init_hrm()
        step_fn = jax.jit(chex.assert_max_traces(ops.step, n=1))

        labels = jnp.array([[-1, -1, -1], [1, -1, -1], [-1, -1, 1], [1, -1, 1]])
        next_hrm_states, hrm_rewards = jax.vmap(step_fn, in_axes=(None, None, 0))(
            hrm, ops.get_initial_hrm_state(hrm), labels
        )

        assert jnp.array_equal(
            next_hrm_states.rm_id, jnp.array([0, 0, 0, 0], dtype=jnp.int32)
        )
        assert jnp.array_equal(
            next_hrm_states.state_id, jnp.array([0, 1, 1, 1], dtype=jnp.int32)
        )
        assert jnp.array_equal(
            next_hrm_states.stack_size, jnp.array([0, 0, 0, 0], dtype=jnp.int32)
        )
        assert jnp.array_equal(
            next_hrm_states.stack,
            -jnp.ones((labels.shape[0], 1, len(StackFields)), dtype=jnp.int32),
        )
        assert jnp.array_equal(
            hrm_rewards.scalar, jnp.array([[0], [0.5], [0.5], [0.5]])
        )
        assert jnp.array_equal(
            hrm_rewards.mask, jnp.array([[True], [True], [True], [True]])
        )
        assert jnp.array_equal(hrm_rewards.src_id, jnp.array([[0], [0], [0], [0]]))
        assert jnp.array_equal(hrm_rewards.dst_id, jnp.array([[0], [1], [1], [1]]))

    def init_hrm(self) -> HRM:
        """
        Generates a flat HRM (i.e., with a single RM) for tasks like `observe 'a' or 'c' then 'b'`.
        """
        hrm = ops.init_hrm(
            root_id=0,
            max_num_rms=1,
            max_num_states=3,
            max_num_edges=2,
            max_num_literals=3,
        )
        ops.load(hrm, PROJECT_DIR / "data/disjunctive_flat_hrm.yaml")
        return hrm


class TestDiamondHRM:
    def test_step(self):
        hrm = self.init_hrm()
        _test_traversal(
            hrm=hrm,
            label_trace=jnp.array(
                [[1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=jnp.int32
            ),
            expected_hrm_states=HRMState(
                rm_id=jnp.array([0, 0, 0], dtype=jnp.int32),
                state_id=jnp.array([1, 3, 4], dtype=jnp.int32),
                stack=-jnp.ones((3, 1, len(StackFields)), dtype=jnp.int32),
                stack_size=jnp.zeros((3,), dtype=jnp.int32),
            ),
            expected_hrm_rewards=HRMReward(
                scalar=jnp.array([[0], [0], [1]], dtype=jnp.float32),
                mask=jnp.array([[True], [True], [True]]),
                src_id=jnp.array([[0], [1], [3]], dtype=jnp.int32),
                dst_id=jnp.array([[1], [3], [4]], dtype=jnp.int32),
            ),
        )
        _test_traversal(
            hrm=hrm,
            label_trace=jnp.array(
                [[-1, 1, -1], [1, -1, -1], [-1, -1, 1]], dtype=jnp.int32
            ),
            expected_hrm_states=HRMState(
                rm_id=jnp.array([0, 0, 0], dtype=jnp.int32),
                state_id=jnp.array([2, 3, 4], dtype=jnp.int32),
                stack=-jnp.ones((3, 1, len(StackFields)), dtype=jnp.int32),
                stack_size=jnp.zeros((3,), dtype=jnp.int32),
            ),
            expected_hrm_rewards=HRMReward(
                scalar=jnp.array([[0], [0], [1]], dtype=jnp.float32),
                mask=jnp.array([[True], [True], [True]]),
                src_id=jnp.array([[0], [2], [3]], dtype=jnp.int32),
                dst_id=jnp.array([[2], [3], [4]], dtype=jnp.int32),
            ),
        )
        _test_traversal(
            hrm=hrm,
            label_trace=jnp.array(
                [[1, 1, -1], [1, -1, -1], [-1, -1, 1]], dtype=jnp.int32
            ),
            expected_hrm_states=HRMState(
                rm_id=jnp.array([0, 0, 0], dtype=jnp.int32),
                state_id=jnp.array([2, 3, 4], dtype=jnp.int32),
                stack=-jnp.ones((3, 1, len(StackFields)), dtype=jnp.int32),
                stack_size=jnp.zeros((3,), dtype=jnp.int32),
            ),
            expected_hrm_rewards=HRMReward(
                scalar=jnp.array([[0], [0], [1]], dtype=jnp.float32),
                mask=jnp.array([[True], [True], [True]]),
                src_id=jnp.array([[0], [2], [3]], dtype=jnp.int32),
                dst_id=jnp.array([[2], [3], [4]], dtype=jnp.int32),
            ),
        )
        _test_traversal(
            hrm=hrm,
            label_trace=jnp.array(
                [[-1, -1, 1], [1, 1, -1], [1, -1, -1], [-1, -1, 1]], dtype=jnp.int32
            ),
            expected_hrm_states=HRMState(
                rm_id=jnp.array([0, 0, 0, 0], dtype=jnp.int32),
                state_id=jnp.array([0, 2, 3, 4], dtype=jnp.int32),
                stack=-jnp.ones((4, 1, len(StackFields)), dtype=jnp.int32),
                stack_size=jnp.zeros((4,), dtype=jnp.int32),
            ),
            expected_hrm_rewards=HRMReward(
                scalar=jnp.array([[0], [0], [0], [1]], dtype=jnp.float32),
                mask=jnp.array([[True], [True], [True], [True]]),
                src_id=jnp.array([[0], [0], [2], [3]]),
                dst_id=jnp.array([[0], [2], [3], [4]]),
            ),
        )

    def init_hrm(self) -> HRM:
        hrm = ops.init_hrm(
            root_id=0,
            max_num_rms=1,
            max_num_states=5,
            max_num_edges=1,
            max_num_literals=3,
        )
        ops.load(hrm, PROJECT_DIR / "data/diamond_flat_hrm.yaml")
        return hrm


class TestCraftworldTwoLevelHRM:
    def test_step(self):
        _test_traversal(
            hrm=self.init_hrm(),
            label_trace=jnp.array(
                [
                    [-1, 1, -1, -1],
                    [-1, -1, -1, 1],
                    [1, -1, -1, -1],
                    [-1, -1, -1, 1],
                    [-1, -1, 1, -1],
                ],
                dtype=jnp.int32,
            ),
            expected_hrm_states=HRMState(
                rm_id=jnp.array([1, 0, 2, 0, 0], dtype=jnp.int32),
                state_id=jnp.array([1, 1, 1, 3, 4], dtype=jnp.int32),
                stack=jnp.array(
                    [
                        [[0, 1, 0, 1], [-1, -1, -1, -1], [-1, -1, -1, -1]],
                        [[0, 1, 0, 1], [-1, -1, -1, -1], [-1, -1, -1, -1]],
                        [[0, 2, 1, 3], [-1, -1, -1, -1], [-1, -1, -1, -1]],
                        [[0, 2, 1, 3], [-1, -1, -1, -1], [-1, -1, -1, -1]],
                        [[0, 2, 1, 3], [-1, -1, -1, -1], [-1, -1, -1, -1]],
                    ],
                    dtype=jnp.int32,
                ),
                stack_size=jnp.array([1, 0, 1, 0, 0], dtype=jnp.int32),
            ),
            expected_hrm_rewards=HRMReward(
                scalar=jnp.array(
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1], [1, 0, 0]],
                    dtype=jnp.float32,
                ),
                mask=jnp.array(
                    [
                        [True, True, False],
                        [True, True, False],
                        [True, False, True],
                        [True, False, True],
                        [True, False, False],
                    ]
                ),
                src_id=jnp.array(
                    [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 1], [3, 0, 0]]
                ),
                dst_id=jnp.array(
                    [[0, 1, 0], [1, 4, 0], [1, 0, 1], [3, 0, 4], [4, 0, 0]]
                ),
            ),
        )
        _test_traversal(
            hrm=self.init_hrm(),
            label_trace=jnp.array(
                [
                    [-1, -1, -1, -1],
                    [1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, 1],
                    [-1, -1, -1, -1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [-1, -1, -1, -1],
                    [-1, -1, 1, -1],
                ],
                dtype=jnp.int32,
            ),
            expected_hrm_states=HRMState(
                rm_id=jnp.array([0, 2, 2, 0, 0, 1, 0, 0, 0], dtype=jnp.int32),
                state_id=jnp.array([0, 1, 1, 2, 2, 1, 3, 3, 4], dtype=jnp.int32),
                stack=jnp.array(
                    [
                        [[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]],
                        [[0, 2, 0, 2], [-1, -1, -1, -1], [-1, -1, -1, -1]],
                        [[0, 2, 0, 2], [-1, -1, -1, -1], [-1, -1, -1, -1]],
                        [[0, 2, 0, 2], [-1, -1, -1, -1], [-1, -1, -1, -1]],
                        [[0, 2, 0, 2], [-1, -1, -1, -1], [-1, -1, -1, -1]],
                        [[0, 1, 2, 3], [-1, -1, -1, -1], [-1, -1, -1, -1]],
                        [[0, 1, 2, 3], [-1, -1, -1, -1], [-1, -1, -1, -1]],
                        [[0, 1, 2, 3], [-1, -1, -1, -1], [-1, -1, -1, -1]],
                        [[0, 1, 2, 3], [-1, -1, -1, -1], [-1, -1, -1, -1]],
                    ],
                    dtype=jnp.int32,
                ),
                stack_size=jnp.array([0, 1, 1, 0, 0, 1, 0, 0, 0], dtype=jnp.int32),
            ),
            expected_hrm_rewards=HRMReward(
                scalar=jnp.array(
                    [
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 1],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0],
                        [1, 0, 0],
                    ],
                    dtype=jnp.float32,
                ),
                mask=jnp.array(
                    [
                        [True, False, False],
                        [True, False, True],
                        [True, False, True],
                        [True, False, True],
                        [True, False, False],
                        [True, True, False],
                        [True, True, False],
                        [True, False, False],
                        [True, False, False],
                    ]
                ),
                src_id=jnp.array(
                    [
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 1],
                        [0, 0, 1],
                        [2, 0, 0],
                        [2, 0, 0],
                        [2, 1, 0],
                        [3, 0, 0],
                        [3, 0, 0],
                    ]
                ),
                dst_id=jnp.array(
                    [
                        [0, 0, 0],
                        [0, 0, 1],
                        [0, 0, 1],
                        [2, 0, 4],
                        [2, 0, 0],
                        [2, 1, 0],
                        [3, 4, 0],
                        [3, 0, 0],
                        [4, 0, 0],
                    ]
                ),
            ),
        )

    def init_hrm(self):
        hrm = ops.init_hrm(
            root_id=0,
            max_num_rms=3,
            max_num_states=5,
            max_num_edges=1,
            max_num_literals=4,
        )
        ops.load(hrm, PROJECT_DIR / "data/cw_diamond_2l_hrm.yaml")
        return hrm


class TestFourLevelHRM:
    def test_step(self):
        base_label = -jnp.ones((11,), dtype=jnp.int32)
        base_stack = -jnp.ones((13, len(StackFields)), dtype=jnp.int32)

        base_reward = jnp.zeros((13,), dtype=jnp.int32)
        base_reward_mask = jnp.zeros((13,), dtype=jnp.bool_)

        _test_traversal(
            hrm=self.init_hrm(),
            label_trace=jnp.array(
                [
                    base_label.at[4].set(1),
                    base_label.at[5].set(1),
                    base_label.at[1].set(1),
                    base_label,
                    base_label.at[0].set(1),
                    base_label.at[1].set(1),
                    base_label.at[2].set(1),
                    base_label.at[3].set(1),
                    base_label.at[1].set(1),
                    base_label.at[9].set(1),
                ],
                dtype=jnp.int32,
            ),
            expected_hrm_states=HRMState(
                rm_id=jnp.array([2, 2, 12, 12, 0, 7, 10, 1, 12, 12], dtype=jnp.int32),
                state_id=jnp.array([1, 3, 1, 1, 1, 1, 2, 1, 2, 4], dtype=jnp.int32),
                stack=jnp.array(
                    [
                        base_stack.at[0].set([12, 2, 0, 1]),
                        base_stack.at[0].set([12, 2, 0, 1]),
                        base_stack,
                        base_stack,
                        base_stack.at[0]
                        .set([12, 10, 1, 2])
                        .at[1]
                        .set([10, 7, 0, 2])
                        .at[2]
                        .set([7, 0, 0, 1]),
                        base_stack.at[0].set([12, 10, 1, 2]).at[1].set([10, 7, 0, 2]),
                        base_stack.at[0].set([12, 10, 1, 2]),
                        base_stack.at[0].set([12, 10, 1, 2]).at[1].set([10, 1, 2, 4]),
                        base_stack,
                        base_stack,
                    ]
                ),
                stack_size=jnp.array([1, 1, 0, 0, 3, 2, 1, 2, 0, 0], dtype=jnp.int32),
            ),
            expected_hrm_rewards=HRMReward(
                scalar=jnp.array(
                    [
                        base_reward,
                        base_reward,
                        base_reward.at[2].set(1),
                        base_reward,
                        base_reward,
                        base_reward.at[0].set(1),
                        base_reward.at[7].set(1),
                        base_reward,
                        base_reward.at[1].set(1).at[10].set(1),
                        base_reward.at[12].set(1),
                    ]
                ),
                mask=jnp.array(
                    [
                        base_reward_mask.at[2].set(True).at[12].set(True),
                        base_reward_mask.at[2].set(True).at[12].set(True),
                        base_reward_mask.at[2].set(True).at[12].set(True),
                        base_reward_mask.at[12].set(True),
                        base_reward_mask.at[0]
                        .set(True)
                        .at[7]
                        .set(True)
                        .at[10]
                        .set(True)
                        .at[12]
                        .set(True),
                        base_reward_mask.at[0]
                        .set(True)
                        .at[7]
                        .set(True)
                        .at[10]
                        .set(True)
                        .at[12]
                        .set(True),
                        base_reward_mask.at[7]
                        .set(True)
                        .at[10]
                        .set(True)
                        .at[12]
                        .set(True),
                        base_reward_mask.at[1]
                        .set(True)
                        .at[10]
                        .set(True)
                        .at[12]
                        .set(True),
                        base_reward_mask.at[1]
                        .set(True)
                        .at[10]
                        .set(True)
                        .at[12]
                        .set(True),
                        base_reward_mask.at[12].set(True),
                    ]
                ),
                src_id=jnp.array(
                    [
                        base_reward,
                        base_reward.at[2].set(1),
                        base_reward.at[2].set(3),
                        base_reward.at[12].set(1),
                        base_reward.at[12].set(1),
                        base_reward.at[0].set(1).at[12].set(1),
                        base_reward.at[7].set(1).at[12].set(1),
                        base_reward.at[10].set(2).at[12].set(1),
                        base_reward.at[1].set(1).at[10].set(2).at[12].set(1),
                        base_reward.at[12].set(2),
                    ]
                ),
                dst_id=jnp.array(
                    [
                        base_reward.at[2].set(1),
                        base_reward.at[2].set(3),
                        base_reward.at[2].set(4).at[12].set(1),
                        base_reward.at[12].set(1),
                        base_reward.at[0].set(1).at[12].set(1),
                        base_reward.at[0].set(4).at[7].set(1).at[12].set(1),
                        base_reward.at[7].set(4).at[10].set(2).at[12].set(1),
                        base_reward.at[1].set(1).at[10].set(2).at[12].set(1),
                        base_reward.at[1].set(4).at[10].set(4).at[12].set(2),
                        base_reward.at[12].set(4),
                    ]
                ),
            ),
        )

    def init_hrm(self):
        hrm = ops.init_hrm(
            root_id=12,
            max_num_rms=13,
            max_num_states=5,
            max_num_edges=1,
            max_num_literals=11,
        )
        ops.load(hrm, PROJECT_DIR / "data/cw_4l_hrm_full.yaml")
        return hrm


class TestXMinigridTwoLevelHRM:
    def test_step(self):
        base_label = -jnp.ones((42,), dtype=jnp.int32)
        base_stack = -jnp.ones((4, len(StackFields)), dtype=jnp.int32)

        base_reward = jnp.zeros((4,), dtype=jnp.int32)
        base_reward_mask = jnp.zeros((4,), dtype=jnp.bool_).at[0].set(True)

        _test_traversal(
            hrm=self.init_hrm(),
            label_trace=jnp.array(
                [
                    base_label,
                    base_label.at[24].set(1),
                    base_label.at[27].set(1),
                    base_label.at[35].set(1),
                    base_label.at[4].set(1),
                ],
                dtype=jnp.int32,
            ),
            expected_hrm_states=HRMState(
                rm_id=jnp.array([0, 3, 0, 2, 0], dtype=jnp.int32),
                state_id=jnp.array([0, 2, 2, 2, 3], dtype=jnp.int32),
                stack=jnp.array(
                    [
                        base_stack,
                        base_stack.at[0].set([0, 1, 0, 2]).at[1].set([1, 3, 0, 3]),
                        base_stack,
                        base_stack.at[0].set([0, 2, 2, 3]),
                        base_stack,
                    ]
                ),
                stack_size=jnp.array([0, 2, 0, 1, 0], dtype=jnp.int32),
            ),
            expected_hrm_rewards=HRMReward(
                scalar=jnp.array(
                    [
                        base_reward.astype(jnp.float32),
                        base_reward.astype(jnp.float32),
                        base_reward.at[1].set(1).at[3].set(1),
                        base_reward.astype(jnp.float32),
                        base_reward.at[0].set(1).at[2].set(1),
                    ]
                ),
                mask=jnp.array(
                    [
                        base_reward_mask,
                        base_reward_mask.at[0]
                        .set(True)
                        .at[1]
                        .set(True)
                        .at[3]
                        .set(True),
                        base_reward_mask.at[0]
                        .set(True)
                        .at[1]
                        .set(True)
                        .at[3]
                        .set(True),
                        base_reward_mask.at[0].set(True).at[2].set(True),
                        base_reward_mask.at[0].set(True).at[2].set(True),
                    ]
                ),
                src_id=jnp.array(
                    [
                        base_reward,
                        base_reward,
                        base_reward.at[3].set(2),
                        base_reward.at[0].set(2),
                        base_reward.at[0].set(2).at[2].set(2),
                    ]
                ),
                dst_id=jnp.array(
                    [
                        base_reward,
                        base_reward.at[3].set(2),
                        base_reward.at[0].set(2).at[1].set(3).at[3].set(3),
                        base_reward.at[0].set(2).at[2].set(2),
                        base_reward.at[0].set(3).at[2].set(3),
                    ]
                ),
            ),
        )

    def init_hrm(self):
        hrm = ops.init_hrm(
            root_id=0,
            max_num_rms=4,
            max_num_states=4,
            max_num_edges=1,
            max_num_literals=42,
        )
        ops.load(hrm, PROJECT_DIR / "data/xminigrid_multilevel.yaml")
        return hrm
