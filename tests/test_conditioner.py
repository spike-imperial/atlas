from abc import ABC, abstractmethod

import chex
import jax
import jax.numpy as jnp

from atlas.hrm import ops
from atlas.conditioners import (
    DummyConditioner,
    RGCNHRMConditioner,
    VanillaHRMConditioner,
)
from atlas.conditioners.hrm_rgcn_conditioner import RGCNHRMConditionerConfig, _RGCNHRMEmbedder
from atlas.conditioners.gnn.rgcn import RGCNConfig
from atlas.conditioners.types import Conditioner, ConditionerState
from atlas.envs.common.literal_embedder import BasicLiteralEmbedder


class _TestConditioner(ABC):
    LABELS = jnp.array(
        [
            [+1, -1, -1, -1, -1],
            [-1, +1, -1, -1, -1],
            [-1, -1, +1, -1, -1],
            [-1, -1, -1, +1, -1],
            [-1, -1, -1, -1, +1],
        ],
        dtype=jnp.int32,
    )

    def test_conditioner(self):
        conditioner = self._init_conditioner(alphabet_size=5)

        # Verify conditioner state
        init_c_state = conditioner.initialize_state(self.LABELS.shape[0], jax.random.PRNGKey(0))
        self._check_batch_dim_conditioner_state(init_c_state)

        # Get an HRM and add the batch and sequence dimensions
        hrms, hrm_states = self._init_hrm_batch()

        # Initialize the conditioner parameters and check their validity
        c_params = conditioner.init(
            jax.random.PRNGKey(0),
            init_c_state,
            hrms,
            hrm_states,
        )
        self._check_conditioner_params(c_params)

        # Apply the conditioner
        c_state, c_out = conditioner.apply(
            c_params,
            init_c_state,
            hrms,
            hrm_states,
        )
        self._check_conditioner_state(c_state)
        self._check_conditioning_vector(c_out.conditioning_vector)

    def _init_hrm_batch(self):
        def _f(label: chex.Array):
            hrm = self._get_hrm()
            hrm_state, _ = ops.step(hrm, ops.get_initial_hrm_state(hrm), label)
            return (
                jax.tree_util.tree_map(lambda x: x[None, ...], hrm),
                jax.tree_util.tree_map(lambda x: x[None, ...], hrm_state),
            )

        return jax.vmap(_f)(self.LABELS)

    def _get_hrm(self):
        hrm = ops.init_hrm(
            root_id=0,
            max_num_rms=1,
            max_num_states=6,
            max_num_edges=1,
            max_num_literals=5,
        )

        for dst_id in range(1, 5):
            ops.add_leaf_call(hrm, 0, 0, dst_id, 0)
            ops.add_condition(hrm, 0, 0, dst_id, 0, dst_id, True)
            ops.add_reward(hrm, 0, 0, dst_id, 0.0)

            ops.add_leaf_call(hrm, 0, dst_id, 5, 0)
            ops.add_condition(hrm, 0, dst_id, 5, 0, 0, True)
            ops.add_reward(hrm, 0, dst_id, 5, 1.0)

        return hrm

    def _check_batch_dim_conditioner_state(self, c_state: ConditionerState):
        """
        Checks whether the leading dimension corresponds to the batch size.
        """
        if len(jax.tree_util.tree_leaves(c_state)) == 0:
            return  # if c_state does not contain any fields, there is nothing to check

        assert jnp.all(
            jnp.array(
                jax.tree_leaves(
                    jax.tree_map(lambda x: x.shape[0] == self.LABELS.shape[0], c_state)
                )
            )
        ), f"The leading dimension does not correspond to the batch size ({self.LABELS.shape[0]})"

    @abstractmethod
    def _init_conditioner(self, **kwargs) -> Conditioner:
        raise NotImplementedError

    @abstractmethod
    def _check_conditioner_params(self, conditioner_params):
        raise NotImplementedError

    @abstractmethod
    def _check_conditioner_state(self, c_state: ConditionerState):
        raise NotImplementedError

    @abstractmethod
    def _check_conditioning_vector(self, conditioning_vector):
        raise NotImplementedError


class TestVanillaConditioner(_TestConditioner):
    def _init_conditioner(self, **kwargs) -> Conditioner:
        return VanillaHRMConditioner(kwargs)

    def _check_conditioner_state(self, c_state: ConditionerState):
        assert (
            len(jax.tree_util.tree_leaves(c_state)) == 0
        ), "The conditioner state should not contain any fields"

    def _check_conditioner_params(self, conditioner_params):
        assert not conditioner_params, "There should not be any conditioner parameters"

    def _check_conditioning_vector(self, conditioning_vector: chex.Array):
        assert jnp.array_equal(
            conditioning_vector.squeeze(axis=1),
            jnp.array(
                [
                    [1, 1, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 1, 0, 0],
                    [1, 0, 0, 0, 0, 1, 0],
                ]
            ),
        )


class TestDummyConditioner(_TestConditioner):
    def _init_conditioner(self, **kwargs) -> Conditioner:
        return DummyConditioner(kwargs)

    def _check_conditioner_state(self, c_state: ConditionerState):
        assert (
            len(jax.tree_util.tree_leaves(c_state)) == 0
        ), "The conditioner state should not contain any fields"

    def _check_conditioner_params(self, conditioner_params):
        assert not conditioner_params, "There should not be any conditioner parameters"

    def _check_conditioning_vector(self, conditioning_vector: chex.Array):
        assert jnp.array_equal(
            conditioning_vector,
            jnp.zeros((self.LABELS.shape[0], 1, 0)),
        )


class TestRGCNConditioner(_TestConditioner):
    D_FEAT_SIZE = 64
    CONFIG = RGCNHRMConditionerConfig(
        cache=True,
        cond_aggregation="state",
        d_edge_feat=D_FEAT_SIZE,
        node_init="zeros",
        rgcn_config=RGCNConfig(
            d_hidden=D_FEAT_SIZE,
            d_out=D_FEAT_SIZE,
            num_layers=2,
            use_edge_feats=[True, False, False],
            use_layer_norm=False,
            use_node_proj=True,
        ),
        use_interrelations=True,
        d_node_init_feat=D_FEAT_SIZE,
    )

    def _init_conditioner(self, **kwargs) -> Conditioner:
        literal_embedder = BasicLiteralEmbedder(**kwargs, d_feat=self.D_FEAT_SIZE)
        return RGCNHRMConditioner(config=self.CONFIG, literal_embedder=literal_embedder, label_fn=None)

    def _check_conditioner_params(self, conditioner_params):
        assert conditioner_params, "The conditioner parameters should not be empty"

    def _check_conditioner_state(self, c_state: ConditionerState):
        assert (
            len(jax.tree_util.tree_leaves(c_state)) == 1
        ), "The conditioner state should contain a single field corresponding to an `rng`"

    def _check_conditioning_vector(self, conditioning_vector):
        assert conditioning_vector.shape == (self.LABELS.shape[0], 1, self.D_FEAT_SIZE)

    def test_adjacency_matrix(self):
        def _make_hrm():
            hrm = ops.init_hrm(
                root_id=0,
                max_num_rms=10,
                max_num_states=5,
                max_num_edges=1,
                max_num_literals=10,
            )

            # RM 0
            ops.add_call(hrm, rm_id=0, src_id=0, dst_id=1, edge_id=0, called_rm_id=1)
            ops.add_condition(hrm, rm_id=0, src_id=0, dst_id=1, edge_id=0, proposition=1, is_positive=False)

            ops.add_call(hrm, rm_id=0, src_id=0, dst_id=2, edge_id=0, called_rm_id=2)

            ops.add_call(hrm, rm_id=0, src_id=1, dst_id=3, edge_id=0, called_rm_id=2)

            ops.add_call(hrm, rm_id=0, src_id=2, dst_id=3, edge_id=0, called_rm_id=1)

            ops.add_leaf_call(hrm, rm_id=0, src_id=3, dst_id=4, edge_id=0)
            ops.add_condition(hrm, rm_id=0, src_id=3, dst_id=4, edge_id=0, proposition=2, is_positive=True)

            # RM 1
            ops.add_leaf_call(hrm, rm_id=1, src_id=0, dst_id=1, edge_id=0)
            ops.add_condition(hrm, rm_id=1, src_id=0, dst_id=1, edge_id=0, proposition=0, is_positive=True)

            ops.add_leaf_call(hrm, rm_id=1, src_id=1, dst_id=4, edge_id=0)
            ops.add_condition(hrm, rm_id=1, src_id=1, dst_id=4, edge_id=0, proposition=3, is_positive=True)

            # RM 2
            ops.add_leaf_call(hrm, rm_id=2, src_id=0, dst_id=1, edge_id=0)
            ops.add_condition(hrm, rm_id=2, src_id=0, dst_id=1, edge_id=0, proposition=1, is_positive=True)

            ops.add_leaf_call(hrm, rm_id=2, src_id=1, dst_id=4, edge_id=0)
            ops.add_condition(hrm, rm_id=2, src_id=1, dst_id=4, edge_id=0, proposition=3, is_positive=True)

            return hrm

        def _get_list(relation_id: int, senders: chex.Array, receivers: chex.Array):
            r_edges = []
            for s, r in zip(senders[relation_id], receivers[relation_id]):
                if s >= 0 and r >= 0:
                    r_edges.append((int(s), int(r)))
            r_edges.sort()
            return jnp.array(r_edges)

        embedder = _RGCNHRMEmbedder(config=self.CONFIG, literal_embed=BasicLiteralEmbedder(alphabet_size=4, d_feat=self.D_FEAT_SIZE))
        senders, receivers = embedder._get_adjacency_matrix(_make_hrm())

        # Relation 0
        assert jnp.array_equal(
            _get_list(0, senders, receivers),
            jnp.array([
                [1, 0], [2, 0], [3, 1], [3, 2], [4, 3], [6, 5], [9, 6], [11, 10], [14, 11]
            ])
        )

        # Relation 1
        assert jnp.array_equal(
            _get_list(1, senders, receivers),
            jnp.array([
                [5, 0], [5, 2], [10, 0], [10, 1]
            ])
        )

        # Relation 2
        assert jnp.array_equal(
            _get_list(2, senders, receivers),
            jnp.array([
                [1, 9], [2, 14], [3, 9], [3, 14]
            ])
        )
