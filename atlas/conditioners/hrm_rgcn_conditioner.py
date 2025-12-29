from typing import Optional, Tuple, Type

import chex
from chex import dataclass
from flax import linen as nn
import jax
import jax.numpy as jnp

from .gnn.rgcn import RGCN, RGCNConfig, RGCNGraph
from .types import Conditioner, ConditionerState, ConditionerOutput
from ..envs.common.labeling_function import LabelingFunction
from ..envs.common.literal_embedder import LiteralEmbedder
from ..hrm.ops import (
    get_max_num_literals, get_max_num_machines, get_max_num_states_per_machine,
    get_max_num_edges_per_state_pair
)
from ..hrm.types import HRM, HRMState


@dataclass
class RGCNConditionerState(ConditionerState):
    rng: chex.PRNGKey


@dataclass
class RGCNHRMConditionerConfig:
    cache: bool                               # Whether to cache the embeddings for the nodes along
                                              #  the input sequence (i.e. assumes that the first HRM
                                              #  in the sequence persists across the sequence, which
                                              #  is not true in the DR runner, for example)
    cond_aggregation: str                     # How to constitute the conditioning vector
    d_edge_feat: int                          # Dimensionality of the edge features
    node_init: str                            # Initialization strategy for nodes
    rgcn_config: RGCNConfig                   # The configuration for the R-GCN layers
    use_interrelations: bool                  # Whether to consider the relations between RMs
    d_node_init_feat: Optional[int] = None    # Initial dimensionality of the node features
                                              #  (only for some init strategies)


class _RGCNHRMEmbedder(nn.Module):
    literal_embed: LiteralEmbedder  # Module for embedding the literals
    config: RGCNHRMConditionerConfig

    def setup(self) -> None:
        # Layer for embedding edges, which result from the aggregation of
        # literal embeddings. The no bias helps keep masked embeddings to 0s.
        self._edge_embed = nn.Dense(features=self.config.d_edge_feat, use_bias=False)

        # The R-GCN itself
        self._rgcn = RGCN(
            config=self.config.rgcn_config,
            num_relations=3 if self.config.use_interrelations else 1
        )

    def __call__(self, c_state: RGCNConditionerState, hrm: HRM, hrm_state: HRMState):
        if self.config.cache:
            # Assume that the HRM is not going to check throughout the input sequence
            hrm_cached = jax.tree_util.tree_map(lambda x: x[0], hrm)

            # Initialize the graph and pass it to the R-GCN
            out_graph = self._rgcn(self._get_init_graph(c_state.rng, hrm_cached))

            # Return the conditioning output
            c_out = jax.vmap(
                lambda _hrm_state: ConditionerOutput(
                    conditioning_vector=self._get_conditioning_vector(out_graph, hrm_cached, _hrm_state)
                )
            )(hrm_state)
        else:
            # Analogous to the above but vmapping over HRMs too
            # The random number generator key is shared, but we could split it too if needed
            c_out = jax.vmap(
                lambda _hrm, _hrm_state: ConditionerOutput(
                    conditioning_vector=self._get_conditioning_vector(
                        self._rgcn(self._get_init_graph(c_state.rng, _hrm)),
                        _hrm,
                        _hrm_state
                    )
                )
            )(hrm, hrm_state)

        return RGCNConditionerState(rng=jax.random.split(c_state.rng)[0]), c_out

    def _get_init_graph(self, rng: chex.PRNGKey, hrm: HRM) -> RGCNGraph:
        senders, receivers = self._get_adjacency_matrix(hrm)
        edge_features = self._get_edge_features(hrm, senders, receivers)
        return RGCNGraph(
            nodes=self._get_init_node_features(rng, hrm, receivers, edge_features),
            edges=edge_features,
            senders=senders,
            receivers=receivers,
        )

    def _get_init_node_features(
        self, rng: chex.PRNGKey, hrm: HRM, receivers: chex.Array, edge_features: chex.Array
    ) -> chex.Array:
        max_num_nodes = get_max_num_machines(hrm) * get_max_num_states_per_machine(hrm)
        if self.config.node_init == "empty":
            # Returns an empty vector of features. It should be used only when
            # the R-GCN layers append the edge features before each transformation.
            return jnp.zeros((max_num_nodes, 0))
        elif self.config.node_init == "edge_features":
            # The node features are an aggregation of the incoming edge features
            # (for relation 0, corresponding to the standard edges in the RM with
            # reversed direction). The resulting features are normalized by the
            # incoming degree.
            node_features = jax.ops.segment_sum(edge_features[0], receivers[0], max_num_nodes)
            degrees = jax.ops.segment_sum(jnp.ones_like(receivers[0], dtype=jnp.float32), receivers[0], max_num_nodes)
            return node_features * jax.lax.reciprocal(jnp.maximum(degrees, 1.0))[:, None]
        elif self.config.node_init == "random":
            # The node features are initialized with a normal distribution with
            # 0 mean and variance 0.1.
            # Sources:
            # - "Graph Normalizing Flows" (NeurIPS 2019).
            # - "E(n) Equivariant Graph Neural Networks" (ICML 2021).
            return 0.1 * jax.random.normal(rng, (max_num_nodes, self.config.d_node_init_feat))
        elif self.config.node_init == "zeros":
            # The node features are initialized as vectors with just 0s.
            return jnp.zeros((max_num_nodes, self.config.d_node_init_feat))

    def _get_edge_features(self, hrm: HRM, senders: chex.Array, receivers: chex.Array) -> chex.Array:
        edge_feats = [self._get_formula_edge_features(hrm, senders, receivers)]
        if self.config.use_interrelations:
            max_num_nodes = get_max_num_machines(hrm) * (get_max_num_states_per_machine(hrm) ** 2)
            edge_feats.append(jnp.zeros((max_num_nodes, self.config.d_edge_feat)))  # relation 1
            edge_feats.append(jnp.zeros((max_num_nodes, self.config.d_edge_feat)))  # relation 2
        return jnp.stack(edge_feats)

    def _get_formula_edge_features(self, hrm: HRM, senders: chex.Array, receivers: chex.Array) -> chex.Array:
        # These features are obtained from the edges for relation 0
        idx = (
            receivers[0] // get_max_num_states_per_machine(hrm),  # rm id
            receivers[0] % get_max_num_states_per_machine(hrm),   # src local id
            senders[0] % get_max_num_states_per_machine(hrm),     # dst local id
            0                                                     # assume one edge
        )

        # The formulas whose literals are embedded
        formulas = hrm.formulas[idx]

        # To embed edges with no literals, we enforce the number of
        # literals to be 1 for these cases (there must be a call for
        # this to happen)
        call_mask = hrm.calls[idx] > -1
        num_literals = jnp.clip(hrm.num_literals[idx], min=1) * call_mask

        # Embed the literals, mask those that are unused, and aggregate (sum)
        # them. Then pass them through a dense layer.
        embedded_literals = self.literal_embed(formulas)

        literal_mask = jnp.arange(get_max_num_literals(hrm))[None, :] < num_literals[:, None]
        edge_features = self._edge_embed(
            jnp.sum(literal_mask[..., None] * embedded_literals, axis=1)
        )

        return edge_features

    def _get_adjacency_matrix(self, hrm: HRM) -> Tuple[chex.Array, chex.Array]:
        max_num_rms = get_max_num_machines(hrm)
        max_num_states = get_max_num_states_per_machine(hrm)
        max_num_edges = get_max_num_edges_per_state_pair(hrm)
        max_num_nodes = max_num_rms * max_num_states

        assert max_num_edges == 1

        # Base source and destination states in the global graph (i.e.
        # they have ids between 0 and max_num_nodes - 1).
        # These represent the ids of the states in the `calls` matrix
        base_src = jnp.arange(max_num_nodes).repeat(max_num_states)
        base_dst = (
            jnp.tile(jnp.arange(max_num_states), max_num_nodes) +
            jnp.repeat(max_num_states * jnp.arange(max_num_rms), max_num_states ** 2)
        )
        calls = hrm.calls.ravel()

        # The sources and destinations of the edges
        # Note that the orientation of the edges is redefined later (dst_0 -> src_0, ...)
        senders = []
        receivers = []

        # Relation 0 (within machines)
        #   Mask the valid calls and place a -1 for the non-existing edges
        valid_calls_mask = calls > -1
        src_0 = valid_calls_mask * base_src - jnp.logical_not(valid_calls_mask)
        dst_0 = valid_calls_mask * base_dst - jnp.logical_not(valid_calls_mask)

        senders.append(dst_0)
        receivers.append(src_0)

        if self.config.use_interrelations:
            # These relations are intended to be used across machines.
            # Our core experiments focused on HRMs with a single machine,
            # and we didn't iterate these for long. However, we have decided
            # to keep them if they serve as a starting point for future research.
            valid_nonleaf_calls_mask = jnp.logical_and(calls > -1, calls < max_num_rms)

            # Relation 1 captures connections between the state initiating the call
            # and the initial state in the called machine.
            init_states_called_rm = max_num_states * calls
            src_1 = valid_nonleaf_calls_mask * base_src - jnp.logical_not(valid_nonleaf_calls_mask)
            dst_1 = valid_nonleaf_calls_mask * init_states_called_rm - jnp.logical_not(valid_nonleaf_calls_mask)

            senders.append(dst_1)
            receivers.append(src_1)

            # Relation 2 captures connections between the accepting state of a called
            # machine and the state in the calling machine to which control is returned.
            acc_states_called_rm = max_num_states * calls + max_num_states - 1
            src_2 = valid_nonleaf_calls_mask * base_dst - jnp.logical_not(valid_nonleaf_calls_mask)
            dst_2 = valid_nonleaf_calls_mask * acc_states_called_rm - jnp.logical_not(valid_nonleaf_calls_mask)

            senders.append(src_2)
            receivers.append(dst_2)

        return jnp.stack(senders), jnp.stack(receivers)

    def _get_conditioning_vector(self, graph: RGCNGraph, hrm: HRM, hrm_state: HRMState) -> chex.Array:
        state_feats = graph.nodes[
            hrm_state.rm_id * get_max_num_states_per_machine(hrm) + hrm_state.state_id
        ]

        if self.config.cond_aggregation == "state":
            # Output is determined by the `hrm_state` only
            return state_feats
        elif self.config.cond_aggregation == "all":
            # Output is determined by an aggregation of the node features
            # concatenated with the feature corresponding to the `hrm_state`
            degree_mask_fn = lambda x: jax.ops.segment_sum(
                jnp.ones_like(graph.senders[0]), x, get_max_num_machines(hrm) * get_max_num_states_per_machine(hrm)
            ) > 0
            degree_mask = jnp.logical_or(
                degree_mask_fn(graph.senders[0]), degree_mask_fn(graph.receivers[0])
            )
            graph_feats = jnp.sum(graph.nodes * degree_mask[:, None], axis=0) / jnp.sum(degree_mask)
            return jnp.concat((graph_feats, state_feats))


class RGCNHRMConditioner(Conditioner):
    """
    Produces a conditioning on the HRM using a Relational Graph Convolutional
    Network (R-GCN).
    """
    config: RGCNHRMConditionerConfig
    literal_embedder: LiteralEmbedder

    def setup(self) -> None:
        # To perform a `vmap` across the input HRMs and HRM states
        self._batched_embedder = nn.vmap(
            _RGCNHRMEmbedder,
            variable_axes={"params": None},
            split_rngs={"params": False}
        )(self.literal_embedder, self.config)

    def __call__(
        self,
        c_state: RGCNConditionerState,
        hrm: HRM,
        hrm_state: HRMState,
        *args,
        **kwargs
    ) -> Tuple[ConditionerState, ConditionerOutput]:
        return self._batched_embedder(c_state, hrm, hrm_state)

    def initialize_state(self, batch_size: int, rng: chex.PRNGKey, **kwargs) -> ConditionerState:
        """
        Initializes the state of the conditioner, which is mandatory
        before making a call to the module.
        """
        return RGCNConditionerState(rng=jax.random.split(rng, batch_size))

    @staticmethod
    def init_conditioner(
        name: str, 
        config: RGCNHRMConditionerConfig, 
        literal_embedder: Type[LiteralEmbedder], 
        label_fn: LabelingFunction
    ):
        """
        Auxiliary method to initialize the module. Added to prevent the construction
        of the literal embedder from being jitted.
        """
        return RGCNHRMConditioner(
            name=name,
            config=config,
            literal_embedder=literal_embedder(label_fn),
            label_fn=label_fn,
        )
