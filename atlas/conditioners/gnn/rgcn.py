from typing import List

import chex
from chex import dataclass
from flax import linen as nn
import jax
import jax.numpy as jnp


@dataclass
class RGCNConfig:
    d_hidden: int               # Number of output features after each layer
    d_out: int                  # Number of output node features
    num_layers: int             # Number of R-GCN layers
    use_edge_feats: List[bool]  # Whether to concatenate the edge feats with the node
                                #  feats before the transformation
                                #  E.g. https://arxiv.org/pdf/1905.12265
    use_layer_norm: bool        # Whether to use layer normalization
    use_node_proj: bool         # Whether to produce a projection of the node
                                #  that is later aggregated with the output
                                #  of the convolutions (plays the role of self-relations)


@dataclass
class RGCNGraph:
    nodes: chex.Array      # [max_num_rms * max_num_states, num_features]
    edges: chex.Array      # [num_relations, max_num_rms * max_num_states * max_num_states, num_features]
    senders: chex.Array    # [num_relations, max_num_rms * max_num_states * max_num_states]
    receivers: chex.Array  # [num_relations, max_num_rms * max_num_states * max_num_states]


class RGCNLayer(nn.Module):
    config: RGCNConfig
    num_relations: int
    num_nodes: int

    def setup(self) -> None:
        # The transformation for the nodes (if `use_node_proj` is enabled)
        self._node_proj = nn.Dense(
            features=self.config.d_hidden,
            use_bias=False,  # keeps the masking to 0s for unused edges in the input
            kernel_init=nn.initializers.he_normal(),
        )

        # The transformation for each relation
        self._rel_projs = [
            nn.Dense(
                features=self.config.d_hidden,
                use_bias=False,  # keeps the masking to 0s for unused edges in the input
                kernel_init=nn.initializers.he_normal(),
            )
            for _ in range(self.num_relations)
        ]

        self._layer_norm = nn.LayerNorm()

    def __call__(self, graph: RGCNGraph):
        node_feats = self._get_relational_node_feats(graph)

        if self.config.use_node_proj:
            # Known as `root_weight` in Pytorch Geometric:
            # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.RGCNConv.html
            node_feats = self._node_proj(graph.nodes) + node_feats

        if self.config.use_layer_norm:
            node_feats = self._layer_norm(node_feats)

        return graph.replace(
            nodes=jax.nn.relu(node_feats)
        )

    def _get_relational_node_feats(self, graph: RGCNGraph):
        node_feats = jnp.zeros_like(self.config.d_hidden)

        for r_id in range(self.num_relations):
            # Get senders and receivers
            senders = graph.senders[r_id]
            receivers = graph.receivers[r_id]

            # Pass the features through the relation-related MLP
            r_feats = graph.nodes[senders]

            # Build the relational features
            if self.config.use_edge_feats[r_id]:
                r_feats = jnp.concat((r_feats, graph.edges[r_id]), axis=1)
            r_feats = self._rel_projs[r_id](r_feats)

            # Aggregate the features
            r_feats = jax.ops.segment_sum(r_feats, receivers, self.num_nodes)

            # Calculate normalization value and normalize
            # The original GCN uses the product of the sqrts
            # of sender and receiver's degree, but R-GCN seems
            # to only use the receiver's degree w.o. sqrt
            receiver_degree = jax.ops.segment_sum(
                jnp.ones_like(senders), receivers, self.num_nodes,
            )
            r_feats = r_feats * jax.lax.reciprocal(jnp.maximum(receiver_degree, 1.0))[:, None]

            # Aggregate the features
            node_feats += r_feats

        return node_feats


class RGCN(nn.Module):
    config: RGCNConfig
    num_relations: int

    @nn.compact
    def __call__(self, graph: RGCNGraph):
        # Pass the graph through several conv layers
        for _ in range(self.config.num_layers):
            graph = RGCNLayer(
                self.config, self.num_relations, graph.nodes.shape[0]
            )(graph)

        # Perform a final projection of the features and return the graph
        node_feats = nn.Dense(
            features=self.config.d_out,
            use_bias=False,
            kernel_init=nn.initializers.he_normal(),
        )(graph.nodes)

        return graph.replace(nodes=node_feats)
