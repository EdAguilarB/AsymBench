"""
asymbench.gnn.architectures.gin
================================
Graph Isomorphism Network with Edge features (GIN-E) for reaction property
regression.

Uses ``GINEConv`` from PyG (Hu et al., 2020), which extends GIN to
incorporate edge features.  GIN-type networks are maximally expressive among
1-WL-equivalent architectures and often outperform GCN/GAT on molecular
property prediction benchmarks.

YAML configuration example
---------------------------
::

    type: gnn
    params:
      architecture: gin
      hidden_dim: 64
      num_layers: 3
      readout_layers: 2
      pooling: mean        # mean | add | max | mean_max
      dropout: 0.0
      train_eps: false     # GIN-specific: learn epsilon per layer
"""

from __future__ import annotations

import torch.nn as nn
from torch_geometric.nn import GINEConv

from asymbench.gnn.base import BaseReactionGNN, _resolve_activation
from asymbench.gnn.featurizer import EDGE_FEAT_DIM


def _gin_mlp(in_dim: int, out_dim: int, act: nn.Module) -> nn.Sequential:
    """Two-layer MLP used as the aggregation function inside each GINEConv."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        type(act)(),  # fresh instance of the chosen activation
        nn.Linear(out_dim, out_dim),
    )


class ReactionGIN(BaseReactionGNN):
    """GIN-E for reaction property regression on the merged reaction graph.

    Parameters
    ----------
    node_in_dim:
        Number of input node features.
    hidden_dim:
        Width of all hidden layers.
    num_layers:
        Number of ``GINEConv`` layers.
    pooling:
        Graph-level pooling — ``"mean"``, ``"add"``, ``"max"``,
        or ``"mean_max"``.
    readout_layers:
        Number of MLP layers in the readout head (including output layer).
    dropout:
        Dropout probability applied after each layer.
    train_eps:
        If ``True``, the epsilon parameter of each GINEConv is learned.
    edge_in_dim:
        Dimensionality of edge features.  Defaults to the package-level
        ``EDGE_FEAT_DIM`` so you rarely need to set this manually.
    activation:
        Non-linearity used inside the GINEConv MLP, after each conv layer,
        and in the readout MLP.  ``"relu"`` (default), ``"leaky_relu"``,
        ``"elu"``, ``"silu"``, ``"gelu"``, ``"tanh"``.
    """

    ARCH_NAME = "gin"

    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        pooling: str = "mean",
        readout_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "relu",
        train_eps: bool = False,
        edge_in_dim: int = EDGE_FEAT_DIM,
        **kwargs,
    ) -> None:
        super().__init__(
            node_in_dim=node_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            pooling=pooling,
            readout_layers=readout_layers,
            dropout=dropout,
            activation=activation,
        )
        self.train_eps = train_eps
        self.edge_in_dim = edge_in_dim

        in_dims = [node_in_dim] + [hidden_dim] * (num_layers - 1)
        self.conv_layers = nn.ModuleList(
            [
                GINEConv(
                    nn=_gin_mlp(in_dim, hidden_dim, self.act),
                    train_eps=train_eps,
                    edge_dim=edge_in_dim,
                )
                for in_dim in in_dims
            ]
        )
        # GINEConv already includes BatchNorm inside its MLP; norm_layers here
        # are kept as Identity so the base get_graph_embedding loop works
        # unchanged without any extra normalisation.
        self.norm_layers = nn.ModuleList(
            [nn.Identity() for _ in range(num_layers)]
        )

        self.make_readout_layers()

    # Uses BaseReactionGNN.get_graph_embedding() which calls conv(x, edge_index, edge_attr)
    # — compatible with GINEConv's forward signature.
