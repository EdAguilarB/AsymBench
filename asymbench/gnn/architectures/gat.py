"""
asymbench.gnn.architectures.gat
================================
Graph Attention Network v2 (GATv2) for reaction property regression.

Uses ``GATv2Conv`` from PyG (Brody et al., 2022), which fixes the static
attention problem of the original GAT by computing attention coefficients
after concatenating the source and target node features.

Edge features are incorporated natively — GAT is a good choice when bond
type, stereo chemistry, or ring membership should influence attention weights.

YAML configuration example
---------------------------
::

    type: gnn
    params:
      architecture: gat
      hidden_dim: 64
      num_layers: 3
      readout_layers: 2
      pooling: mean        # mean | add | max | mean_max
      dropout: 0.1
      num_heads: 4         # GAT-specific: attention heads per layer
"""

from __future__ import annotations

import torch.nn as nn
from torch_geometric.nn import GATv2Conv

from asymbench.gnn.base import BaseReactionGNN
from asymbench.gnn.featurizer import EDGE_FEAT_DIM


class ReactionGAT(BaseReactionGNN):
    """GATv2 for reaction property regression on the merged reaction graph.

    Parameters
    ----------
    node_in_dim:
        Number of input node features.
    hidden_dim:
        Output dimensionality of each attention head aggregation.
    num_layers:
        Number of ``GATv2Conv`` layers.
    pooling:
        Graph-level pooling — ``"mean"``, ``"add"``, ``"max"``,
        or ``"mean_max"``.
    readout_layers:
        Number of MLP layers in the readout head (including output layer).
    dropout:
        Dropout probability applied inside attention and after each layer.
    num_heads:
        Number of attention heads.  Outputs are averaged across heads
        (``concat=False``) so the embedding width stays ``hidden_dim``.
    edge_in_dim:
        Dimensionality of edge features.  Defaults to the package-level
        ``EDGE_FEAT_DIM`` so you rarely need to set this manually.
    activation:
        Non-linearity after each conv layer and in the readout MLP.
        ``"relu"`` (default), ``"leaky_relu"``, ``"elu"``, ``"silu"``,
        ``"gelu"``, ``"tanh"``.
    """

    ARCH_NAME = "gat"

    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        pooling: str = "mean",
        readout_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
        num_heads: int = 4,
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
        self.num_heads = num_heads
        self.edge_in_dim = edge_in_dim

        in_channels = [node_in_dim] + [hidden_dim] * (num_layers - 1)
        self.conv_layers = nn.ModuleList(
            [
                GATv2Conv(
                    in_channels=in_ch,
                    out_channels=hidden_dim,
                    edge_dim=edge_in_dim,
                    heads=num_heads,
                    concat=False,  # average heads → output stays hidden_dim
                    dropout=dropout,
                )
                for in_ch in in_channels
            ]
        )
        self.norm_layers = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        )

        self.make_readout_layers()

    # Uses BaseReactionGNN.get_graph_embedding() which calls conv(x, edge_index, edge_attr)
    # — compatible with GATv2Conv's forward signature.
