"""
asymbench.gnn.architectures.gcn
================================
Graph Convolutional Network (GCN) for reaction property regression.

Based on Kipf & Welling (2017) — ``GCNConv`` from PyG.

.. note::
   GCN does not use edge features during message passing.  If your
   featurisation includes bond-level features that matter for your task,
   consider ``ReactionGAT`` or ``ReactionGIN`` instead.

YAML configuration example
---------------------------
::

    type: gnn
    params:
      architecture: gcn
      hidden_dim: 64
      num_layers: 3
      readout_layers: 2
      pooling: mean        # mean | add | max | mean_max
      dropout: 0.0
      improved: false      # GCN-specific: add self-loops with weight 2
"""

from __future__ import annotations

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from asymbench.gnn.base import BaseReactionGNN


class ReactionGCN(BaseReactionGNN):
    """GCN for reaction property regression on the merged reaction graph.

    Parameters
    ----------
    node_in_dim:
        Number of input node features.
    hidden_dim:
        Width of all hidden layers.
    num_layers:
        Number of ``GCNConv`` layers.
    pooling:
        Graph-level pooling — ``"mean"``, ``"add"``, ``"max"``,
        or ``"mean_max"``.
    readout_layers:
        Number of MLP layers in the readout head (including output layer).
    dropout:
        Dropout probability (0.0 = disabled).
    improved:
        Use improved GCN normalisation (self-loop weight = 2 instead of 1).
        See Kipf & Welling (2017) for details.
    """

    ARCH_NAME = "gcn"

    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        pooling: str = "mean",
        readout_layers: int = 2,
        dropout: float = 0.0,
        improved: bool = False,
        **kwargs,  # absorb unknown YAML params gracefully
    ) -> None:
        super().__init__(
            node_in_dim=node_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            pooling=pooling,
            readout_layers=readout_layers,
            dropout=dropout,
        )
        self.improved = improved

        dims = [node_in_dim] + [hidden_dim] * num_layers
        self.conv_layers = nn.ModuleList(
            [
                GCNConv(dims[i], dims[i + 1], improved=improved)
                for i in range(num_layers)
            ]
        )
        self.norm_layers = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        )

        self.make_readout_layers()

    # GCNConv does not accept edge_attr — override to drop it
    def get_graph_embedding(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None,
        batch: Tensor,
    ) -> Tensor:
        for conv, bn in zip(self.conv_layers, self.norm_layers):
            x = F.relu(bn(conv(x, edge_index)))
            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.pooling_fn(x, batch)
