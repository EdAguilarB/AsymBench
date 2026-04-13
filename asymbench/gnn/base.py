"""
asymbench.gnn.base
==================
Base GNN class shared across all reaction GNN architectures.

``BaseReactionGNN`` implements the full forward pass, graph embedding
extraction, pooling, and MLP readout for regression.  Architecture-specific
models in ``asymbench.gnn.architectures`` inherit from this class and only
define their own convolutional layers, plus an optional override of
``get_graph_embedding`` when the conv call signature differs (e.g. GCN does
not use edge features).

Supported pooling strategies
-----------------------------
``"mean"``
    Global mean pooling — recommended default.
``"add"``
    Global add (sum) pooling.
``"max"``
    Global max pooling.
``"mean_max"``
    Concatenation of global mean and max pool; doubles the embedding
    dimension fed to the readout MLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

# ---------------------------------------------------------------------------
# Activation helper
# ---------------------------------------------------------------------------

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "silu": nn.SiLU,  # also known as Swish
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


def _resolve_activation(name: str) -> nn.Module:
    """Return an activation ``nn.Module`` from a string name.

    Parameters
    ----------
    name:
        Case-insensitive activation name.  Supported values:
        ``"relu"`` (default), ``"leaky_relu"``, ``"elu"``,
        ``"silu"`` / ``"swish"``, ``"gelu"``, ``"tanh"``.

    Raises
    ------
    ValueError
        If *name* is not recognised.
    """
    key = name.lower()
    if key == "swish":  # common alias for SiLU
        key = "silu"
    if key not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation {name!r}. "
            f"Choose from: {list(_ACTIVATIONS)}."
        )
    return _ACTIVATIONS[key]()


# ---------------------------------------------------------------------------
# Pooling helpers
# ---------------------------------------------------------------------------


def _mean_max_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1
    )


_POOLING_FNS: dict = {
    "mean": (global_mean_pool, 1),
    "add": (global_add_pool, 1),
    "max": (global_max_pool, 1),
    "mean_max": (_mean_max_pool, 2),
}


def _make_pooling_fn(pooling: str):
    """Return ``(pooling_callable, output_dim_multiplier)``."""
    key = pooling.lower()
    if key not in _POOLING_FNS:
        raise ValueError(
            f"Unknown pooling {pooling!r}. "
            f"Choose from: {list(_POOLING_FNS)}."
        )
    return _POOLING_FNS[key]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseReactionGNN(nn.Module):
    """
    Base GNN for reaction property regression.

    Implements the shared forward pass:
    message-passing → pooling → MLP readout.

    Subclasses must:
    1. Call ``super().__init__(...)`` with the shared parameters.
    2. Define ``self.conv_layers`` (``nn.ModuleList``) and
       ``self.norm_layers`` (``nn.ModuleList``) for their architecture.
    3. Call ``self.make_readout_layers()`` at the end of their ``__init__``.
    4. Override ``get_graph_embedding()`` if the conv call signature differs
       from the default (which passes ``edge_attr`` as a positional arg).

    Parameters
    ----------
    node_in_dim:
        Number of input node features.
    hidden_dim:
        Width of all hidden layers (conv and readout).
    num_layers:
        Number of graph convolutional layers (≥ 1).
    pooling:
        Graph-level pooling strategy.  See module docstring.
    readout_layers:
        Total number of MLP layers including the output layer.
        ``readout_layers=2`` → ``[Linear(hidden, hidden//2), ReLU, Linear(hidden//2, 1)]``.
    dropout:
        Dropout probability applied after each conv and readout activation.
        Set to ``0.0`` to disable.
    activation:
        Non-linearity applied after each conv layer and between readout
        layers.  Accepted values: ``"relu"`` (default), ``"leaky_relu"``,
        ``"elu"``, ``"silu"`` / ``"swish"``, ``"gelu"``, ``"tanh"``.
    """

    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        pooling: str = "mean",
        readout_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "relu",
        reaction_feature_dim: int = 0,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if readout_layers < 1:
            raise ValueError("readout_layers must be >= 1")

        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.readout_layers = readout_layers
        self.dropout = dropout
        self.act = _resolve_activation(activation)
        self.reaction_feature_dim = reaction_feature_dim

        self.pooling_fn, pool_mult = _make_pooling_fn(pooling)
        self.graph_embedding_dim = hidden_dim * pool_mult
        # Actual input to the readout MLP: graph embedding + scaled rxn features
        self.readout_input_dim = (
            self.graph_embedding_dim + reaction_feature_dim
        )

        # Subclasses define these before calling make_readout_layers()
        self.conv_layers: nn.ModuleList
        self.norm_layers: nn.ModuleList

    # ------------------------------------------------------------------
    # Layer construction
    # ------------------------------------------------------------------

    def make_readout_layers(self) -> None:
        """Build the MLP readout head.

        Must be called at the **end** of each subclass ``__init__``, after
        ``conv_layers`` and ``norm_layers`` have been assigned.

        Creates a sequential MLP that halves the dimension at each
        intermediate layer, ending with a single output neuron::

            graph_embedding_dim → hidden//2 → … → 1
        """
        layers: list[nn.Module] = []
        dim = self.readout_input_dim

        for _ in range(self.readout_layers - 1):
            out_dim = max(dim // 2, 1)
            layers.append(nn.Linear(dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(type(self.act)())  # fresh instance, same class
            if self.dropout > 0.0:
                layers.append(nn.Dropout(p=self.dropout))
            dim = out_dim

        layers.append(nn.Linear(dim, 1))
        self.readout = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def get_graph_embedding(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Run message passing and pooling to produce a graph-level embedding.

        Default implementation calls ``conv(x, edge_index, edge_attr)`` which
        is compatible with any PyG layer that accepts edge features as the
        third positional argument (e.g. ``GATv2Conv``, ``GINEConv``).

        **Override this method** in subclasses whose conv layers have a
        different call signature (e.g. ``GCNConv`` ignores edge features
        and should be called as ``conv(x, edge_index)``).

        Parameters
        ----------
        x:
            Node feature matrix, shape ``(N, node_in_dim)``.
        edge_index:
            Edge connectivity in COO format, shape ``(2, E)``.
        edge_attr:
            Edge feature matrix, shape ``(E, edge_in_dim)``, or ``None``.
        batch:
            Batch vector mapping each node to its graph, shape ``(N,)``.

        Returns
        -------
        torch.Tensor
            Graph-level embedding, shape ``(batch_size, graph_embedding_dim)``.
        """
        for conv, bn in zip(self.conv_layers, self.norm_layers):
            x = self.act(bn(conv(x, edge_index, edge_attr)))
            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.pooling_fn(x, batch)

    def forward(self, data: Data) -> torch.Tensor:
        """Full forward pass: message passing → pooling → MLP readout.

        Parameters
        ----------
        data:
            A (possibly batched) PyG ``Data`` object with attributes
            ``x``, ``edge_index``, and optionally ``edge_attr`` and ``batch``.

        Returns
        -------
        torch.Tensor
            Scalar prediction per graph, shape ``(batch_size,)``.
        """
        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is not None:
            edge_attr = edge_attr.float()

        # Fallback for unbatched single graphs
        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        embedding = self.get_graph_embedding(x, edge_index, edge_attr, batch)

        rxn = getattr(data, "reaction_features", None)
        if rxn is not None and self.reaction_feature_dim > 0:
            rxn = rxn.float().to(embedding.device)
            embedding = torch.cat([embedding, rxn], dim=1)

        return self.readout(embedding).squeeze(-1)
