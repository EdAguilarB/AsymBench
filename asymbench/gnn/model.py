import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool


# ---------------------------------------------------------------------------
# Legacy multi-molecule model (kept for backwards compatibility)
# ---------------------------------------------------------------------------

class MoleculeGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return global_mean_pool(x, batch)


class ReactionGNN(nn.Module):
    def __init__(self, num_mols, in_dim, hidden_dim=64):
        super().__init__()

        self.mol_gnns = nn.ModuleList(
            [MoleculeGNN(in_dim, hidden_dim) for _ in range(num_mols)]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * num_mols, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch_list):
        embeddings = []

        for gnn, data in zip(self.mol_gnns, batch_list):
            embeddings.append(gnn(data))

        x = torch.cat(embeddings, dim=1)
        return self.head(x)


# ---------------------------------------------------------------------------
# Current model: operates on the merged reaction graph
# ---------------------------------------------------------------------------

class ReactionGCN(nn.Module):
    """GCN for reaction prediction on the merged reaction graph.

    The reaction is represented as a single disconnected molecular graph
    (see :class:`~asymbench.gnn.reaction_graph.ReactionGraphBuilder`).
    All participant molecules (substrate, ligand, solvent, …) share the
    same set of GCN layers; a global pooling over all nodes yields a
    fixed-size reaction embedding that is fed to the MLP prediction head.

    Parameters
    ----------
    node_in_dim:
        Number of input node features (use
        :data:`~asymbench.gnn.featurizer.NODE_FEAT_DIM`).
    hidden_dim:
        Width of every hidden layer.
    num_layers:
        Number of GCNConv layers (minimum 1).
    pooling:
        Graph-level pooling strategy: ``"mean"`` or ``"add"``.
    """

    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        pooling: str = "mean",
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        _pool_fns = {"mean": global_mean_pool, "add": global_add_pool}
        if pooling not in _pool_fns:
            raise ValueError(
                f"Unknown pooling {pooling!r}. Choose from {list(_pool_fns)}."
            )

        dims = [node_in_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(
            [GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        )
        self.pool = _pool_fns[pooling]

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = bn(conv(x, edge_index).relu())
        x = self.pool(x, batch)
        return self.head(x).squeeze(-1)
