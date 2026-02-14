import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


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
