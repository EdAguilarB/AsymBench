import torch
from torch_geometric.data import Dataset

from asymbench.gnn.featurizer import smiles_to_graph


class ReactionDataset(Dataset):
    def __init__(self, df, smiles_cols, target):
        self.df = df.reset_index(drop=True)
        self.smiles_cols = smiles_cols
        self.target = target

    def len(self):
        return len(self.df)

    def get(self, idx):
        row = self.df.iloc[idx]

        graphs = []
        for col in self.smiles_cols:
            g = smiles_to_graph(row[col])
            if g is None:
                return None
            graphs.append(g)

        y = torch.tensor([row[self.target]], dtype=torch.float)

        return graphs, y
