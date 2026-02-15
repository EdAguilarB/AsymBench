import torch
from rdkit import Chem
from torch_geometric.data import Data


def atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetHybridization(),
        atom.GetIsAromatic(),
    ]


def bond_features(bond):
    return [bond.GetBondTypeAsDouble(), bond.GetIsConjugated(), bond.IsInRing()]


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_feats = []
    for atom in mol.GetAtoms():
        node_feats.append(atom_features(atom))

    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bf = bond_features(bond)

        edge_index.append([i, j])
        edge_index.append([j, i])

        edge_attr.append(bf)
        edge_attr.append(bf)

    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    x = torch.tensor(node_feats, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
