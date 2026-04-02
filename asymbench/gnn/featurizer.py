from typing import List, Optional

import torch
from rdkit import Chem
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# Allowable sets for one-hot encoding
# ---------------------------------------------------------------------------

ATOM_ELEMENTS: List[str] = ["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"]

ATOM_DEGREES: List[int] = [0, 1, 2, 3, 4, 5, 6]

ATOM_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

ATOM_FORMAL_CHARGES: List[int] = [-2, -1, 0, 1, 2]

# CW / CCW — unspecified chirality encodes as all-zeros
ATOM_CHIRAL_TAGS = [
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

# Z / E — no stereo encodes as all-zeros
BOND_STEREO = [
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]

# Pre-computed feature dimensions (useful for model initialisation)
NODE_FEAT_DIM: int = (
    len(ATOM_ELEMENTS)       # element
    + len(ATOM_DEGREES)      # degree
    + len(ATOM_HYBRIDIZATIONS)  # hybridisation
    + len(ATOM_FORMAL_CHARGES)  # formal charge
    + len(ATOM_CHIRAL_TAGS)  # chirality
    + 1                      # is_aromatic
    + 1                      # is_in_ring
)

EDGE_FEAT_DIM: int = (
    len(BOND_TYPES)    # bond type
    + len(BOND_STEREO) # stereo
    + 1                # is_in_ring
    + 1                # is_conjugated
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _one_hot(value, allowable_set: list) -> List[int]:
    """Return a one-hot list aligned to *allowable_set*.
    Unknown values produce an all-zeros vector (treated as 'other/none').
    """
    return [int(value == item) for item in allowable_set]


# ---------------------------------------------------------------------------
# Per-atom and per-bond feature vectors
# ---------------------------------------------------------------------------

def atom_features(atom: Chem.rdchem.Atom) -> List[float]:
    """Return a flat float feature vector for a single RDKit atom."""
    feats: List[float] = []
    feats += _one_hot(atom.GetSymbol(), ATOM_ELEMENTS)
    feats += _one_hot(atom.GetDegree(), ATOM_DEGREES)
    feats += _one_hot(atom.GetHybridization(), ATOM_HYBRIDIZATIONS)
    feats += _one_hot(atom.GetFormalCharge(), ATOM_FORMAL_CHARGES)
    feats += _one_hot(atom.GetChiralTag(), ATOM_CHIRAL_TAGS)
    feats.append(int(atom.GetIsAromatic()))
    feats.append(int(atom.IsInRing()))
    return feats


def bond_features(bond: Chem.rdchem.Bond) -> List[float]:
    """Return a flat float feature vector for a single RDKit bond."""
    feats: List[float] = []
    feats += _one_hot(bond.GetBondType(), BOND_TYPES)
    feats += _one_hot(bond.GetStereo(), BOND_STEREO)
    feats.append(int(bond.IsInRing()))
    feats.append(int(bond.GetIsConjugated()))
    return feats


# ---------------------------------------------------------------------------
# Molecule → PyG graph
# ---------------------------------------------------------------------------

def mol_to_graph(mol: Chem.Mol) -> Data:
    """Convert an RDKit *Mol* object into a :class:`torch_geometric.data.Data`.

    Nodes carry atom features; edges are bidirectional and carry bond features.
    Stereochemistry is perceived before featurisation so that chiral tags and
    bond-stereo flags are populated correctly.
    """
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    x = torch.tensor(
        [atom_features(atom) for atom in mol.GetAtoms()],
        dtype=torch.float,
    )

    edge_index_list: List[List[int]] = []
    edge_attr_list: List[List[float]] = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        # Add both directions
        edge_index_list += [[i, j], [j, i]]
        edge_attr_list += [bf, bf]

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_FEAT_DIM), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def smiles_to_graph(smiles: str, include_hydrogens: bool = False) -> Optional[Data]:
    """Parse *smiles* with RDKit and return a PyG :class:`Data` graph.

    Returns ``None`` when the SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if include_hydrogens:
        mol = Chem.rdmolops.AddHs(mol)
    return mol_to_graph(mol)
