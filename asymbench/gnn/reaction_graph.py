"""Reaction graph representation.

A reaction is represented as a single disconnected graph formed by
merging the molecular graphs of all participant molecules (substrate,
ligand, solvent, …).  Nodes and edges within each component retain
their original features; there are no inter-molecular edges.

Example
-------
>>> builder = ReactionGraphBuilder(smiles_cols=["substrate", "ligand", "solvent"])
>>> reaction_graph = builder.build(df.iloc[0])
>>> reaction_graph
Data(x=[63, 33], edge_index=[2, 132], edge_attr=[132, 8])
"""

from typing import List, Optional

import pandas as pd
import torch
from torch_geometric.data import Data

from asymbench.gnn.featurizer import (
    EDGE_FEAT_DIM,
    NODE_FEAT_DIM,
    smiles_to_graph,
)


def merge_molecular_graphs(graphs: List[Data]) -> Data:
    """Merge a list of molecular :class:`Data` graphs into one disconnected graph.

    Node indices in each successive graph are offset by the cumulative node
    count of all preceding graphs, so the edge connectivity within each
    component is preserved without collisions.

    Parameters
    ----------
    graphs:
        Non-empty list of PyG :class:`Data` objects, each with ``x``,
        ``edge_index``, and ``edge_attr`` attributes.

    Returns
    -------
    Data
        A single :class:`Data` object whose node/edge tensors are the
        concatenation of those of the input graphs.
    """
    if not graphs:
        raise ValueError("graphs must be a non-empty list")

    x_parts = []
    edge_index_parts = []
    edge_attr_parts = []
    node_offset = 0

    for g in graphs:
        x_parts.append(g.x)
        edge_index_parts.append(g.edge_index + node_offset)
        edge_attr_parts.append(g.edge_attr)
        node_offset += g.num_nodes

    return Data(
        x=torch.cat(x_parts, dim=0),
        edge_index=torch.cat(edge_index_parts, dim=1),
        edge_attr=torch.cat(edge_attr_parts, dim=0),
    )


class ReactionGraphBuilder:
    """Build reaction graph representations from tabular reaction data.

    A reaction graph is a single :class:`torch_geometric.data.Data` object
    whose nodes and edges are the union of the molecular graphs for every
    molecule listed in *smiles_cols*.  The component molecular graphs are
    disconnected from one another (no inter-molecular edges are added).

    Parameters
    ----------
    smiles_cols:
        Ordered list of DataFrame column names that hold SMILES strings for
        each reaction component (e.g. ``["substrate_smiles", "ligand_smiles",
        "solvent_smiles"]``).
    include_hydrogens:
        If ``True``, explicit hydrogen atoms are added to each molecule before
        featurisation.

    Attributes
    ----------
    node_feature_dim : int
        Number of features per atom node.
    edge_feature_dim : int
        Number of features per directed bond edge.
    """

    def __init__(
        self, smiles_cols: List[str], include_hydrogens: bool = False
    ) -> None:
        if not smiles_cols:
            raise ValueError("smiles_cols must be a non-empty list")
        self.smiles_cols = smiles_cols
        self.include_hydrogens = include_hydrogens

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def node_feature_dim(self) -> int:
        return NODE_FEAT_DIM

    @property
    def edge_feature_dim(self) -> int:
        return EDGE_FEAT_DIM

    def build(self, row: pd.Series) -> Optional[Data]:
        """Build a reaction graph from a single DataFrame row.

        Parameters
        ----------
        row:
            A pandas Series (one reaction) that contains at least the columns
            listed in *smiles_cols*.

        Returns
        -------
        Data or None
            The merged reaction graph, or ``None`` if any SMILES is invalid.
        """
        mol_graphs: List[Data] = []
        for col in self.smiles_cols:
            g = smiles_to_graph(str(row[col]), self.include_hydrogens)
            if g is None:
                return None
            mol_graphs.append(g)

        graph = merge_molecular_graphs(mol_graphs)
        # Store per-molecule SMILES (ordered) so explainability can map
        # per-atom scores back to molecular fragments.
        graph.mol_smiles = [(col, str(row[col])) for col in self.smiles_cols]
        return graph

    def build_dataset(
        self, df: pd.DataFrame, target_col: Optional[str] = None
    ) -> List[Data]:
        """Build reaction graphs for every row in *df*.

        Rows whose SMILES cannot be parsed are silently skipped.

        Parameters
        ----------
        df:
            DataFrame with at least the columns in *smiles_cols* (and
            optionally *target_col*).
        target_col:
            If provided, the corresponding value is stored as ``data.y``
            (shape ``[1]``, dtype ``float``).

        Returns
        -------
        list of Data
            One graph per valid reaction, with ``data.idx`` set to the
            DataFrame index of that row.
        """
        data_list: List[Data] = []
        for idx, row in df.iterrows():
            graph = self.build(row)
            if graph is None:
                continue
            if target_col is not None:
                graph.y = torch.tensor([row[target_col]], dtype=torch.float)
            graph.idx = idx
            data_list.append(graph)
        return data_list
