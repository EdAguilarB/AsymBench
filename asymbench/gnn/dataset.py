from typing import List, Optional

import pandas as pd
from torch_geometric.data import Dataset

from asymbench.gnn.reaction_graph import ReactionGraphBuilder


class ReactionGraphDataset(Dataset):
    """In-memory PyG dataset of reaction graphs.

    Each sample is a single :class:`torch_geometric.data.Data` object
    representing the reaction as a disconnected graph of its constituent
    molecules (see :class:`~asymbench.gnn.reaction_graph.ReactionGraphBuilder`).

    Parameters
    ----------
    df:
        DataFrame with one reaction per row.
    smiles_cols:
        Column names holding SMILES for each reaction component.
    target_col:
        Column name for the regression/classification target.  If ``None``,
        ``data.y`` is not set.
    include_hydrogens:
        Add explicit H atoms before featurisation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        smiles_cols: List[str],
        target_col: Optional[str] = None,
        include_hydrogens: bool = False,
    ) -> None:
        super().__init__()
        builder = ReactionGraphBuilder(smiles_cols, include_hydrogens)
        self._data: List = builder.build_dataset(df, target_col)

    def len(self) -> int:
        return len(self._data)

    def get(self, idx: int):
        return self._data[idx]
