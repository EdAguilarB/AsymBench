from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

from asymbench.representations.base import BaseRepresentation


@dataclass
class GraphRepresentation(BaseRepresentation):
    """Reaction representation that produces graph-structured data.

    Each call to :meth:`transform` converts a DataFrame into a
    :class:`~asymbench.gnn.dataset.ReactionGraphDataset` where every
    reaction is encoded as a single disconnected molecular graph
    (see :class:`~asymbench.gnn.reaction_graph.ReactionGraphBuilder`).

    Target values are **not** embedded here; they are injected (scaled)
    by :class:`~asymbench.core.gnn_experiment.GNNExperiment` before
    the DataLoader is constructed.

    YAML example::

        representations:
          - type: graph
            params:
              include_hydrogens: false
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        data_cfg = self.config.get("data", {})
        self.smiles_cols: List[str] = list(data_cfg.get("smiles_columns", []))
        if not self.smiles_cols:
            raise KeyError(
                "config['data']['smiles_columns'] must be provided and non-empty"
            )
        self.include_hydrogens: bool = self.rep_params.get(
            "include_hydrogens", False
        )

    def transform(self, df: pd.DataFrame):
        """Build a :class:`ReactionGraphDataset` for every row in *df*.

        Parameters
        ----------
        df:
            DataFrame with at least the SMILES columns declared in the config.

        Returns
        -------
        ReactionGraphDataset
            In-memory PyG dataset; ``data.y`` is *not* set (see class docstring).
        """
        from asymbench.gnn.dataset import ReactionGraphDataset

        return ReactionGraphDataset(
            df=df,
            smiles_cols=self.smiles_cols,
            target_col=None,
            include_hydrogens=self.include_hydrogens,
        )

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "type": self.rep_type,
            "params": self.rep_params,
            "smiles_columns": self.smiles_cols,
            "include_hydrogens": self.include_hydrogens,
        }
