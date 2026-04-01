from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from unimol_tools import UniMolRepr

from asymbench.representations.base import BaseSmilesFeaturizer


@dataclass
class UniMolFeaturizer(BaseSmilesFeaturizer):
    """
    UniMol featurizer for SMILES-based molecular embeddings.

    Expected rep_params
    -------------------
    data_type : str
        One of {"molecule", "oled", "pocket"}. Default: "molecule".
    remove_hs : bool
        Whether to remove hydrogens. Default: False.
    model_name : str
        One of {"unimolv1", "unimolv2"}. Default: "unimolv1".
    model_size : str
        Only used for unimolv2. One of {"84m", "164m", "310m", "570m", "1.1B"}.
        Default: "84m".
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        self.data_type: str = (
            str(self.rep_params.get("data_type", "molecule")).strip().lower()
        )
        self.remove_hs: bool = bool(self.rep_params.get("remove_hs", False))
        self.model_name: str = (
            str(self.rep_params.get("model_name", "unimolv1")).strip().lower()
        )
        self.model_size: str = (
            str(self.rep_params.get("model_size", "84m")).strip().lower()
        )

        self._clf = UniMolRepr(
            data_type=self.data_type,
            remove_hs=self.remove_hs,
            model_name=self.model_name,
            model_size=self.model_size,
        )

        # Resolve embedding dimension with a single probe call
        probe = self._clf.get_repr(["CCO"], return_atomic_reprs=False)
        self._feature_dim: int = probe[0].shape[0]

    # ------------------------------------------------------------------ #
    #  BaseSmilesFeaturizer interface                                       #
    # ------------------------------------------------------------------ #

    @property
    def feature_dim_per_mol(self) -> int:
        return self._feature_dim

    def featurize_mol(self, mol: Chem.Mol) -> np.ndarray:
        smiles = Chem.MolToSmiles(mol, canonical=True)
        result = self._clf.get_repr([smiles], return_atomic_reprs=False)
        return np.asarray(result[0], dtype=float)

    def feature_names_per_mol(self) -> List[str]:
        width = len(str(self._feature_dim - 1))
        return [
            f"unimol_{self.model_name}_emb_{i:0{width}d}"
            for i in range(self._feature_dim)
        ]

    # ------------------------------------------------------------------ #
    #  Batch transform — one get_repr call per column, not per molecule    #
    # ------------------------------------------------------------------ #

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        colnames = self.feature_names_total()
        out_blocks = []

        for col in self.smiles_cols:
            result = self._clf.get_repr(
                df[col].tolist(), return_atomic_reprs=False
            )
            out_blocks.append(np.stack(result, axis=0))

        out = np.concatenate(out_blocks, axis=1)
        return pd.DataFrame(out, columns=colnames, index=df.index)
