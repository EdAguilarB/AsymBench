from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem


@dataclass
class BaseCorpusSmilesFeaturizer(abc.ABC):
    """
    Base for featurizers that require seeing a whole column/corpus
    to define the feature space (e.g., CIRCuS).

    Contract:
      - fit(df) learns feature space (optional but recommended)
      - transform(df) returns pd.DataFrame aligned to df.index
      - fit_transform(df) convenience
      - get_metadata() for caching/signatures
    """

    config: Dict[str, Any]
    sanitize: bool = True

    def __post_init__(self) -> None:
        rep_cfg = self.config.get("representation", {})
        data_cfg = self.config.get("data", {})

        self.rep_type: str = rep_cfg.get("type", self.__class__.__name__)
        self.rep_params: Dict[str, Any] = dict(rep_cfg.get("params", {}))

        self.smiles_cols: List[str] = list(data_cfg.get("smiles_columns", []))
        if not self.smiles_cols:
            raise KeyError(
                "config['data']['smiles_columns'] must be provided and non-empty"
            )

        self._is_fitted: bool = False
        self._feature_names: Optional[List[str]] = None

    @abc.abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseCorpusSmilesFeaturizer":
        raise NotImplementedError

    @abc.abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def feature_names_total(self) -> Optional[List[str]]:
        return self._feature_names

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "type": self.rep_type,
            "params": self.rep_params,
            "smiles_columns": self.smiles_cols,
            "fitted": self._is_fitted,
            "n_features": (
                None
                if self._feature_names is None
                else len(self._feature_names)
            ),
        }


@dataclass
class BaseRepresentation(abc.ABC):
    config: Dict[str, Any]

    def __post_init__(self) -> None:
        rep_cfg = self.config.get("representation", {})
        self.rep_type: str = rep_cfg.get("type", self.__class__.__name__)
        self.rep_params: Dict[str, Any] = dict(rep_cfg.get("params", {}))
        if "feature_name" in self.rep_params:
            self.rep_type = self.rep_type + self.rep_params["feature_name"]

    @abc.abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return feature matrix aligned to df.index."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class BaseSmilesFeaturizer(BaseRepresentation, abc.ABC):
    """
    Base class for SMILES featurizers that handle:
      - multiple SMILES columns
      - SMILES -> Mol conversion
      - invalid SMILES handling
      - per-row concatenation across molecule columns
      - feature naming
      - caching

    Subclasses must implement:
      - feature_dim_per_mol (property)
      - featurize_mol(mol) -> np.ndarray (shape: (feature_dim_per_mol,))
      - feature_names_per_mol() -> List[str] (length: feature_dim_per_mol)
    """

    config: Dict[str, Any]
    use_cache: bool = True
    sanitize: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        data_cfg = self.config.get("data", {})

        self.smiles_cols: List[str] = list(data_cfg.get("smiles_columns", []))
        if not self.smiles_cols:
            raise KeyError(
                "config['data']['smiles_columns'] must be provided and non-empty"
            )

        # Cache: raw SMILES -> np.ndarray (per-molecule feature vector)
        self._cache: Dict[str, np.ndarray] = {} if self.use_cache else {}

    # ---------- Required subclass interface ----------

    @property
    @abc.abstractmethod
    def feature_dim_per_mol(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def featurize_mol(self, mol: Chem.Mol) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def feature_names_per_mol(self) -> List[str]:
        """Return per-molecule feature names (length feature_dim_per_mol)."""
        raise NotImplementedError

    # ---------- Shared ----------

    def feature_dim_total(self) -> int:
        return self.feature_dim_per_mol * len(self.smiles_cols)

    def feature_names_total(self) -> List[str]:
        """
        Prefix per-molecule feature names with the SMILES column name so
        features from different molecules never collide.
        Example: catalyst__MolWt, substrate__MolWt
        """
        base_names = self.feature_names_per_mol()
        if len(base_names) != self.feature_dim_per_mol:
            raise ValueError(
                f"{self.__class__.__name__}.feature_names_per_mol returned {len(base_names)} names, "
                f"expected {self.feature_dim_per_mol}"
            )

        names: List[str] = []
        for col in self.smiles_cols:
            names.extend([f"{col}__{n}" for n in base_names])
        return names

    def _zero_features(self) -> np.ndarray:
        return np.zeros(self.feature_dim_per_mol, dtype=float)

    def smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        if smiles is None:
            return None
        if not isinstance(smiles, str):
            smiles = str(smiles)

        smiles = smiles.strip()
        if not smiles:
            return None

        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=self.sanitize)
        except Exception:
            return None
        return mol

    def featurize_smiles(self, smiles: str) -> np.ndarray:
        if self.use_cache and smiles in self._cache:
            return self._cache[smiles]

        mol = self.smiles_to_mol(smiles)
        if mol is None:
            feats = self._zero_features()
        else:
            feats = np.asarray(self.featurize_mol(mol), dtype=float)
            if feats.shape != (self.feature_dim_per_mol,):
                raise ValueError(
                    f"{self.__class__.__name__}.featurize_mol returned shape {feats.shape}, "
                    f"expected ({self.feature_dim_per_mol},)"
                )

        if self.use_cache:
            self._cache[smiles] = feats
        return feats

    def transform(self, df) -> pd.DataFrame:
        """
        Transform dataframe into a feature DataFrame with named columns.

        Returns
        -------
        X_df : pd.DataFrame shape (n_samples, feature_dim_total)
        """
        n = len(df)
        colnames = self.feature_names_total()
        out = np.zeros((n, len(colnames)), dtype=float)

        for i, (_, row) in enumerate(df.iterrows()):
            row_feats = []
            for col in self.smiles_cols:
                row_feats.append(self.featurize_smiles(row[col]))
            out[i, :] = np.concatenate(row_feats, axis=0)

        return pd.DataFrame(out, columns=colnames, index=df.index)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "type": self.rep_type,
            "params": self.rep_params,
            "smiles_columns": self.smiles_cols,
            "feature_dim_per_mol": self.feature_dim_per_mol,
            "feature_dim_total": self.feature_dim_total(),
            "sanitize": self.sanitize,
            "use_cache": self.use_cache,
        }
