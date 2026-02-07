from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from asymbench.representations.base_featurizer import BaseRepresentation


@dataclass
class DataFrameLookupRepresentation(BaseRepresentation):
    """
    Representation that retrieves precomputed descriptors from a table by index.

    Use cases
    ---------
    - bespoke descriptors computed elsewhere
    - reaction-level descriptors
    - quantum descriptors, physics features, experimental conditions, etc.

    Config expectations
    -------------------
    config["representation"]["params"] must include:
      - "features_path": path to CSV/Parquet with descriptors
      - one of:
          * "index_col": column in features file to use as index (if not already indexed)
          * OR features file already has an index saved (parquet typically does)
      - Optional:
          * "feature_columns": list of columns to use (default: all non-index columns)
          * "prefix": prefix added to all feature names (default: "bespoke")
          * "join_key": dataset column to use for lookup instead of df.index (default: None)
              - If join_key is provided, we match on df[join_key] values.
              - If not provided, we match on df.index.
          * "strict": bool, if True raise when any keys missing (default True)
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        params = self.rep_params
        self.features_path = Path(params["features_path"])
        self.index_col = params.get("index_col", None)
        self.feature_columns = params.get("feature_columns", None)
        self.prefix = params.get("prefix", "bespoke")
        self.join_key = params.get("join_key", None)
        self.strict = bool(params.get("strict", True))

        self._features = self._load_features()

        # keep only requested columns
        if self.feature_columns is not None:
            missing = [
                c for c in self.feature_columns if c not in self._features.columns
            ]
            if missing:
                raise KeyError(
                    f"Requested feature_columns not found in features table: {missing[:5]}"
                )
            self._features = self._features.loc[:, self.feature_columns]

        # prefix column names to avoid collisions with other reps / metadata
        self._features.columns = [f"{self.prefix}__{c}" for c in self._features.columns]

    def _load_features(self) -> pd.DataFrame:
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features file not found: {self.features_path}")

        if self.features_path.suffix.lower() in [".parquet"]:
            feats = pd.read_parquet(self.features_path)
        elif self.features_path.suffix.lower() in [".csv"]:
            feats = pd.read_csv(self.features_path)
        else:
            raise ValueError("features_path must be .csv or .parquet")

        if self.index_col is not None:
            if self.index_col not in feats.columns:
                raise KeyError(
                    f"index_col='{self.index_col}' not present in features file columns."
                )
            feats = feats.set_index(self.index_col)

        # Ensure index is unique for lookup
        if not feats.index.is_unique:
            raise ValueError(
                "Features table index is not unique; cannot do deterministic lookup."
            )

        return feats

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Determine lookup keys
        if self.join_key is None:
            keys = df.index
        else:
            if self.join_key not in df.columns:
                raise KeyError(f"join_key='{self.join_key}' not in dataset columns.")
            keys = df[self.join_key]

        # Align features in the same order as df
        # Reindex preserves order and introduces NaN for missing keys
        X = self._features.reindex(keys)

        # Validate missing
        missing_mask = X.isna().all(axis=1)
        if missing_mask.any():
            n_missing = int(missing_mask.sum())
            example = (
                keys[missing_mask].iloc[0]
                if hasattr(keys, "iloc")
                else list(keys[missing_mask])[0]
            )
            msg = (
                f"{n_missing} rows missing in features lookup table. "
                f"Example missing key: {example}"
            )
            if self.strict:
                raise KeyError(msg)
            # if not strict, fill missing rows with zeros (safe default)
            X = X.fillna(0.0)

        # Ensure returned index matches df.index (important for downstream slicing)
        X.index = df.index
        return X

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "type": self.rep_type,
            "params": {
                "features_path": str(self.features_path),
                "index_col": self.index_col,
                "feature_columns": self.feature_columns,
                "prefix": self.prefix,
                "join_key": self.join_key,
                "strict": self.strict,
            },
            "n_features": int(self._features.shape[1]),
        }
