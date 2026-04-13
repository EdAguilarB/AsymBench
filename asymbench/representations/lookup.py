from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from asymbench.representations.base import BaseRepresentation

logger = logging.getLogger(__name__)


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
          * "index_col": column in features file to use as index (if not
            already indexed)
          * OR features file already has an index saved (parquet typically
            does)
      - Optional:
          * "feature_columns": explicit list of columns to use as features.
            When omitted (or set to null/~), **all columns** in the file
            are used after the index column has been set.  Use this mode
            when the CSV contains only an identifier column and feature
            columns with no other metadata mixed in.
          * "prefix": string prepended to every feature column name as
            ``<prefix>__<col>`` to avoid clashes with other representations.
            Set to null/~ or omit entirely to keep the original column
            names unchanged.  (default: no prefix)
          * "join_key": dataset column to use for lookup instead of
            df.index (default: None — match on df.index)
          * "strict": bool, if True raise when any keys are missing from
            the features table (default: True)

    Examples
    --------
    Use a hand-picked subset of columns::

        type: bespoke
        params:
          features_path: data/features.csv
          feature_name: v1
          index_col: Example
          feature_columns: [col_a, col_b, col_c]
          prefix: bespoke
          strict: true

    Use every column in the file (identifier-only CSV)::

        type: bespoke
        params:
          features_path: data/all_features.csv
          feature_name: all
          index_col: Example
          strict: true
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        params = self.rep_params
        self.features_path = Path(params["features_path"])
        self.index_col: Optional[str] = params.get("index_col", None)
        self.feature_columns: Optional[List[str]] = params.get(
            "feature_columns", None
        )
        # None / empty string → no prefix; any other string → "<prefix>__<col>"
        self.prefix: str = params.get("prefix") or ""
        self.join_key: Optional[str] = params.get("join_key", None)
        self.strict: bool = bool(params.get("strict", True))

        self._features = self._load_features()

        self._features.index = _canonicalize_keys(self._features.index)

        if not self._features.index.is_unique:
            dupes = (
                self._features.index[self._features.index.duplicated()]
                .unique()
                .tolist()[:10]
            )
            raise ValueError(
                f"Features index is not unique. Example duplicate keys: {dupes}"
            )

        # ── Column selection ─────────────────────────────────────────────
        if self.feature_columns is not None:
            # Explicit list — validate and filter
            missing = [
                c
                for c in self.feature_columns
                if c not in self._features.columns
            ]
            if missing:
                raise KeyError(
                    f"Requested feature_columns not found in features table: "
                    f"{missing[:5]}"
                )
            self._features = self._features.loc[:, self.feature_columns]
            logger.info(
                "Loaded %d explicitly specified feature column(s) from '%s'.",
                len(self._features.columns),
                self.features_path.name,
            )
        else:
            # No explicit list — use every column remaining after indexing
            non_numeric = [
                c
                for c in self._features.columns
                if not pd.api.types.is_numeric_dtype(self._features[c])
            ]
            if non_numeric:
                logger.warning(
                    "feature_columns was not specified for '%s'; using all "
                    "%d columns, but %d appear non-numeric and may cause "
                    "issues downstream: %s",
                    self.features_path.name,
                    len(self._features.columns),
                    len(non_numeric),
                    non_numeric[:10],
                )
            else:
                logger.info(
                    "feature_columns not specified; using all %d numeric "
                    "column(s) from '%s'.",
                    len(self._features.columns),
                    self.features_path.name,
                )

        # ── Optional prefix ──────────────────────────────────────────────
        if self.prefix:
            self._features.columns = [
                f"{self.prefix}__{c}" for c in self._features.columns
            ]

    def _load_features(self) -> pd.DataFrame:
        if not self.features_path.exists():
            raise FileNotFoundError(
                f"Features file not found: {self.features_path}"
            )

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
            raw_keys = df.index
        else:
            if self.join_key not in df.columns:
                raise KeyError(
                    f"join_key='{self.join_key}' not in dataset columns."
                )
            raw_keys = df[self.join_key]

        keys = _canonicalize_keys(raw_keys)

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
                # None means "all columns" — preserve the original intent
                "feature_columns": self.feature_columns,
                # Empty string means "no prefix" — store None for clarity
                "prefix": self.prefix or None,
                "join_key": self.join_key,
                "strict": self.strict,
            },
            "n_features": int(self._features.shape[1]),
            "using_all_columns": self.feature_columns is None,
        }


@dataclass
class PrecomputedRepresentation(BaseRepresentation):
    """
    Wraps a pre-computed feature DataFrame for efficient reuse across experiments.

    ``transform(df)`` resolves features by index lookup — O(n) with no
    molecular computation.  The wrapped DataFrame must cover every row index
    that will be passed to ``transform`` (both train and test splits, including
    the external test set if applicable).
    """

    _features: pd.DataFrame

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._features.loc[df.index]

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "type": self.rep_type,
            "params": self.rep_params,
            "n_features": len(self._features.columns),
            "precomputed": True,
        }


def _canonicalize_keys(
    values, *, dtype: str = "str", strip: bool = True, lower: bool = False
):
    """
    Return a pd.Index of canonical keys safe for reindexing.
    """
    s = pd.Series(values)

    # Handle missing
    # (keep NA as <NA> so you can detect them)
    if dtype == "str":
        s = s.astype("string")
        if strip:
            s = s.str.strip()
        if lower:
            s = s.str.lower()
        return pd.Index(s)

    if dtype == "int":
        # strict integer conversion; throws if non-int-like values exist
        s = pd.to_numeric(s, errors="raise").astype("int64")
        return pd.Index(s)

    raise ValueError("dtype must be 'str' or 'int'")
