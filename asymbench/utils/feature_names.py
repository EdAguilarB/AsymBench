from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd

# XGBoost hard restrictions (per your error)
_XGB_FORBIDDEN = r"[\[\]<>]"


def sanitize_feature_name(
    name: str, *, replace_with: str = "_", max_len: int | None = 200
) -> str:
    """
    Sanitize a single feature name to be safe for XGBoost / SHAP TreeExplainer.

    - Ensures string
    - Replaces forbidden chars: [, ], <
    - Collapses whitespace
    - Removes other control characters
    - Optionally truncates to max_len
    """
    s = str(name)

    # Replace forbidden chars
    s = re.sub(_XGB_FORBIDDEN, replace_with, s)

    # Replace any remaining non-printable/control chars
    s = re.sub(r"[\x00-\x1f\x7f-\x9f]", replace_with, s)

    # Normalize whitespace
    s = re.sub(r"\s+", replace_with, s).strip(replace_with)

    # Avoid empty names
    if not s:
        s = "feature"

    # Truncate
    if max_len is not None and len(s) > max_len:
        s = s[:max_len]

    return s


def sanitize_feature_names(
    cols: Iterable,
    *,
    replace_with: str = "_",
    max_len: int | None = 200,
    ensure_unique: bool = True,
) -> Tuple[List[str], Dict[str, str]]:
    """
    Sanitize a list/Index of feature names.

    Returns
    -------
    new_cols : List[str]
        Sanitized (and optionally de-duplicated) feature names.
    mapping : Dict[str, str]
        Mapping from original -> sanitized. (If duplicates exist in original names,
        later keys will overwrite in this dict; if you need lossless mapping,
        store both directions in the caller.)
    """
    original = [str(c) for c in cols]
    sanitized = [
        sanitize_feature_name(c, replace_with=replace_with, max_len=max_len)
        for c in original
    ]

    if ensure_unique:
        # De-duplicate by appending __{n}
        seen: Dict[str, int] = {}
        unique: List[str] = []
        for s in sanitized:
            if s not in seen:
                seen[s] = 0
                unique.append(s)
            else:
                seen[s] += 1
                unique.append(f"{s}__{seen[s]}")
        sanitized = unique

    mapping = {o: n for o, n in zip(original, sanitized)}
    return sanitized, mapping


@dataclass
class FeatureNameSanitizer:
    """
    sklearn-like transformer for feature name sanitation on DataFrames.

    Usage:
      sanitizer = FeatureNameSanitizer()
      X_train = sanitizer.fit_transform(X_train)
      X_test  = sanitizer.transform(X_test)

    Stores:
      - original_to_sanitized_
      - sanitized_to_original_
    """

    replace_with: str = "_"
    max_len: int | None = 200
    ensure_unique: bool = True

    fitted_: bool = False
    original_to_sanitized_: Dict[str, str] | None = None
    sanitized_to_original_: Dict[str, str] | None = None
    columns_: List[str] | None = None

    def fit(self, X: pd.DataFrame) -> "FeatureNameSanitizer":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureNameSanitizer expects a pandas.DataFrame")

        new_cols, mapping = sanitize_feature_names(
            X.columns,
            replace_with=self.replace_with,
            max_len=self.max_len,
            ensure_unique=self.ensure_unique,
        )

        self.original_to_sanitized_ = mapping
        # best-effort inverse mapping (unique if ensure_unique=True)
        self.sanitized_to_original_ = {v: k for k, v in mapping.items()}
        self.columns_ = new_cols
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError(
                "FeatureNameSanitizer.transform called before fit()."
            )
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureNameSanitizer expects a pandas.DataFrame")

        X_out = X.copy()
        X_out.columns = self.columns_

        return X_out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
