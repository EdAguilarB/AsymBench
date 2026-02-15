from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

ScalerType = Literal["none", "standard", "minmax"]
NaStrategy = Literal["raise", "fill"]
CorrMethod = Literal["pearson", "spearman"]


@dataclass
class FeaturePreprocessor:
    """
    Options-driven feature preprocessing with strict validation.

    Expects preprocessing_options dict with keys:

    Required keys
    -------------
    - "scaling": one of {"none","standard","minmax"}
    - "feature_selection": dict with required keys:
        - "variance_filter": dict with required keys:
            - "enabled": bool
            - "threshold": float
        - "correlation_filter": dict with required keys:
            - "enabled": bool
            - "threshold": float
            - "method": {"pearson","spearman"}

    Optional keys
    -------------
    - "nan_strategy": {"raise","fill"}   (default: "raise")
    - "fill_value": float               (default: 0.0; used if nan_strategy="fill")
    """

    preprocessing_options: Dict[str, Any]

    fitted_: bool = False
    kept_columns_: Optional[List[str]] = None
    dropped_columns_: Optional[List[str]] = None
    scaler_params_: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self._validate_and_parse_options(self.preprocessing_options)

    # -------------------------
    # Public API
    # -------------------------

    def fit(self, X: pd.DataFrame) -> "FeaturePreprocessor":
        X = self._validate_input(X, stage="fit")

        cols_initial = list(X.columns)
        dropped: List[str] = []

        # variance filter
        if self.var_enabled:
            variances = X.var(axis=0, ddof=0)
            keep_mask = variances > self.var_threshold
            kept = variances.index[keep_mask].tolist()
            dropped_var = variances.index[~keep_mask].tolist()
            dropped.extend(dropped_var)
            X = X[kept]
        else:
            kept = cols_initial

        # correlation filter
        if self.corr_enabled:
            kept_after_corr, dropped_corr = self._corr_filter(
                X, threshold=self.corr_threshold, method=self.corr_method
            )
            dropped.extend(dropped_corr)
            kept = kept_after_corr
            X = X[kept]

        # scaler fit
        self.scaler_params_ = None
        if self.scaling != "none":
            self.scaler_params_ = self._fit_scaler(X, method=self.scaling)

        self.kept_columns_ = kept
        self.dropped_columns_ = [
            c for c in cols_initial if c not in self.kept_columns_
        ]
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError(
                "FeaturePreprocessor.transform called before fit()."
            )

        X = self._validate_input(X, stage="transform")

        missing = [c for c in self.kept_columns_ if c not in X.columns]
        if missing:
            raise KeyError(
                f"Input is missing {len(missing)} feature columns learned during fit. "
                f"First missing: {missing[0]}"
            )

        X_out = X.loc[:, self.kept_columns_].copy()

        if self.scaling != "none":
            X_out = self._apply_scaler(
                X_out, self.scaler_params_, method=self.scaling
            )

        return X_out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "scaling": self.scaling,
            "nan_strategy": self.nan_strategy,
            "fill_value": self.fill_value,
            "variance_filter": {
                "enabled": self.var_enabled,
                "threshold": self.var_threshold,
            },
            "correlation_filter": {
                "enabled": self.corr_enabled,
                "threshold": self.corr_threshold,
                "method": self.corr_method,
            },
            "n_features_in": (
                None
                if self.kept_columns_ is None
                else (len(self.kept_columns_) + len(self.dropped_columns_))
            ),
            "n_features_out": (
                None if self.kept_columns_ is None else len(self.kept_columns_)
            ),
            "dropped_columns": self.dropped_columns_,
        }

    # -------------------------
    # Strict options validation
    # -------------------------

    def _require(self, dct: Dict[str, Any], key: str, ctx: str) -> Any:
        if key not in dct:
            raise KeyError(f"Missing required key '{key}' in {ctx}.")
        return dct[key]

    def _validate_and_parse_options(self, opt: Dict[str, Any]) -> None:
        if not isinstance(opt, dict):
            raise TypeError("preprocessing_options must be a dict.")

        scaling = self._require(opt, "scaling", "preprocessing_options")
        if scaling not in ("none", "standard", "minmax"):
            raise ValueError(
                "preprocessing_options['scaling'] must be one of: 'none', 'standard', 'minmax'."
            )

        fs = self._require(opt, "feature_selection", "preprocessing_options")
        if not isinstance(fs, dict):
            raise TypeError(
                "preprocessing_options['feature_selection'] must be a dict."
            )

        var = self._require(fs, "variance_filter", "feature_selection")
        corr = self._require(fs, "correlation_filter", "feature_selection")

        if not isinstance(var, dict):
            raise TypeError(
                "feature_selection['variance_filter'] must be a dict."
            )
        if not isinstance(corr, dict):
            raise TypeError(
                "feature_selection['correlation_filter'] must be a dict."
            )

        var_enabled = self._require(var, "enabled", "variance_filter")
        var_threshold = self._require(var, "threshold", "variance_filter")
        if not isinstance(var_enabled, bool):
            raise TypeError("variance_filter['enabled'] must be bool.")
        if not isinstance(var_threshold, (int, float)):
            raise TypeError("variance_filter['threshold'] must be a number.")

        corr_enabled = self._require(corr, "enabled", "correlation_filter")
        corr_threshold = self._require(corr, "threshold", "correlation_filter")
        corr_method = self._require(corr, "method", "correlation_filter")

        if not isinstance(corr_enabled, bool):
            raise TypeError("correlation_filter['enabled'] must be bool.")
        if not isinstance(corr_threshold, (int, float)):
            raise TypeError("correlation_filter['threshold'] must be a number.")
        if corr_method not in ("pearson", "spearman"):
            raise ValueError(
                "correlation_filter['method'] must be one of: 'pearson', 'spearman'."
            )

        nan_strategy = opt.get("nan_strategy", "raise")
        if nan_strategy not in ("raise", "fill"):
            raise ValueError(
                "preprocessing_options['nan_strategy'] must be 'raise' or 'fill'."
            )

        fill_value = opt.get("fill_value", 0.0)
        if not isinstance(fill_value, (int, float)):
            raise TypeError(
                "preprocessing_options['fill_value'] must be a number."
            )

        # Store parsed values
        self.scaling: ScalerType = scaling
        self.nan_strategy: NaStrategy = nan_strategy
        self.fill_value: float = float(fill_value)

        self.var_enabled: bool = var_enabled
        self.var_threshold: float = float(var_threshold)

        self.corr_enabled: bool = corr_enabled
        self.corr_threshold: float = float(corr_threshold)
        self.corr_method: CorrMethod = corr_method

    # -------------------------
    # Input validation
    # -------------------------

    def _validate_input(self, X: pd.DataFrame, stage: str) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas.DataFrame, got {type(X)}")

        non_numeric = [
            c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])
        ]
        if non_numeric:
            raise TypeError(
                f"All columns must be numeric for preprocessing. Found non-numeric columns: {non_numeric[:5]}"
                + (" ..." if len(non_numeric) > 5 else "")
            )

        X = X.copy()

        has_inf = np.isinf(X.to_numpy()).any()
        has_nan = X.isna().to_numpy().any()

        if has_inf or has_nan:
            if self.nan_strategy == "raise":
                raise ValueError(
                    f"NaN or inf detected in X during {stage}. "
                    f"Set preprocessing_options['nan_strategy']='fill' to auto-handle."
                )
            X = X.replace([np.inf, -np.inf], np.nan).fillna(self.fill_value)

        return X

    # -------------------------
    # Filters
    # -------------------------

    def _corr_filter(
        self, X: pd.DataFrame, threshold: float, method: CorrMethod
    ) -> Tuple[List[str], List[str]]:
        corr = X.corr(method=method).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop = set()
        for col in upper.columns:
            if col in to_drop:
                continue
            # if col has high corr with any previous feature, drop col (greedy)
            if (upper[col] >= threshold).any():
                to_drop.add(col)

        kept = [c for c in X.columns if c not in to_drop]
        dropped = [c for c in X.columns if c in to_drop]
        return kept, dropped

    # -------------------------
    # Scaling
    # -------------------------

    def _fit_scaler(
        self, X: pd.DataFrame, method: ScalerType
    ) -> Dict[str, Any]:
        arr = X.to_numpy(dtype=float)

        if method == "standard":
            mean = arr.mean(axis=0)
            std = arr.std(axis=0, ddof=0)
            std = np.where(std == 0.0, 1.0, std)
            return {"mean": mean, "std": std}

        if method == "minmax":
            minv = arr.min(axis=0)
            maxv = arr.max(axis=0)
            denom = maxv - minv
            denom = np.where(denom == 0.0, 1.0, denom)
            return {"min": minv, "denom": denom}

        raise ValueError(f"Unknown scaling method: {method}")

    def _apply_scaler(
        self, X: pd.DataFrame, params: Dict[str, Any], method: ScalerType
    ) -> pd.DataFrame:
        arr = X.to_numpy(dtype=float)

        if method == "standard":
            arr = (arr - params["mean"]) / params["std"]
        elif method == "minmax":
            arr = (arr - params["min"]) / params["denom"]
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        return pd.DataFrame(arr, columns=X.columns, index=X.index)
