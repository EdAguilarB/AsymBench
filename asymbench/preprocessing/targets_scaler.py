from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

ScalerType = Literal["none", "standard", "minmax"]


@dataclass
class TargetScaler:
    """
    Simple sklearn-like scaler for regression targets.

    Parameters
    ----------
    scaling : {"none", "standard", "minmax"}
        Scaling strategy:
          - "none": no scaling
          - "standard": (y - mean) / std
          - "minmax": (y - min) / (max - min)
    """

    scaling: ScalerType = "none"

    # learned parameters
    fitted_: bool = False
    mean_: Optional[float] = None
    std_: Optional[float] = None
    min_: Optional[float] = None
    denom_: Optional[float] = None

    def __post_init__(self):
        if self.scaling not in ("none", "standard", "minmax"):
            raise ValueError("scaling must be one of: 'none', 'standard', 'minmax'")

    # -------------------------
    # Public API
    # -------------------------

    def fit(self, y: Union[np.ndarray, pd.Series, pd.DataFrame]) -> "TargetScaler":
        y_arr = self._to_1d_array(y)

        if self.scaling == "standard":
            self.mean_ = float(np.mean(y_arr))
            std = float(np.std(y_arr))
            self.std_ = std if std != 0.0 else 1.0

        elif self.scaling == "minmax":
            self.min_ = float(np.min(y_arr))
            maxv = float(np.max(y_arr))
            denom = maxv - self.min_
            self.denom_ = denom if denom != 0.0 else 1.0

        self.fitted_ = True
        return self

    def transform(self, y: Union[np.ndarray, pd.Series, pd.DataFrame]) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("TargetScaler.transform called before fit().")

        y_arr = self._to_1d_array(y)

        if self.scaling == "none":
            return y_arr.copy()

        if self.scaling == "standard":
            return (y_arr - self.mean_) / self.std_

        if self.scaling == "minmax":
            return (y_arr - self.min_) / self.denom_

        raise ValueError(f"Unknown scaling method: {self.scaling}")

    def inverse_transform(
        self, y_scaled: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("TargetScaler.inverse_transform called before fit().")

        y_arr = self._to_1d_array(y_scaled)

        if self.scaling == "none":
            return y_arr.copy()

        if self.scaling == "standard":
            return y_arr * self.std_ + self.mean_

        if self.scaling == "minmax":
            return y_arr * self.denom_ + self.min_

        raise ValueError(f"Unknown scaling method: {self.scaling}")

    def fit_transform(
        self, y: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> np.ndarray:
        return self.fit(y).transform(y)

    # -------------------------
    # Utilities
    # -------------------------

    def _to_1d_array(self, y: Union[np.ndarray, pd.Series, pd.DataFrame]) -> np.ndarray:
        """
        Convert input to 1D numpy array.
        """
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("TargetScaler expects a single target column.")
            y = y.iloc[:, 0]

        if isinstance(y, pd.Series):
            y = y.to_numpy()

        y = np.asarray(y, dtype=float)

        if y.ndim != 1:
            raise ValueError("TargetScaler expects a 1D target array.")

        return y
