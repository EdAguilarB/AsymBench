from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from asymbench.representations.base_featurizer import BaseRepresentation


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
