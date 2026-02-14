import pandas as pd
from doptools import ChythonCircus

from asymbench.representations.base_featurizer import BaseCorpusSmilesFeaturizer


class CircusFeaturizer(BaseCorpusSmilesFeaturizer):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.radius = int(self.rep_params.get("radius", 2))
        self.lower = int(self.rep_params.get("lower", 0))
        self.fmt = self.rep_params.get("fmt", "smiles")

        self._generators = [
            ChythonCircus(lower=self.lower, upper=self.radius, fmt=self.fmt)
            for _ in range(len(self.smiles_cols))
        ]

    def _transform_no_check(self, df: pd.DataFrame) -> pd.DataFrame:
        blocks = []
        for col, gen in zip(self.smiles_cols, self._generators):
            X_col = gen.transform(df[col])
            X_col = X_col.rename(columns={c: f"{col}__{c}" for c in X_col.columns})
            blocks.append(X_col)

        X = pd.concat(blocks, axis=1)
        X.index = df.index
        return X

    def fit(self, df: pd.DataFrame) -> "CircusFeaturizer":
        # Fit each generator on its column
        for col, gen in zip(self.smiles_cols, self._generators):
            gen.fit(df[col])

        # Now we're effectively fitted
        self._is_fitted = True

        # Determine feature names on (small) slice
        X_sample = self._transform_no_check(df.iloc[: min(len(df), 50)])
        self._feature_names = list(X_sample.columns)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError(
                "CircusFeaturizer.transform called before fit(). Use fit on TRAIN only."
            )

        X = self._transform_no_check(df)

        # Ensure consistent columns between train/test:
        # - keep training columns
        # - fill unseen columns in test with 0
        if self._feature_names is not None:
            X = X.reindex(columns=self._feature_names, fill_value=0.0)

        return X