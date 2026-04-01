import json

from doptools import ChythonCircus
import pandas as pd

from asymbench.representations.base_featurizer import (
    BaseCorpusSmilesFeaturizer,
)


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
            X_col = X_col.rename(
                columns={c: f"{col}__{c}" for c in X_col.columns}
            )
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


class CachingCircusRepresentation:
    """
    Wraps CircusFeaturizer with a shared fit-cache keyed by training-set identity.

    When multiple experiments share the same training split (same split strategy,
    train size, and seed) but differ only in the downstream model, ``fit()`` is
    executed only once; subsequent calls restore the fitted state from the cache.

    The cache dict is owned by ``BenchmarkRunner`` and shared across all
    instances that belong to the same benchmark run.
    """

    def __init__(self, config: dict, fit_cache: dict) -> None:
        self._inner = CircusFeaturizer(config)
        self._fit_cache = fit_cache
        self._cache_key_str = json.dumps(config["representation"], sort_keys=True)
        # Expose attributes read by Experiment._run_signature()
        self.rep_type = self._inner.rep_type
        self.rep_params = self._inner.rep_params

    def fit(self, df_train: pd.DataFrame) -> "CachingCircusRepresentation":
        cache_key = (tuple(sorted(df_train.index.tolist())), self._cache_key_str)

        if cache_key in self._fit_cache:
            cached = self._fit_cache[cache_key]
            # Restore fitted state from the cached instance.
            # The generators were fit-once and are not mutated after caching.
            self._inner._generators = cached._generators
            self._inner._is_fitted = True
            self._inner._feature_names = cached._feature_names
        else:
            self._inner.fit(df_train)
            # Store the fitted inner instance; it will not be re-fit after this.
            self._fit_cache[cache_key] = self._inner
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._inner.transform(df)

    def get_metadata(self) -> dict:
        return self._inner.get_metadata()
