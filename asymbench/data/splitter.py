from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import pandas as pd
import numpy as np
from astartes.molecules import train_test_split_molecules


@dataclass
class MoleculeSplitter:
    """
    Stores split configuration and returns split indices on demand.

    Expected config keys:
      - "train_size"
      - "sampler"
    """

    config: Dict[str, Any]
    random_state: Optional[int] = None

    def __post_init__(self) -> None:
        if "train_size" not in self.config:
            raise KeyError("config must contain 'train_size'")
        if "sampler" not in self.config:
            raise KeyError("config must contain 'sampler'")

        self.train_size = self.config["train_size"]
        self.sampler = self.config["sampler"]

    def get_train_test_set(
        self,
        data: pd.DataFrame,
        mols: List | pd.Series,
        y: pd.Series,
        external_test: pd.DataFrame | None = None,
    ):

        if external_test is not None:

            missing = [
                col
                for col in data.columns.tolist() + [y.name]
                if col not in external_test.columns
            ]

            assert (
                not missing
            ), f"Missing columns in external test set: {missing}"

            df_test = external_test[data.columns]
            df_train = data.copy()
            y_train = y.copy()
            y_test = external_test[y.name]

        else:
            train_idxs, test_idxs = self.get_splits(mols, y)
            df_train = data.iloc[train_idxs]
            df_test = data.iloc[test_idxs]
            y_train = y.iloc[train_idxs]
            y_test = y.iloc[test_idxs]

        return df_train, df_test, y_train, y_test

    def get_splits(
        self,
        mols: List | pd.Series,
        y: pd.DataFrame | np.ndarray | None = None,
        return_indices: bool = True,
    ):
        """
        Returns indices (or split objects) using the stored random state.

        Parameters
        ----------
        mols : sequence
            Molecules passed to train_test_split_molecules.
        y : array-like
            Target values aligned to mols.
        return_indices : bool
            Whether to return indices.

        Returns
        -------
        idxs : Any
            Split indices or objects returned by train_test_split_molecules.
        """
        idxs = train_test_split_molecules(
            molecules=mols,
            y=y,
            train_size=self.train_size,
            sampler=self.sampler,
            random_state=self.random_state,
            return_indices=return_indices,
        )
        train_idxs, test_idxs = idxs[-2], idxs[-1]

        return train_idxs, test_idxs
