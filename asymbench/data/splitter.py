from dataclasses import dataclass
from typing import Any, Dict, Optional

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

    def get_splits(self, mols, y=None, return_indices: bool = True):
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
