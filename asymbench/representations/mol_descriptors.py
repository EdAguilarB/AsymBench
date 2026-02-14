import numpy as np
from rdkit.Chem import Descriptors

from asymbench.representations.base_featurizer import BaseSmilesFeaturizer


class RDKitDescriptorFeaturizer(BaseSmilesFeaturizer):
    def __post_init__(self) -> None:
        super().__post_init__()
        # Descriptors.descList entries are (name, function)
        self.desc_names = [name for name, _ in Descriptors.descList]
        self.desc_fns = [fn for _, fn in Descriptors.descList]

    @property
    def feature_dim_per_mol(self) -> int:
        return len(Descriptors.descList)

    def featurize_mol(self, mol) -> np.ndarray:
        return np.array([fn(mol) for fn in self.desc_fns], dtype=float)

    def feature_names_per_mol(self):
        # Keep names exactly as RDKit provides (stable + recognizable)
        return list(self.desc_names)
