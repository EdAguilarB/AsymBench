import numpy as np
from rdkit.Chem import rdFingerprintGenerator

from asymbench.representations.base_featurizer import BaseSmilesFeaturizer


class MorganFeaturizer(BaseSmilesFeaturizer):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.radius = int(self.rep_params.get("radius", 2))
        self.n_bits = int(self.rep_params.get("n_bits", 2048))

        # Create generator once
        self._generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius, fpSize=self.n_bits
        )

    @property
    def feature_dim_per_mol(self) -> int:
        return self.n_bits

    def featurize_mol(self, mol) -> np.ndarray:
        fp = self._generator.GetFingerprint(mol)
        return np.asarray(fp, dtype=float)

    def feature_names_per_mol(self):
        width = len(str(self.n_bits - 1))
        return [f"morgan_bit_{i:0{width}d}" for i in range(self.n_bits)]
