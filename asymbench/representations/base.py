from asymbench.representations.mol_descriptors import RDKitDescriptorFeaturizer
from asymbench.representations.morgan_fp import MorganFeaturizer


def get_representation(config):
    rep_type = config["representation"]["type"]

    if rep_type == "morgan":
        return MorganFeaturizer(config)

    if rep_type == "rdkit":
        return RDKitDescriptorFeaturizer(config)

    raise ValueError(f"Unknown representation: {rep_type}")
