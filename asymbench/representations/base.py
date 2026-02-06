from asymbench.representations.morgan_fp import MorganFeaturizer
from asymbench.representations.mol_descriptors import RDKitDescriptorFeaturizer
from asymbench.representations.df_lookup import DataFrameLookupRepresentation


def get_representation(config):
    rep_type = config["representation"]["type"]

    if rep_type == "morgan":
        return MorganFeaturizer(config)

    if rep_type == "rdkit":
        return RDKitDescriptorFeaturizer(config)

    if rep_type in ("df_lookup", "bespoke", "precomputed"):
        return DataFrameLookupRepresentation(config)

    raise ValueError(f"Unknown representation: {rep_type}")
