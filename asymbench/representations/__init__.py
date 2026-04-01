from asymbench.representations.circus import (
    CircusFeaturizer,
)
from asymbench.representations.descriptors import RDKitDescriptorFeaturizer
from asymbench.representations.fingerprints import MorganFeaturizer
from asymbench.representations.lookup import (
    DataFrameLookupRepresentation,
)
from asymbench.representations.transformers import HFTransformerFeaturizer
from asymbench.representations.unimol import UniMolFeaturizer


def get_representation(config):
    rep_type = config["representation"]["type"]

    if rep_type == "morgan":
        return MorganFeaturizer(config)

    if rep_type == "rdkit":
        return RDKitDescriptorFeaturizer(config)

    if rep_type == "circus":
        return CircusFeaturizer(config)

    if rep_type in ("df_lookup", "bespoke", "precomputed"):
        return DataFrameLookupRepresentation(config)

    if rep_type == "hf_transformer":
        return HFTransformerFeaturizer(config)

    if rep_type == "unimol":
        return UniMolFeaturizer(config)

    raise ValueError(f"Unknown representation: {rep_type}")
