import pandas as pd


def load_dataset(config):
    path = config["path"]
    data = pd.read_csv(path)
    data = data.set_index(config["id_col"])
    data = data[config["smiles_columns"] + [config["target"]]]
    return data
