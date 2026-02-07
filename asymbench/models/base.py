from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from asymbench.models.gnn_model import GNNWrapper


def get_model(model_type, params=None, **kwargs):
    params = params or {}

    if model_type == "random_forest":
        return RandomForestRegressor(**params)

    elif model_type == "svr":
        return SVR(**params)

    elif model_type == "xgb":
        return XGBRegressor(**params)

    elif model_type == "mlp":
        return MLPRegressor(**params)

    elif model_type == "gnn":
        return GNNWrapper(**kwargs)

    else:
        raise ValueError(f"Unknown model type: {model_type}")
