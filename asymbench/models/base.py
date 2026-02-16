from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

from asymbench.models.gnn_model import GNNWrapper


def _set_if_missing(params: dict, key: str, value):
    """Set params[key]=value only if key not already present."""
    if key not in params and value is not None:
        params[key] = value


def get_model(model_type, params=None, seed: int | None = None, **kwargs):
    params = dict(params or {})  # copy so we don't mutate caller

    if model_type == "random_forest":
        # sklearn RF is stochastic -> random_state controls it
        _set_if_missing(params, "random_state", seed)
        return RandomForestRegressor(**params)

    elif model_type == "svr":
        # SVR is deterministic given data; no seed needed
        return SVR(**params)

    elif model_type == "xgb":
        _set_if_missing(params, "random_state", seed)
        return XGBRegressor(**params)

    elif model_type == "mlp":
        _set_if_missing(params, "random_state", seed)
        return MLPRegressor(**params)

    else:
        raise ValueError(f"Unknown model type: {model_type}")
