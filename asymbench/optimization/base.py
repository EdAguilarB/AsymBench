from asymbench.optimization.optuna_optimizer import OptunaSklearnOptimizer


def get_optimizer(model_cfg, seed: int):
    # later: if model_cfg["type"] == "gnn": return OptunaGNNOptimizer(...)
    return OptunaSklearnOptimizer(model_cfg=model_cfg, seed=seed)
