from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable

import numpy as np
import optuna
from sklearn.model_selection import KFold, cross_val_score

from asymbench.models.base import get_model


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


@dataclass
class OptunaSklearnOptimizer:
    """
    Optuna optimizer for sklearn models using CV on TRAIN only.

    Expects model_cfg like:
      {"type": "...", "hpo": {"enabled": True, "n_trials": 50, "cv": 3, "search_space": {...}}}
    """

    model_cfg: Dict[str, Any]
    seed: int
    direction: str = "minimize"  # minimize RMSE

    def optimize(
        self, X_train, y_train
    ) -> Tuple[Any, Dict[str, Any], float, Dict[str, Any]]:
        hpo = self.model_cfg.get("hpo", {}) or {}
        if not hpo.get("enabled", False):
            model = get_model(
                self.model_cfg["type"], params=self.model_cfg.get("params", {})
            )
            return model, self.model_cfg.get("params", {}), np.nan, {"enabled": False}

        n_trials = int(hpo["n_trials"])
        cv = int(hpo.get("cv", 3))
        space = hpo["search_space"]

        def suggest(trial: optuna.Trial, spec: Dict[str, Any]):
            t = spec["type"]
            if t == "int":
                return trial.suggest_int(
                    spec.get("name"),
                    spec["low"],
                    spec["high"],
                    step=spec.get("step", 1),
                )
            if t == "float":
                return trial.suggest_float(
                    spec.get("name"),
                    spec["low"],
                    spec["high"],
                    log=bool(spec.get("log", False)),
                )
            if t == "categorical":
                return trial.suggest_categorical(spec.get("name"), spec["choices"])
            raise ValueError(f"Unknown search space type: {t}")

        # Normalize spec: each param dict doesn’t contain name; add it
        space_named = {k: dict(v, name=k) for k, v in space.items()}

        def objective(trial: optuna.Trial) -> float:
            params = {k: suggest(trial, spec) for k, spec in space_named.items()}

            model = get_model(self.model_cfg["type"], params=params)

            # Use sklearn CV and neg RMSE
            kf = KFold(n_splits=cv, shuffle=True, random_state=self.seed)
            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=kf,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
            )
            # convert to RMSE (positive)
            return float(-scores.mean())

        study = optuna.create_study(
            direction=self.direction, sampler=optuna.samplers.TPESampler(seed=self.seed)
        )
        study.optimize(objective, n_trials=n_trials)

        best_params = dict(study.best_params)
        best_score = float(study.best_value)  # RMSE
        best_model = get_model(self.model_cfg["type"], params=best_params)

        meta = {
            "enabled": True,
            "n_trials": n_trials,
            "cv": cv,
            "best_value_rmse": best_score,
        }

        return best_model, best_params, best_score, meta
