from __future__ import annotations

from dataclasses import dataclass
import gc
import logging
from typing import Any, Dict, Tuple

import numpy as np
import optuna
from sklearn.model_selection import KFold, cross_val_score

from asymbench.models.base import get_model

logger = logging.getLogger(__name__)

# Suppress Optuna’s per-trial INFO chatter globally.  The benchmark already
# prints its own progress; 50+ lines of Optuna output per HPO run adds noise
# without adding information.  Set OPTUNA_VERBOSITY=INFO in the environment
# to re-enable if you need to debug a specific study.


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _to_float(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        # handles "1e-6", "0.01", etc.
        return float(x.strip())
    raise TypeError(f"Expected numeric, got {type(x)}: {x}")


def _to_int(x):
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if isinstance(x, str):
        return int(float(x.strip()))
    raise TypeError(f"Expected int-like, got {type(x)}: {x}")


@dataclass
class OptunaSklearnOptimizer:
    """
    Optuna optimizer for sklearn models using CV on TRAIN only.

    Expects model_cfg like:
      {"type": "...", "hpo": {"enabled": True, "n_trials": 50, "cv": 3, "search_space": {...}}}

    Implementation notes
    --------------------
    Cross-validation inside the Optuna objective runs with ``n_jobs=1``
    (single process, sequential folds).  Using ``n_jobs=-1`` would spawn a
    ``loky`` worker-process pool on every trial call.  On machines with a
    GPU, forked workers inherit the parent’s CUDA context from any prior GNN
    run, leaving a growing number of GPU processes visible in ``nvidia-smi``.
    Even on CPU-only machines the overhead of process creation dominates for
    the small number of CV folds used here (typically 3).  Optuna trials are
    already sequential, so intra-trial parallelism gives negligible gain.

    XGBoost is forced to CPU (``device=’cpu’``) during HPO unless the caller
    has explicitly set ``device`` in the search space.  XGBoost’s default
    ``tree_method=’auto’`` silently switches to the GPU when CUDA is
    available, which causes dozens of simultaneous GPU processes when many
    trials are running with parallel workers.
    """

    model_cfg: Dict[str, Any]
    seed: int
    direction: str = "minimize"  # minimize RMSE

    def optimize(
        self, X_train, y_train
    ) -> Tuple[Any, Dict[str, Any], float, Dict[str, Any]]:
        optuna.logging.set_verbosity(optuna.logging.INFO)
        hpo = self.model_cfg.get("hpo", {}) or {}
        if not hpo.get("enabled", False):
            model = get_model(
                self.model_cfg["type"],
                params=self.model_cfg.get("params", {}),
                seed=self.seed,
            )
            return (
                model,
                self.model_cfg.get("params", {}),
                np.nan,
                {"enabled": False},
            )

        n_trials = int(hpo["n_trials"])
        cv = int(hpo.get("cv", 3))
        space = hpo["search_space"]
        model_type = self.model_cfg["type"]

        def suggest(trial: optuna.Trial, spec: Dict[str, Any]):
            t = spec["type"]

            if t == "int":
                low = _to_int(spec["low"])
                high = _to_int(spec["high"])
                step = _to_int(spec.get("step", 1))
                return trial.suggest_int(spec["name"], low, high, step=step)

            if t == "float":
                low = _to_float(spec["low"])
                high = _to_float(spec["high"])
                log = bool(spec.get("log", False))
                step = spec.get("step", None)
                step = _to_float(step) if step is not None else None
                return trial.suggest_float(
                    spec["name"], low, high, log=log, step=step
                )

            if t == "categorical":
                return trial.suggest_categorical(spec["name"], spec["choices"])

            raise ValueError(f"Unknown search space type: {t}")

        # Normalize spec: each param dict doesn’t contain name; add it
        space_named = {k: dict(v, name=k) for k, v in space.items()}

        def objective(trial: optuna.Trial) -> float:
            params = {
                k: suggest(trial, spec) for k, spec in space_named.items()
            }

            # Force XGBoost to use CPU during HPO unless the user has
            # explicitly included ‘device’ in the search space.  XGBoost’s
            # default tree_method=’auto’ will otherwise switch to GPU when
            # CUDA is available, spawning a new GPU process per trial.
            if model_type == "xgb" and "device" not in params:
                params["device"] = "cpu"

            model = get_model(model_type, params=params, seed=self.seed)

            # n_jobs=1 — run CV folds sequentially in the current process.
            # Avoids spawning a loky worker pool per trial, which on GPU
            # machines leaves orphaned processes holding CUDA contexts and
            # causes nvidia-smi to show an ever-growing process list.
            kf = KFold(n_splits=cv, shuffle=True, random_state=self.seed)
            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=kf,
                scoring="neg_root_mean_squared_error",
                n_jobs=1,
            )
            # convert to RMSE (positive)
            return float(-scores.mean())

        logger.info(
            "Starting HPO for %s: %d trials, %d-fold CV.",
            model_type,
            n_trials,
            cv,
        )
        study = optuna.create_study(
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = dict(study.best_params)
        best_score = float(study.best_value)  # RMSE
        logger.info(
            "HPO complete for %s: best RMSE=%.4f, params=%s",
            model_type,
            best_score,
            best_params,
        )

        # Re-instantiate the best model without the forced device override so
        # the final model can use whatever compute the user configured.
        best_model = get_model(model_type, params=best_params, seed=self.seed)

        # Explicit garbage collection after each study to release trial
        # objects and any lingering model references promptly.
        gc.collect()

        meta = {
            "enabled": True,
            "n_trials": n_trials,
            "cv": cv,
            "best_value_rmse": best_score,
        }

        return best_model, best_params, best_score, meta
