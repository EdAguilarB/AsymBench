from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


def _ensure_df(X, feature_names=None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    X = np.asarray(X)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=list(feature_names))


@dataclass
class ShapExplainer:
    """
    SHAP explanation helper that:
      - chooses a sensible explainer (TreeExplainer for tree models when possible)
      - computes global and local explanations
      - saves publication-ready SHAP plots

    Parameters
    ----------
    max_background : int
        Max samples used as background for KernelExplainer / generic Explainer.
    max_explain : int
        Max samples to compute SHAP values for (to keep runtime manageable).
    """

    max_background: int = 200
    max_explain: int = 500
    seed: int = 0

    explainer_: Optional[shap.Explainer] = None

    def fit(self, model: Any, X_train: pd.DataFrame) -> "ShapExplainer":
        X_train = _ensure_df(X_train)

        bg = X_train
        if len(X_train) > self.max_background:
            bg = X_train.sample(self.max_background, random_state=self.seed)

        # Tree-based models: use TreeExplainer
        if hasattr(model, "get_booster") or model.__class__.__name__ in (
            "RandomForestRegressor",
            "ExtraTreesRegressor",
            "GradientBoostingRegressor",
            "XGBRegressor",
        ):
            self.explainer_ = shap.TreeExplainer(model)
            self._uses_tree_explainer = True
            self._background = None
            return self

        # Otherwise: use generic Explainer with a callable predict function + background masker
        self.explainer_ = shap.Explainer(model.predict, bg)
        self._uses_tree_explainer = False
        self._background = bg
        return self

    def explain(
        self,
        X: pd.DataFrame,
        outdir: Path,
        prefix: str = "train",
        make_plots: bool = True,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        if self.explainer_ is None:
            raise RuntimeError("ShapExplainer.explain called before fit().")

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        X = _ensure_df(X)

        # limit number of samples for speed
        X_eval = X
        if len(X) > self.max_explain:
            X_eval = X.sample(self.max_explain, random_state=self.seed)

        if self._uses_tree_explainer:
            shap_values = self.explainer_(X_eval)  # works for trees
        else:
            shap_values = self.explainer_(
                X_eval
            )  # callable path works for SVR/MLP

        # ---- global importance table ----
        # mean absolute shap per feature
        vals = np.asarray(shap_values.values)
        if vals.ndim == 3:
            # multioutput: pick first output (or aggregate)
            vals = vals[:, :, 0]
        mean_abs = np.mean(np.abs(vals), axis=0)

        importance = (
            pd.DataFrame(
                {"feature": X_eval.columns, "mean_abs_shap": mean_abs}
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        importance_path = outdir / f"{prefix}_shap_importance.csv"
        importance.to_csv(importance_path, index=False)

        artifacts: Dict[str, Any] = {
            "n_explained": int(len(X_eval)),
            "importance_csv": str(importance_path),
        }

        # ---- plots ----
        if make_plots:
            # (1) beeswarm summary (global)
            fig = plt.figure()
            shap.summary_plot(
                shap_values, X_eval, show=False, max_display=top_k
            )
            p = outdir / f"{prefix}_shap_summary_beeswarm.png"
            plt.tight_layout()
            plt.savefig(p, dpi=350, bbox_inches="tight")
            plt.close(fig)
            artifacts["summary_beeswarm_png"] = str(p)

        return artifacts

    def explain_single(
        self,
        X_row: pd.DataFrame,
        outdir: Path,
        name: str = "sample0",
        top_k: int = 15,
    ) -> Dict[str, Any]:
        """
        Local explanation for one sample (waterfall plot).
        """
        if self.explainer_ is None:
            raise RuntimeError(
                "ShapExplainer.explain_single called before fit()."
            )

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        X_row = _ensure_df(X_row)
        if len(X_row) != 1:
            raise ValueError("X_row must contain exactly one row.")

        sv = self.explainer_(X_row)

        artifacts: Dict[str, Any] = {}

        # waterfall plot (best for papers)
        fig = plt.figure()
        shap.plots.waterfall(sv[0], max_display=top_k, show=False)
        p = outdir / f"shap_waterfall_{name}.png"
        plt.tight_layout()
        plt.savefig(p, dpi=350, bbox_inches="tight")
        plt.close(fig)
        artifacts["waterfall_png"] = str(p)

        return artifacts
