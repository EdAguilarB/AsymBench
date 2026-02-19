from pathlib import Path

import pandas as pd

from asymbench.data.splitter import MoleculeSplitter
from asymbench.evaluation.metrics import evaluate_predictions
from asymbench.explainability.shap_explainer import ShapExplainer
from asymbench.optimization.optuna_optimizer import OptunaSklearnOptimizer
from asymbench.preprocessing.feature_preprocessor import FeaturePreprocessor
from asymbench.preprocessing.targets_scaler import TargetScaler
from asymbench.representations.base_featurizer import BaseSmilesFeaturizer
from asymbench.utils.feature_names import FeatureNameSanitizer
from asymbench.utils.run_store import RunStore
from asymbench.visualization.parity import make_parity_plot


class Experiment:
    def __init__(
        self,
        dataset: pd.DataFrame,
        smiles_columns: list,
        target: str,
        split_by_mol_col: str,
        representation: BaseSmilesFeaturizer,
        preprocessing: FeaturePreprocessor,
        y_scaling: TargetScaler,
        optimizer: OptunaSklearnOptimizer,
        split_strategy: MoleculeSplitter,
        seed: int,
        cache_dir: Path = Path("experiment_runs"),
        external_test_set: pd.DataFrame | None = None,
    ):
        self.dataset = dataset
        self.smiles_columns = smiles_columns
        self.target = target
        self.split_by_mol_col = split_by_mol_col
        self.representation = representation
        self.preprocessing = preprocessing
        self.y_scaling = y_scaling
        self.optimizer = optimizer
        self.split_strategy = split_strategy
        self.seed = seed
        self.cache_dir = cache_dir
        self.run_store = RunStore(base_dir=self.cache_dir)
        self.model_cfg = getattr(self.optimizer, "model_cfg", None)
        self.external_test_set = external_test_set

    def run(self):

        model_type = self.model_cfg.get("type", None)

        signature = self._run_signature(model_type=model_type)
        print(
            f"Running experiment with the following configuration:\n{signature}"
        )
        if self.run_store.exists_complete(signature):
            metrics = self.run_store.load_metrics(signature)
            metrics["cache_hit"] = True
            return metrics

        self.run_store.mark_started(signature)
        run_dir = self.run_store.run_dir_from_signature(signature)

        # 1) Mark started (writes signature + status)
        self.run_store.mark_started(signature)
        run_dir = self.run_store.run_dir_from_signature(signature)

        try:
            # 2) Get relevant data from dataframe
            split_by_mols = self.dataset.loc[:, self.split_by_mol_col]
            y = self.dataset.loc[:, self.target]

            # 3) split the data
            df_train, df_test, y_train, y_test = self.split_strategy.get_train_test_set(
                data=self.dataset, mols=split_by_mols, y=y, external_test=self.external_test_set
            )

            # 4) Create representation
            X_train, X_test = self._fit_transform_representation(
                df_train, df_test
            )

            # 5) preprocess the data
            X_train = self.preprocessing.fit_transform(X_train)
            X_test = self.preprocessing.transform(X_test)
            name_sanitizer = FeatureNameSanitizer()
            X_train = name_sanitizer.fit_transform(X_train)
            X_test = name_sanitizer.transform(X_test)

            y_train = self.y_scaling.fit_transform(y_train)
            y_test = self.y_scaling.transform(y_test)

            # 6) HPO on train
            best_model, best_params, best_cv_score, hpo_meta = (
                self.optimizer.optimize(X_train, y_train)
            )

            # 7) Fit best model on train and predict on test
            best_model.fit(X_train, y_train)
            preds_test = best_model.predict(X_test)

            expl_dir = run_dir / "explainability"
            shapx = ShapExplainer(max_background=200, max_explain=500, seed=self.seed).fit(best_model, X_train)
            shap_artifacts_train = shapx.explain(X_train, outdir=expl_dir, prefix="train")
            shap_artifacts_test  = shapx.explain(X_test,  outdir=expl_dir, prefix="test")

            # 8) inverse transform
            y_test = self.y_scaling.inverse_transform(y_test)
            preds_test = self.y_scaling.inverse_transform(preds_test)

            results = pd.DataFrame({
                X_test.index.name: X_test.index,
                "y": y_test,
                "y_pred": preds_test
            }).to_csv(run_dir / "preds.csv", index=False)

            metrics = evaluate_predictions(y_test, preds_test)

            # 10) Add important metadata for analysis
            metrics.update(
                {
                    # "run_id": run_id,
                    "cache_hit": False,
                    "model_type": model_type,
                    "rep_type": getattr(
                        self.representation,
                        "rep_type",
                        type(self.representation).__name__,
                    ),
                    "split_sampler": getattr(
                        self.split_strategy, "config", {}
                    ).get("sampler"),
                    "train_size": getattr(
                        self.split_strategy, "config", {}
                    ).get("train_size"),
                    "seed": self.seed,
                    "best_params": best_params,
                    "best_cv_score": (
                        float(best_cv_score)
                        if best_cv_score is not None
                        else None
                    ),
                    "hpo_meta": hpo_meta,
                    "n_features": int(X_train.shape[1]),
                }
            )

            plot_path = run_dir / "parity_test.png"
            make_parity_plot(
                y_true=y_test,
                y_pred=preds_test,
                metrics=metrics,
                title=f"Parity plot ({self.target})",
                subtitle=None,
                outpath=plot_path,
                dpi=350,
            )

            # 12) Mark completed (writes metrics + status)
            self.run_store.mark_completed(signature, metrics)
            return metrics

        except Exception as e:
            # Save failure info for post-mortem
            self.run_store.mark_failed(signature, e)
            raise

    def _fit_transform_representation(self, df_train, df_test):
        rep = self.representation
        if hasattr(rep, "fit"):
            rep.fit(df_train)
        X_train = rep.transform(df_train)
        X_test = rep.transform(df_test)
        return X_train, X_test

    def _run_signature(self, model_type: str) -> dict:
        rep_type = getattr(
            self.representation, "rep_type", type(self.representation).__name__
        )

        # df_lookup feature columns (if applicable)
        rep_params = getattr(self.representation, "rep_params", {})
        feature_name = None
        if rep_type in ("df_lookup", "bespoke", "precomputed"):
            feature_name = rep_params.get("feature_name", None)
            rep_type = rep_type + "_" + feature_name

        # split sampler + train size
        split_cfg = getattr(self.split_strategy, "config", {})
        sampler = split_cfg.get("sampler", split_cfg.get("type", "unknown"))
        train_size = split_cfg.get("train_size", None)

        return {
            "representation": {"type": rep_type},
            "model": {"type": model_type},
            "split": {
                "sampler": sampler,
                "train_size": train_size,
                "split_by_mol_col": self.split_by_mol_col,
            },
            "seed": self.seed,
        }
