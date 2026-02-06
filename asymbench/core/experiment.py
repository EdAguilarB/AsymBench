import pandas as pd

from asymbench.data.splitter import MoleculeSplitter
from asymbench.preprocessing.feature_preprocessor import FeaturePreprocessor
from asymbench.preprocessing.targets_scaler import TargetScaler
from asymbench.representations.base_featurizer import BaseSmilesFeaturizer
from asymbench.optimization.optuna_optimizer import OptunaSklearnOptimizer
from asymbench.evaluation.metrics import evaluate_predictions
from asymbench.visualization.parity import make_parity_plot
from asymbench.utils.run_paths import build_run_dir


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

    def run(self):

        # 1) Get relevant data from dataframe
        split_by_mols = self.dataset.loc[:, self.split_by_mol_col]
        y = self.dataset.loc[:, self.target]

        # 2) Create representation
        X = self.representation.transform(self.dataset)

        # 3) split the data
        _, _, train_idxs, test_idxs = self.split_strategy.get_splits(
            mols=split_by_mols, return_indices=True
        )
        X_train, y_train = X.iloc[train_idxs], y.iloc[train_idxs]
        X_test, y_test = X.iloc[test_idxs], y.iloc[test_idxs]

        # 4) preprocess the data
        X_train = self.preprocessing.fit_transform(X_train)
        X_test = self.preprocessing.transform(X_test)
        y_train = self.y_scaling.fit_transform(y_train)
        y_test = self.y_scaling.transform(y_test)

        # 5)
        best_model, best_params, best_cv_score, hpo_meta = self.optimizer.optimize(
            X_train, y_train
        )

        # 4) Fit best model on full train
        best_model.fit(X_train, y_train)
        preds_test = best_model.predict(X_test)

        y_test = self.y_scaling.inverse_transform(y_test)
        preds_test = self.y_scaling.inverse_transform(preds_test)

        metrics = evaluate_predictions(y_test, preds_test)

        run_dir = self._get_save_dir_path(best_model, X_train)
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

        return metrics

    def _get_save_dir_path(self, best_model, X_train):
        run_dir = build_run_dir(
            base_dir="experiments/plots",
            dataset_name=(
                self.config_dataset_name
                if hasattr(self, "config_dataset_name")
                else "dataset"
            ),
            rep_type=(
                self.representation.rep_type
                if hasattr(self.representation, "rep_type")
                else type(self.representation).__name__
            ),
            model_type=getattr(best_model, "__class__", type(best_model)).__name__,
            split_type=(
                self.split_strategy.config.get("type", "split")
                if hasattr(self.split_strategy, "config")
                else "split"
            ),
            split_by_mol_col=self.split_by_mol_col,
            train_size=(
                self.split_strategy.train_size
                if hasattr(self.split_strategy, "train_size")
                else "na"
            ),
            seed=self.seed,
            extra={
                "y_scaling": getattr(self.y_scaling, "scaling", "none"),
                "nfeat": X_train.shape[1],
            },
        )

        return run_dir
