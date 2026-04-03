from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from asymbench.data.splitter import MoleculeSplitter
from asymbench.evaluation.metrics import evaluate_predictions
from asymbench.gnn.featurizer import NODE_FEAT_DIM
from asymbench.gnn.architectures import BaseReactionGNN, build_reaction_gnn
from asymbench.gnn.trainer import predict, train_epoch
from asymbench.preprocessing.targets_scaler import TargetScaler
from asymbench.utils.run_store import RunStore
from asymbench.visualization.parity import make_parity_plot


class GNNExperiment:
    """Single GNN training/evaluation run.

    Mirrors the interface of :class:`~asymbench.core.experiment.Experiment`
    but replaces the sklearn pipeline with a PyTorch Geometric training loop.

    Parameters
    ----------
    dataset:
        Full reaction DataFrame (all splits).
    smiles_columns:
        Ordered list of SMILES column names.
    target:
        Name of the regression target column.
    split_by_mol_col:
        Column used as the *molecules* argument to the splitter.
    representation:
        A :class:`~asymbench.representations.graph.GraphRepresentation`
        instance.  ``representation.transform(df)`` must return a
        :class:`~asymbench.gnn.dataset.ReactionGraphDataset`.
    model_cfg:
        Model config dict from YAML, e.g.::

            type: gnn
            params:
              hidden_dim: 64
              num_layers: 3
              epochs: 100
              lr: 0.001
              batch_size: 32
              pooling: mean

    y_scl_cfg:
        Target scaling config, e.g. ``{"scaling": "standard"}``.
    split_strategy:
        A :class:`~asymbench.data.splitter.MoleculeSplitter` instance.
    seed:
        Random seed (used for run signature only; PyTorch seeding is
        left to the caller for now).
    cache_dir:
        Root directory for run artefacts.
    external_test_set:
        Optional external test DataFrame (passed straight through to the
        splitter).
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        smiles_columns: list,
        target: str,
        split_by_mol_col: str,
        representation,
        model_cfg: dict,
        y_scl_cfg: dict,
        split_strategy: MoleculeSplitter,
        seed: int,
        cache_dir: Path = Path("experiment_runs"),
        external_test_set: pd.DataFrame | None = None,
        explainability_cfg: dict | None = None,
    ) -> None:
        self.dataset = dataset
        self.smiles_columns = smiles_columns
        self.target = target
        self.split_by_mol_col = split_by_mol_col
        self.representation = representation
        self.model_cfg = model_cfg
        self.model_params: dict = dict(model_cfg.get("params", {}))
        self.y_scaling = TargetScaler(y_scl_cfg.get("scaling", "none"))
        self.split_strategy = split_strategy
        self.seed = seed
        self.cache_dir = Path(cache_dir)
        self.run_store = RunStore(base_dir=self.cache_dir)
        self.external_test_set = external_test_set
        self.explainability_cfg: dict = explainability_cfg or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> dict:
        signature = self._run_signature()
        print(f"Running GNN experiment:\n{signature}")

        if self.run_store.exists_complete(signature):
            metrics = self.run_store.load_metrics(signature)
            metrics["cache_hit"] = True
            return metrics

        self.run_store.mark_started(signature)
        run_dir = self.run_store.run_dir_from_signature(signature)

        try:
            df_train, df_test, y_train, y_test = self._split_data()

            y_train_scaled = self.y_scaling.fit_transform(y_train)
            y_test_scaled = self.y_scaling.transform(y_test)

            train_loader = self._make_loader(df_train, y_train_scaled, shuffle=True)
            test_loader = self._make_loader(df_test, y_test_scaled, shuffle=False)

            model = self._build_model()
            self._train(model, train_loader)

            if self.explainability_cfg.get("enabled", False):
                self._explain(model, train_loader, test_loader, run_dir)

            preds_train = self.y_scaling.inverse_transform(
                predict(model, train_loader, self.device)
            )
            preds_test = self.y_scaling.inverse_transform(
                predict(model, test_loader, self.device)
            )

            y_train_orig = y_train.to_numpy()
            y_test_orig = y_test.to_numpy()

            metrics = self._build_metrics(y_test_orig, preds_test)
            self._make_parity_plot(y_test_orig, preds_test, metrics, run_dir)
            self._save_predictions(
                df_train, y_train_orig, preds_train,
                df_test, y_test_orig, preds_test,
                run_dir,
            )
            self.run_store.mark_completed(signature, metrics)
            return metrics

        except Exception as exc:
            self.run_store.mark_failed(signature, exc)
            raise

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _split_data(self):
        split_by_mols = self.dataset.loc[:, self.split_by_mol_col]
        y = self.dataset.loc[:, self.target]
        return self.split_strategy.get_train_test_set(
            data=self.dataset,
            mols=split_by_mols,
            y=y,
            external_test=self.external_test_set,
        )

    def _make_loader(
        self, df: pd.DataFrame, y_scaled: np.ndarray, *, shuffle: bool
    ) -> DataLoader:
        """Build a PyG DataLoader with scaled targets injected into each graph."""
        dataset = self.representation.transform(df)
        for i, data in enumerate(dataset._data):
            data.y = torch.tensor([y_scaled[i]], dtype=torch.float)
        batch_size = self.model_params.get("batch_size", 32)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _build_model(self) -> BaseReactionGNN:
        # Training-loop params are not forwarded to the model constructor
        _TRAINING_KEYS = {"epochs", "lr", "batch_size"}
        arch_params = {
            k: v
            for k, v in self.model_params.items()
            if k not in _TRAINING_KEYS
        }
        return build_reaction_gnn(
            node_in_dim=NODE_FEAT_DIM,
            **arch_params,
        ).to(self.device)

    def _train(self, model: BaseReactionGNN, loader: DataLoader) -> None:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.model_params.get("lr", 1e-3),
        )
        epochs = self.model_params.get("epochs", 100)
        for epoch in range(epochs):
            loss = train_epoch(model, loader, optimizer, self.device)
            if (epoch + 1) % 10 == 0:
                print(f"  epoch {epoch + 1:>4}/{epochs}  loss={loss:.4f}")

    def _explain(
        self,
        model: BaseReactionGNN,
        train_loader: DataLoader,
        test_loader: DataLoader,
        run_dir: Path,
    ) -> None:
        """Run CaptumExplainer and save node masks + fragment importances."""
        from asymbench.explainability.gnn_explainer import (
            GNNExplainer,
            get_fragment_importance,
        )

        expl_cfg = self.explainability_cfg
        explainer = GNNExplainer(
            model=model,
            n_steps=expl_cfg.get("n_steps", 50),
            fragmentation=expl_cfg.get("fragmentation", "brics"),
        ).fit()

        for prefix, loader in [("train", train_loader), ("test", test_loader)]:
            outdir = run_dir / "explainability" / prefix
            dataset = loader.dataset
            node_masks = explainer.explain_dataset(dataset, self.device)
            frag_importances = get_fragment_importance(
                dataset._data,
                node_masks,
                explainer.fragmentation,
            )
            explainer.save(node_masks, frag_importances, outdir)

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    @property
    def arch_label(self) -> str:
        """Specific architecture name used for logging and result storage.

        Returns the ``architecture`` param (e.g. ``"gcn"``, ``"gat"``,
        ``"gin"``), falling back to ``"gcn"`` so existing configs that
        omit the key continue to work.
        """
        return self.model_params.get("architecture", "gcn")

    def _build_metrics(self, y_test: np.ndarray, preds_test: np.ndarray) -> dict:
        metrics = evaluate_predictions(y_test, preds_test)
        split_cfg = getattr(self.split_strategy, "config", {})
        metrics.update(
            {
                "cache_hit": False,
                "model_type": self.arch_label,
                "rep_type": "graph",
                "split_sampler": split_cfg.get("sampler"),
                "train_size": split_cfg.get("train_size"),
                "seed": self.seed,
                "model_params": self.model_params,
            }
        )
        return metrics

    def _make_parity_plot(
        self,
        y_test: np.ndarray,
        preds_test: np.ndarray,
        metrics: dict,
        run_dir: Path,
    ) -> None:
        make_parity_plot(
            y_true=y_test,
            y_pred=preds_test,
            metrics=metrics,
            title=f"Parity plot ({self.target})",
            subtitle=None,
            outpath=run_dir / "parity_test.png",
            dpi=350,
        )

    def _save_predictions(
        self,
        df_train: pd.DataFrame,
        y_train: np.ndarray,
        preds_train: np.ndarray,
        df_test: pd.DataFrame,
        y_test: np.ndarray,
        preds_test: np.ndarray,
        run_dir: Path,
    ) -> None:
        train_df = pd.DataFrame(
            {
                "split": "train",
                self.target: y_train,
                f"{self.target}_pred": preds_train,
            },
            index=df_train.index,
        )
        test_df = pd.DataFrame(
            {
                "split": "test",
                self.target: y_test,
                f"{self.target}_pred": preds_test,
            },
            index=df_test.index,
        )
        pd.concat([train_df, test_df]).to_csv(run_dir / "predictions.csv", index=True)

    def _run_signature(self) -> dict:
        split_cfg = getattr(self.split_strategy, "config", {})
        return {
            "representation": {"type": "graph"},
            "model": {"type": self.arch_label},
            "split": {
                "sampler": split_cfg.get("sampler", "unknown"),
                "train_size": split_cfg.get("train_size"),
                "split_by_mol_col": self.split_by_mol_col,
            },
            "seed": self.seed,
        }
