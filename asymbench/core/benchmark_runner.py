import itertools
import json
from pathlib import Path

import pandas as pd

from asymbench.core.experiment import Experiment
from asymbench.core.gnn_experiment import GNNExperiment
from asymbench.data.loader import load_dataset
from asymbench.data.splitter import MoleculeSplitter
from asymbench.optimization.base import get_optimizer
from asymbench.preprocessing.feature_preprocessor import FeaturePreprocessor
from asymbench.preprocessing.targets_scaler import TargetScaler
from asymbench.representations import get_representation
from asymbench.representations.circus import CachingCircusRepresentation
from asymbench.representations.graph import GraphRepresentation
from asymbench.representations.lookup import PrecomputedRepresentation

_GNN_MODEL_TYPES = {"gnn"}
_SKLEARN_INCOMPATIBLE_REPS = {"graph"}


class BenchmarkRunner:
    def __init__(self, config):
        self.config = config

        # Load dataset once
        self.dataset = load_dataset(config["dataset"])
        if config.get("external_test_set", False):
            self.external_test = load_dataset(config["external_test_set"])
        else:
            self.external_test = None

        # Pre-compute fit-free representations on the full dataset so that
        # each experiment only does an O(n) index lookup instead of re-running
        # (potentially expensive) featurization from scratch.
        self._precomputed: dict[str, pd.DataFrame] = {}
        # Shared fit-cache for CircusFeaturizer: keyed by (sorted training
        # indices, rep config JSON) so the same training set is never fit twice.
        self._circus_fit_cache: dict = {}
        self._precompute_fit_free_representations()

    # ------------------------------------------------------------------
    # Representation pre-computation helpers
    # ------------------------------------------------------------------

    def _rep_key(self, rep_cfg: dict) -> str:
        """Stable JSON key for a representation config dict."""
        return json.dumps(rep_cfg, sort_keys=True)

    def _precompute_fit_free_representations(self) -> None:
        """Compute features for all fit-free representations once on the full dataset.

        Representations that expose a ``fit()`` method (e.g. CircusFeaturizer)
        require a training set and are skipped here — they are handled
        per-experiment via ``CachingCircusRepresentation``.
        """
        # Combine training data and external test so that
        # PrecomputedRepresentation.transform() works for any split.
        if self.external_test is not None:
            full_data = pd.concat([self.dataset, self.external_test])
        else:
            full_data = self.dataset

        for rep_cfg in self.config["representations"]:
            key = self._rep_key(rep_cfg)
            if key in self._precomputed:
                continue  # already computed (e.g. duplicate config entries)

            rep_config = {
                "representation": rep_cfg,
                "data": self.config["dataset"],
            }
            rep = get_representation(rep_config)

            if rep_cfg.get("type") == "graph":
                continue  # graph reps are built per-split inside GNNExperiment

            if hasattr(rep, "fit"):
                continue  # trainable representation — handled per-experiment

            print(f"Pre-computing representation: {rep_cfg['type']} ...")
            X = rep.transform(full_data)
            # Embed reaction-condition columns once so that every subsequent
            # index-lookup via PrecomputedRepresentation.transform() returns
            # them for free — no per-experiment concatenation needed.
            rxn_feats = self.config["dataset"].get("reaction_features", [])
            if rxn_feats:
                X = pd.concat([X, full_data[rxn_feats]], axis=1)
            self._precomputed[key] = X

    def run(self):
        results = []

        combos = itertools.product(
            self.config["representations"],
            self.config["models"],
            self.config["splits"],
            self.config["train_set_sizes"],
            self.config["split_by_mol_col"],
            self.config["seeds"],
        )
        pre_cfg = self.config["preprocessing"]
        y_scl_cfg = self.config["target_scaling"]
        expl_cfg = self.config.get("explainability", {"enabled": False})

        for (
            rep_cfg,
            model_cfg,
            split_cfg,
            train_set_size,
            split_by_mol_col,
            seed,
        ) in combos:

            # Skip incompatible representation/model pairings.
            is_graph_rep = rep_cfg.get("type") in _SKLEARN_INCOMPATIBLE_REPS
            is_gnn_model = model_cfg.get("type") in _GNN_MODEL_TYPES
            if is_graph_rep != is_gnn_model:
                continue

            split_cfg["train_size"] = train_set_size

            exp = self._build_experiment(
                rep_cfg=rep_cfg,
                pre_cfg=pre_cfg,
                y_scl_cfg=y_scl_cfg,
                model_cfg=model_cfg,
                split_cfg=split_cfg,
                split_by_mol_col=split_by_mol_col,
                seed=seed,
                expl_cfg=expl_cfg,
            )
            metrics = exp.run()

            result = {
                "representation": rep_cfg["type"],
                # For GNN models, model_type in metrics is the specific
                # architecture (gcn/gat/gin); for sklearn models it is the
                # model class name.  Prefer that over the raw YAML "type" key
                # so different GNN architectures are distinguishable in results.
                "model": metrics.get("model_type", model_cfg["type"]),
                "split": split_cfg["sampler"],
                "seed": seed,
                **metrics,
            }

            results.append(result)

        self._save_results(results)
        return results

    def _build_experiment(
        self,
        rep_cfg,
        pre_cfg,
        y_scl_cfg,
        model_cfg,
        split_cfg,
        split_by_mol_col,
        seed,
        expl_cfg=None,
    ):
        rep_config = {
            "representation": rep_cfg,
            "data": self.config["dataset"],
        }
        split_strategy = MoleculeSplitter(split_cfg)

        expl_cfg = expl_cfg or {"enabled": False}

        # --- GNN path ---
        if rep_cfg.get("type") == "graph":
            representation = GraphRepresentation(rep_config)
            return GNNExperiment(
                dataset=self.dataset,
                smiles_columns=self.config["dataset"]["smiles_columns"],
                target=self.config["dataset"]["target"],
                split_by_mol_col=split_by_mol_col,
                representation=representation,
                model_cfg=model_cfg,
                y_scl_cfg=y_scl_cfg,
                split_strategy=split_strategy,
                seed=seed,
                cache_dir=self.config["log_dirs"]["runs"],
                external_test_set=self.external_test,
                explainability_cfg=expl_cfg,
            )

        # --- sklearn path ---
        key = self._rep_key(rep_cfg)
        if key in self._precomputed:
            # Fit-free: wrap pre-computed features — no molecular computation at run time.
            representation = PrecomputedRepresentation(
                config=rep_config, _features=self._precomputed[key]
            )
        else:
            # Trainable (e.g. CircusFeaturizer): inject the shared fit-cache so
            # the same training set is never fit() more than once.
            representation = CachingCircusRepresentation(
                config=rep_config,
                fit_cache=self._circus_fit_cache,
                reaction_features=self.config["dataset"].get(
                    "reaction_features", []
                ),
            )

        preprocessing = FeaturePreprocessor(pre_cfg)
        y_scaling = TargetScaler(y_scl_cfg["scaling"])
        optimizer = get_optimizer(model_cfg, seed)

        return Experiment(
            dataset=self.dataset,
            smiles_columns=self.config["dataset"]["smiles_columns"],
            target=self.config["dataset"]["target"],
            split_by_mol_col=split_by_mol_col,
            representation=representation,
            preprocessing=preprocessing,
            y_scaling=y_scaling,
            optimizer=optimizer,
            split_strategy=split_strategy,
            seed=seed,
            cache_dir=self.config["log_dirs"]["runs"],
            external_test_set=self.external_test,
            explainability_cfg=expl_cfg,
            reaction_features=self.config["dataset"].get(
                "reaction_features", []
            ),
        )

    def _save_results(self, results):
        log_dir = self.config["log_dirs"]["benchmark"]
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{log_dir}/raw_results.json", "w") as f:
            json.dump(results, f, indent=2)
