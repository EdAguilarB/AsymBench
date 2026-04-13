import hashlib
import itertools
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

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


def _result_rep_label(rep_cfg: dict) -> str:
    """Return the human-readable label for a representation config.

    Priority:
      1. ``rep_cfg["name"]`` — user-supplied override (free-form string).
         Allows distinct configs of the same ``type`` to be told apart in
         results, logs, cache filenames, and run directory paths.
      2. Auto-label from ``type`` + key params (existing fallback logic):
         * ``hf_transformer`` — append ``model_type``
           (e.g. ``"hf_transformer_chemberta"``)
         * ``df_lookup`` / ``bespoke`` / ``precomputed`` — append
           ``feature_name`` (e.g. ``"bespoke_v1"``)
         * All other types — use ``type`` as-is (e.g. ``"morgan"``)

    This label is the single source of truth used in:
      - results JSON (``"representation"`` key)
      - disk-cache filenames under ``log_dirs.benchmark/representations/``
      - run directory paths (via ``_run_signature()``)
    """
    if rep_cfg.get("name"):
        return rep_cfg["name"]

    label = rep_cfg["type"]
    params = rep_cfg.get("params", {})

    if label in ("df_lookup", "bespoke", "precomputed"):
        feature_name = params.get("feature_name", "")
        if feature_name:
            label = f"{label}_{feature_name}"
    elif label == "hf_transformer":
        model_type = params.get("model_type", "")
        if model_type:
            label = f"{label}_{model_type}"

    return label


class BenchmarkRunner:
    def __init__(self, config):
        self.config = config

        # Load dataset once
        self.dataset = load_dataset(config["dataset"])
        if config.get("external_test_set", False):
            self.external_test = load_dataset(config["external_test_set"])
        else:
            self.external_test = None

        # Validate that no two representation configs resolve to the same label.
        # This catches both missing ``name:`` fields on duplicate-type configs
        # and accidental name collisions — before any expensive work begins.
        self._validate_rep_labels()

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

    def _validate_rep_labels(self) -> None:
        """Raise ``ValueError`` if any two rep configs resolve to the same label.

        Checked at startup so the user gets a clear error before any
        expensive computation begins.  Works for both auto-generated labels
        and user-supplied ``name:`` values.
        """
        seen: dict[str, dict] = {}
        for rep_cfg in self.config["representations"]:
            label = _result_rep_label(rep_cfg)
            if label in seen:
                raise ValueError(
                    f"Two representation configs resolve to the same label "
                    f"'{label}'. Add a unique 'name:' field to each one.\n"
                    f"  First:  {seen[label]}\n"
                    f"  Second: {rep_cfg}"
                )
            seen[label] = rep_cfg

    def _rep_key(self, rep_cfg: dict) -> str:
        """Stable JSON key for a representation config dict.

        The ``name`` field is intentionally excluded: two configs that differ
        only by name represent the same featurization and must share cached
        in-memory features.
        """
        key_cfg = {k: v for k, v in rep_cfg.items() if k != "name"}
        return json.dumps(key_cfg, sort_keys=True)

    def _rep_cache_path(self, rep_cfg: dict) -> Path:
        """Return the CSV path for a disk-cached representation.

        The filename is ``<human_label>_<hash>.csv`` where the hash covers
        ``type`` + ``params`` + reaction features + dataset path.  The
        ``name`` field is **excluded** from the hash so that renaming a
        representation does not invalidate its on-disk feature cache.
        """
        rxn_feats = sorted(self.config["dataset"].get("reaction_features", []))
        # Exclude 'name' so renames don't bust the on-disk cache
        rep_for_hash = {k: v for k, v in rep_cfg.items() if k != "name"}
        cache_key_str = json.dumps(
            {
                "rep": rep_for_hash,
                "reaction_features": rxn_feats,
                "dataset_path": self.config["dataset"]["path"],
            },
            sort_keys=True,
        )
        hash_suffix = hashlib.md5(cache_key_str.encode()).hexdigest()[:8]
        label = _result_rep_label(rep_cfg)  # uses name if present
        cache_dir = (
            Path(self.config["log_dirs"]["benchmark"]) / "representations"
        )
        return cache_dir / f"{label}_{hash_suffix}.csv"

    def _precompute_fit_free_representations(self) -> None:
        """Compute features for all fit-free representations once on the full dataset.

        Representations that expose a ``fit()`` method (e.g. CircusFeaturizer)
        require a training set and are skipped here — they are handled
        per-experiment via ``CachingCircusRepresentation``.

        Each computed matrix is persisted to a CSV file under
        ``log_dirs.benchmark/representations/`` so that subsequent runs can
        skip the (potentially expensive) featurization and load from disk
        instead.
        """
        if self.external_test is not None:
            full_data = pd.concat([self.dataset, self.external_test])
        else:
            full_data = self.dataset

        for rep_cfg in self.config["representations"]:
            key = self._rep_key(rep_cfg)
            if key in self._precomputed:
                continue  # already loaded this run (e.g. duplicate config entries)

            if rep_cfg.get("type") == "graph":
                continue  # graph reps are built per-split inside GNNExperiment

            # ── Disk-cache check FIRST ────────────────────────────────────
            # This must happen before get_representation() so that heavy
            # featurizers (Unimol, HF transformers) never load their model
            # or generate conformers when the result is already on disk.
            disk_path = self._rep_cache_path(rep_cfg)
            if disk_path.exists():
                logger.info(
                    "Loading cached representation from disk: %s",
                    disk_path.name,
                )
                self._precomputed[key] = pd.read_csv(disk_path, index_col=0)
                continue

            # ── Not cached — instantiate, check trainability, compute ─────
            rep_config = {
                "representation": rep_cfg,
                "data": self.config["dataset"],
            }
            rep = get_representation(rep_config)

            if hasattr(rep, "fit"):
                continue  # trainable representation — handled per-experiment

            logger.info(
                "Pre-computing representation: %s ...",
                _result_rep_label(rep_cfg),
            )
            X = rep.transform(full_data)
            # Embed reaction-condition columns so index-lookups return them
            # for free — no per-experiment concatenation needed.
            rxn_feats = self.config["dataset"].get("reaction_features", [])
            if rxn_feats:
                X = pd.concat([X, full_data[rxn_feats]], axis=1)
            disk_path.parent.mkdir(parents=True, exist_ok=True)
            X.to_csv(disk_path)
            logger.info("Representation cached to disk: %s", disk_path.name)
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
                "representation": _result_rep_label(rep_cfg),
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
        preprocessing = FeaturePreprocessor(pre_cfg)
        y_scaling = TargetScaler(y_scl_cfg["scaling"])

        expl_cfg = expl_cfg or {"enabled": False}

        # Resolve the label once here — passed to both experiment types so
        # that run directory paths and results JSON are always consistent.
        rep_label = _result_rep_label(rep_cfg)

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
                preprocessing=preprocessing,
                y_scaling=y_scaling,
                split_strategy=split_strategy,
                seed=seed,
                cache_dir=self.config["log_dirs"]["runs"],
                external_test_set=self.external_test,
                explainability_cfg=expl_cfg,
                reaction_features=self.config["dataset"].get(
                    "reaction_features", []
                ),
                rep_label=rep_label,
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
            rep_label=rep_label,
        )

    def _save_results(self, results):
        log_dir = self.config["log_dirs"]["benchmark"]
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{log_dir}/raw_results.json", "w") as f:
            json.dump(results, f, indent=2)
