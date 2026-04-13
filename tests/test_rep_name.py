"""Tests for the user-configurable representation ``name`` field.

Covers:
  - _result_rep_label() with and without ``name``
  - _rep_key() excludes ``name`` from the cache key
  - Disk-cache hash excludes ``name`` (rename-safe)
  - Duplicate label validation in BenchmarkRunner.__init__()
  - Experiment._run_signature() honours rep_label
  - GNNExperiment._run_signature() honours rep_label
"""

from __future__ import annotations

import hashlib
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from asymbench.core.benchmark_runner import BenchmarkRunner, _result_rep_label
from asymbench.core.experiment import Experiment
from asymbench.core.gnn_experiment import GNNExperiment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_CFG = {
    "dataset": {
        "path": "dummy.csv",
        "smiles_columns": ["smiles"],
        "target": "y",
    },
    "external_test_set": False,
    "representations": [],
    "models": [],
    "splits": [],
    "train_set_sizes": [0.8],
    "split_by_mol_col": ["smiles"],
    "seeds": [0],
    "preprocessing": {
        "scaling": "none",
        "feature_selection": {
            "variance_filter": {"enabled": False},
            "correlation_filter": {"enabled": False},
        },
    },
    "target_scaling": {"scaling": "none"},
    "log_dirs": {"runs": "/tmp/runs", "benchmark": "/tmp/benchmark"},
}

_FAKE_DF = pd.DataFrame({"smiles": ["C"], "y": [1.0]})


def _make_runner(reps: list[dict]) -> BenchmarkRunner:
    """Construct a BenchmarkRunner with a fixed representation list.

    Dataset I/O and pre-computation are mocked so no files are needed.
    """
    cfg = {**_MINIMAL_CFG, "representations": reps}
    with (
        patch(
            "asymbench.core.benchmark_runner.load_dataset",
            return_value=_FAKE_DF,
        ),
        patch.object(
            BenchmarkRunner,
            "_precompute_fit_free_representations",
        ),
    ):
        return BenchmarkRunner(cfg)


def _cache_hash(rep_cfg: dict, dataset_path: str = "data.csv") -> str:
    """Replicate the hash logic from BenchmarkRunner._rep_cache_path()."""
    rep_for_hash = {k: v for k, v in rep_cfg.items() if k != "name"}
    cache_key_str = json.dumps(
        {
            "rep": rep_for_hash,
            "reaction_features": [],
            "dataset_path": dataset_path,
        },
        sort_keys=True,
    )
    return hashlib.md5(cache_key_str.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# _result_rep_label
# ---------------------------------------------------------------------------


class TestResultRepLabel:
    def test_name_takes_priority(self):
        cfg = {
            "type": "morgan",
            "params": {"radius": 2, "n_bits": 2048},
            "name": "morgan_2048",
        }
        assert _result_rep_label(cfg) == "morgan_2048"

    def test_name_hf_transformer(self):
        cfg = {
            "type": "hf_transformer",
            "params": {"model_type": "chemberta"},
            "name": "ChemBERTa-77M",
        }
        assert _result_rep_label(cfg) == "ChemBERTa-77M"

    def test_fallback_morgan(self):
        cfg = {"type": "morgan", "params": {"radius": 2, "n_bits": 2048}}
        assert _result_rep_label(cfg) == "morgan"

    def test_fallback_hf_transformer(self):
        cfg = {
            "type": "hf_transformer",
            "params": {"model_type": "chemberta"},
        }
        assert _result_rep_label(cfg) == "hf_transformer_chemberta"

    def test_fallback_bespoke(self):
        cfg = {"type": "bespoke", "params": {"feature_name": "v1"}}
        assert _result_rep_label(cfg) == "bespoke_v1"

    def test_fallback_df_lookup(self):
        cfg = {"type": "df_lookup", "params": {"feature_name": "dft"}}
        assert _result_rep_label(cfg) == "df_lookup_dft"

    def test_fallback_precomputed(self):
        cfg = {"type": "precomputed", "params": {"feature_name": "embed"}}
        assert _result_rep_label(cfg) == "precomputed_embed"

    def test_fallback_rdkit_no_enrichment(self):
        cfg = {"type": "rdkit"}
        assert _result_rep_label(cfg) == "rdkit"

    def test_empty_name_falls_back(self):
        """An empty string name must not be used — fall back to auto-label."""
        cfg = {"type": "morgan", "params": {}, "name": ""}
        assert _result_rep_label(cfg) == "morgan"


# ---------------------------------------------------------------------------
# _rep_key excludes name
# ---------------------------------------------------------------------------


class TestRepKey:
    def test_same_key_different_names(self):
        runner = _make_runner([])
        cfg_a = {"type": "morgan", "params": {"n_bits": 2048}, "name": "a"}
        cfg_b = {"type": "morgan", "params": {"n_bits": 2048}, "name": "b"}
        assert runner._rep_key(cfg_a) == runner._rep_key(cfg_b)

    def test_different_key_different_params(self):
        runner = _make_runner([])
        cfg_a = {"type": "morgan", "params": {"n_bits": 2048}, "name": "a"}
        cfg_b = {"type": "morgan", "params": {"n_bits": 1024}, "name": "a"}
        assert runner._rep_key(cfg_a) != runner._rep_key(cfg_b)

    def test_key_without_name_matches_named(self):
        runner = _make_runner([])
        cfg_named = {"type": "rdkit", "name": "my_rdkit"}
        cfg_unnamed = {"type": "rdkit"}
        assert runner._rep_key(cfg_named) == runner._rep_key(cfg_unnamed)


# ---------------------------------------------------------------------------
# Cache hash excludes name (rename-safe)
# ---------------------------------------------------------------------------


class TestCacheHash:
    def test_hash_stable_across_rename(self):
        cfg_old = {"type": "morgan", "params": {"n_bits": 2048}, "name": "old"}
        cfg_new = {"type": "morgan", "params": {"n_bits": 2048}, "name": "new"}
        assert _cache_hash(cfg_old) == _cache_hash(cfg_new)

    def test_hash_differs_for_different_params(self):
        cfg_a = {"type": "morgan", "params": {"n_bits": 2048}}
        cfg_b = {"type": "morgan", "params": {"n_bits": 1024}}
        assert _cache_hash(cfg_a) != _cache_hash(cfg_b)

    def test_hash_differs_for_different_types(self):
        cfg_a = {"type": "morgan", "params": {}}
        cfg_b = {"type": "rdkit", "params": {}}
        assert _cache_hash(cfg_a) != _cache_hash(cfg_b)


# ---------------------------------------------------------------------------
# Duplicate label validation
# ---------------------------------------------------------------------------


class TestValidateRepLabels:
    def test_two_unnamed_same_type_raises(self):
        reps = [
            {"type": "morgan", "params": {"n_bits": 2048}},
            {"type": "morgan", "params": {"n_bits": 1024}},
        ]
        with pytest.raises(ValueError, match="same label 'morgan'"):
            _make_runner(reps)

    def test_two_named_same_name_raises(self):
        reps = [
            {"type": "morgan", "params": {"n_bits": 2048}, "name": "fp"},
            {"type": "rdkit", "name": "fp"},
        ]
        with pytest.raises(ValueError, match="same label 'fp'"):
            _make_runner(reps)

    def test_unique_names_no_raise(self):
        reps = [
            {"type": "morgan", "params": {"n_bits": 2048}, "name": "morgan_2048"},
            {"type": "morgan", "params": {"n_bits": 1024}, "name": "morgan_1024"},
        ]
        runner = _make_runner(reps)
        assert runner is not None

    def test_unique_auto_labels_no_raise(self):
        reps = [
            {"type": "morgan", "params": {}},
            {"type": "rdkit"},
        ]
        runner = _make_runner(reps)
        assert runner is not None

    def test_hf_transformer_same_model_type_raises(self):
        reps = [
            {"type": "hf_transformer", "params": {"model_type": "chemberta"}},
            {"type": "hf_transformer", "params": {"model_type": "chemberta"}},
        ]
        with pytest.raises(
            ValueError, match="same label 'hf_transformer_chemberta'"
        ):
            _make_runner(reps)

    def test_mixed_named_and_unnamed_conflict_raises(self):
        """Named rep whose name collides with auto-label of another rep."""
        reps = [
            {"type": "morgan", "params": {}},
            {"type": "rdkit", "name": "morgan"},  # name clashes with auto-label above
        ]
        with pytest.raises(ValueError, match="same label 'morgan'"):
            _make_runner(reps)


# ---------------------------------------------------------------------------
# Experiment._run_signature uses rep_label
# ---------------------------------------------------------------------------


class TestExperimentRepLabel:
    def _make_experiment(self, rep_label=None):
        mock_rep = MagicMock()
        mock_rep.rep_type = "morgan"
        mock_rep.rep_params = {}

        mock_optimizer = MagicMock()
        mock_optimizer.model_cfg = {"type": "random_forest"}

        mock_split = MagicMock()
        mock_split.config = {"sampler": "random", "train_size": 0.8}

        return Experiment(
            dataset=_FAKE_DF,
            smiles_columns=["smiles"],
            target="y",
            split_by_mol_col="smiles",
            representation=mock_rep,
            preprocessing=MagicMock(),
            y_scaling=MagicMock(),
            optimizer=mock_optimizer,
            split_strategy=mock_split,
            seed=0,
            rep_label=rep_label,
        )

    def test_rep_label_in_signature(self):
        exp = self._make_experiment(rep_label="morgan_2048")
        sig = exp._run_signature(model_type="random_forest")
        assert sig["representation"]["type"] == "morgan_2048"

    def test_fallback_when_no_rep_label(self):
        exp = self._make_experiment(rep_label=None)
        sig = exp._run_signature(model_type="random_forest")
        # Falls back to rep_type from the mock representation
        assert sig["representation"]["type"] == "morgan"

    def test_different_labels_produce_different_signatures(self):
        exp_a = self._make_experiment(rep_label="morgan_2048")
        exp_b = self._make_experiment(rep_label="morgan_1024")
        sig_a = exp_a._run_signature(model_type="random_forest")
        sig_b = exp_b._run_signature(model_type="random_forest")
        assert sig_a["representation"]["type"] != sig_b["representation"]["type"]


# ---------------------------------------------------------------------------
# GNNExperiment._run_signature uses rep_label
# ---------------------------------------------------------------------------


class TestGNNExperimentRepLabel:
    def _make_gnn_experiment(self, rep_label=None):
        mock_split = MagicMock()
        mock_split.config = {"sampler": "random", "train_size": 0.8}

        return GNNExperiment(
            dataset=_FAKE_DF,
            smiles_columns=["smiles"],
            target="y",
            split_by_mol_col="smiles",
            representation=MagicMock(),
            model_cfg={"type": "gnn", "params": {"architecture": "gcn"}},
            preprocessing=MagicMock(),
            y_scaling=MagicMock(),
            split_strategy=mock_split,
            seed=0,
            rep_label=rep_label,
        )

    def test_rep_label_in_signature(self):
        exp = self._make_gnn_experiment(rep_label="my_graph")
        sig = exp._run_signature()
        assert sig["representation"]["type"] == "my_graph"

    def test_fallback_to_graph_when_no_rep_label(self):
        exp = self._make_gnn_experiment(rep_label=None)
        sig = exp._run_signature()
        assert sig["representation"]["type"] == "graph"
