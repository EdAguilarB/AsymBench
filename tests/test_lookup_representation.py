"""Tests for DataFrameLookupRepresentation.

Covers the two usage modes:
  1. Explicit feature_columns list (existing behaviour)
  2. All-columns mode — feature_columns omitted or null (new behaviour)

Also covers the prefix bug-fix: prefix=None / prefix="" must not produce
"None__<col>" column names.
"""

from __future__ import annotations

import io
import textwrap

import pandas as pd
import pytest

from asymbench.representations.lookup import DataFrameLookupRepresentation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rep(params: dict) -> DataFrameLookupRepresentation:
    """Construct a DataFrameLookupRepresentation from a params dict."""
    config = {
        "representation": {
            "type": "bespoke",
            "params": params,
        },
        "data": {"smiles_columns": ["smiles"]},
    }
    return DataFrameLookupRepresentation(config=config)


def _write_csv(tmp_path, content: str, filename: str = "features.csv") -> str:
    """Write a CSV string to a temp file and return its path as a string."""
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content).strip())
    return str(p)


# ---------------------------------------------------------------------------
# All-columns mode (feature_columns omitted)
# ---------------------------------------------------------------------------

class TestAllColumnsMode:
    def test_uses_all_columns_when_feature_columns_omitted(self, tmp_path):
        csv = _write_csv(tmp_path, """
            id,feat_a,feat_b,feat_c
            rx1,1.0,2.0,3.0
            rx2,4.0,5.0,6.0
        """)
        rep = _make_rep({
            "features_path": csv,
            "index_col": "id",
            # feature_columns intentionally omitted
        })
        assert rep.feature_columns is None
        # All three feature columns must be present
        assert set(rep._features.columns) == {"feat_a", "feat_b", "feat_c"}

    def test_transform_returns_correct_values_all_columns(self, tmp_path):
        csv = _write_csv(tmp_path, """
            id,feat_a,feat_b
            rx1,10.0,20.0
            rx2,30.0,40.0
        """)
        rep = _make_rep({"features_path": csv, "index_col": "id"})

        df = pd.DataFrame({"smiles": ["C", "N"]}, index=["rx1", "rx2"])
        X = rep.transform(df)

        assert list(X.columns) == ["feat_a", "feat_b"]
        assert X.loc["rx1", "feat_a"] == pytest.approx(10.0)
        assert X.loc["rx2", "feat_b"] == pytest.approx(40.0)

    def test_get_metadata_flags_all_columns(self, tmp_path):
        csv = _write_csv(tmp_path, """
            id,a,b
            r1,1,2
        """)
        rep = _make_rep({"features_path": csv, "index_col": "id"})
        meta = rep.get_metadata()
        assert meta["using_all_columns"] is True
        assert meta["params"]["feature_columns"] is None

    def test_non_numeric_columns_still_loaded_with_warning(
        self, tmp_path, caplog
    ):
        """Non-numeric columns trigger a warning but are still included."""
        import logging
        csv = _write_csv(tmp_path, """
            id,numeric_col,text_col
            r1,1.0,hello
            r2,2.0,world
        """)
        with caplog.at_level(logging.WARNING, logger="asymbench.representations.lookup"):
            rep = _make_rep({"features_path": csv, "index_col": "id"})

        assert "non-numeric" in caplog.text.lower()
        # Both columns are still present
        assert "numeric_col" in rep._features.columns
        assert "text_col" in rep._features.columns


# ---------------------------------------------------------------------------
# Explicit feature_columns mode (existing behaviour preserved)
# ---------------------------------------------------------------------------

class TestExplicitColumnsMode:
    def test_explicit_columns_subset(self, tmp_path):
        csv = _write_csv(tmp_path, """
            id,a,b,c
            r1,1,2,3
            r2,4,5,6
        """)
        rep = _make_rep({
            "features_path": csv,
            "index_col": "id",
            "feature_columns": ["a", "c"],
        })
        assert list(rep._features.columns) == ["a", "c"]

    def test_missing_explicit_column_raises(self, tmp_path):
        csv = _write_csv(tmp_path, """
            id,a,b
            r1,1,2
        """)
        with pytest.raises(KeyError, match="not found in features table"):
            _make_rep({
                "features_path": csv,
                "index_col": "id",
                "feature_columns": ["a", "does_not_exist"],
            })

    def test_get_metadata_not_all_columns(self, tmp_path):
        csv = _write_csv(tmp_path, """
            id,a,b
            r1,1,2
        """)
        rep = _make_rep({
            "features_path": csv,
            "index_col": "id",
            "feature_columns": ["a"],
        })
        meta = rep.get_metadata()
        assert meta["using_all_columns"] is False
        assert meta["params"]["feature_columns"] == ["a"]


# ---------------------------------------------------------------------------
# Prefix behaviour
# ---------------------------------------------------------------------------

class TestPrefix:
    def test_prefix_applied(self, tmp_path):
        csv = _write_csv(tmp_path, """
            id,feat
            r1,1.0
        """)
        rep = _make_rep({
            "features_path": csv,
            "index_col": "id",
            "prefix": "bespoke",
        })
        assert list(rep._features.columns) == ["bespoke__feat"]

    def test_prefix_none_keeps_original_names(self, tmp_path):
        """prefix=None (YAML: prefix:) must NOT produce 'None__feat'."""
        csv = _write_csv(tmp_path, """
            id,feat
            r1,1.0
        """)
        rep = _make_rep({
            "features_path": csv,
            "index_col": "id",
            "prefix": None,
        })
        assert list(rep._features.columns) == ["feat"]
        # Explicitly check the bug regression: no "None" in any column name
        for col in rep._features.columns:
            assert "None" not in col

    def test_prefix_empty_string_keeps_original_names(self, tmp_path):
        """prefix='' must also keep original column names."""
        csv = _write_csv(tmp_path, """
            id,feat
            r1,1.0
        """)
        rep = _make_rep({
            "features_path": csv,
            "index_col": "id",
            "prefix": "",
        })
        assert list(rep._features.columns) == ["feat"]

    def test_prefix_omitted_keeps_original_names(self, tmp_path):
        """prefix not provided at all → no prefix applied (default)."""
        csv = _write_csv(tmp_path, """
            id,feat
            r1,1.0
        """)
        rep = _make_rep({
            "features_path": csv,
            "index_col": "id",
            # prefix intentionally omitted
        })
        assert list(rep._features.columns) == ["feat"]

    def test_prefix_stored_as_none_in_metadata_when_empty(self, tmp_path):
        """get_metadata() returns None for prefix when no prefix was applied."""
        csv = _write_csv(tmp_path, """
            id,feat
            r1,1.0
        """)
        rep = _make_rep({"features_path": csv, "index_col": "id", "prefix": None})
        assert rep.get_metadata()["params"]["prefix"] is None

    def test_prefix_stored_correctly_in_metadata_when_set(self, tmp_path):
        csv = _write_csv(tmp_path, """
            id,feat
            r1,1.0
        """)
        rep = _make_rep({
            "features_path": csv,
            "index_col": "id",
            "prefix": "bespoke",
        })
        assert rep.get_metadata()["params"]["prefix"] == "bespoke"


# ---------------------------------------------------------------------------
# strict / missing-key behaviour (regression)
# ---------------------------------------------------------------------------

class TestStrictMode:
    def test_strict_raises_on_missing_key(self, tmp_path):
        csv = _write_csv(tmp_path, """
            id,feat
            r1,1.0
        """)
        rep = _make_rep({
            "features_path": csv,
            "index_col": "id",
            "strict": True,
        })
        df = pd.DataFrame({"smiles": ["C"]}, index=["r_missing"])
        with pytest.raises(KeyError, match="missing in features lookup"):
            rep.transform(df)

    def test_non_strict_fills_zeros(self, tmp_path):
        csv = _write_csv(tmp_path, """
            id,feat
            r1,1.0
        """)
        rep = _make_rep({
            "features_path": csv,
            "index_col": "id",
            "strict": False,
        })
        df = pd.DataFrame({"smiles": ["C"]}, index=["r_missing"])
        X = rep.transform(df)
        assert X.iloc[0, 0] == pytest.approx(0.0)
