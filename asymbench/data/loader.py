from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_dataset(config: dict) -> pd.DataFrame:
    """Load a reaction dataset CSV and return a clean DataFrame.

    Parameters
    ----------
    config:
        Dataset config dict (typically ``config["dataset"]`` from YAML).
        Recognised keys:

        Required
        ^^^^^^^^
        ``path``
            Path to the CSV file.
        ``id_col``
            Column to use as the DataFrame index (unique row identifier).
        ``smiles_columns``
            List of SMILES column names for molecular components.
        ``target``
            Name of the numeric regression target column.

        Optional
        ^^^^^^^^
        ``reaction_features``
            List of additional numeric column names to include (e.g.
            temperature, reaction time).  These columns are kept alongside
            the SMILES and target columns so they are available downstream
            for concatenation to the feature matrix.

    Returns
    -------
    pd.DataFrame
        Index set to ``id_col``; columns are the union of
        ``smiles_columns``, ``[target]``, and ``reaction_features``
        (if provided).  Rows with NaN in *any kept column* are dropped.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    KeyError
        If ``id_col``, any SMILES column, the target column, or any
        requested reaction-feature column is absent from the CSV.
    ValueError
        If the dataset is empty after loading, or if ``id_col`` values
        are not unique.
    """
    path = Path(config["path"])
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = pd.read_csv(path)

    # ------------------------------------------------------------------ #
    # 1. Validate and set index                                            #
    # ------------------------------------------------------------------ #
    id_col = config["id_col"]
    if id_col not in data.columns:
        raise KeyError(
            f"id_col '{id_col}' not found in CSV columns. "
            f"Available columns: {data.columns.tolist()}"
        )
    data = data.set_index(id_col)

    if not data.index.is_unique:
        n_dupes = int(data.index.duplicated().sum())
        raise ValueError(
            f"id_col '{id_col}' contains {n_dupes} duplicate value(s). "
            "Row identifiers must be unique."
        )

    # ------------------------------------------------------------------ #
    # 2. Validate SMILES columns                                           #
    # ------------------------------------------------------------------ #
    smiles_cols: list[str] = config["smiles_columns"]
    _check_columns_exist(data, smiles_cols, context="smiles_columns")

    # ------------------------------------------------------------------ #
    # 3. Validate target column                                            #
    # ------------------------------------------------------------------ #
    target: str = config["target"]
    if target not in data.columns:
        raise KeyError(
            f"target column '{target}' not found in CSV columns. "
            f"Available columns: {data.columns.tolist()}"
        )

    # ------------------------------------------------------------------ #
    # 4. Validate and collect reaction feature columns                     #
    # ------------------------------------------------------------------ #
    reaction_features: list[str] = config.get("reaction_features") or []
    if reaction_features:
        _check_columns_exist(data, reaction_features, context="reaction_features")

        # Warn about any reaction feature that overlaps with SMILES or target
        overlap = set(reaction_features) & (set(smiles_cols) | {target})
        if overlap:
            logger.warning(
                "reaction_features %s overlap with smiles_columns or target. "
                "Duplicate columns will be deduplicated.",
                sorted(overlap),
            )

    # ------------------------------------------------------------------ #
    # 5. Select only the columns we need (deduplicated, order preserved)   #
    # ------------------------------------------------------------------ #
    keep = list(dict.fromkeys(smiles_cols + [target] + reaction_features))
    data = data[keep]

    # ------------------------------------------------------------------ #
    # 6. Drop rows with NaN in any kept column                             #
    # ------------------------------------------------------------------ #
    n_before = len(data)
    data = data.dropna()
    n_dropped = n_before - len(data)
    if n_dropped:
        logger.warning(
            "Dropped %d row(s) with missing values from '%s'.",
            n_dropped,
            path.name,
        )

    if data.empty:
        raise ValueError(
            f"Dataset at '{path}' is empty after loading and NaN removal."
        )

    return data


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _check_columns_exist(
    df: pd.DataFrame, columns: list[str], context: str
) -> None:
    """Raise a descriptive KeyError if any column in *columns* is absent."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"The following {context} column(s) were not found in the CSV: "
            f"{missing}. Available columns: {df.columns.tolist()}"
        )
