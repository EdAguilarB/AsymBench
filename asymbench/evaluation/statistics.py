import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

_VALID_ALTERNATIVES = ("two-sided", "greater", "less")
_VALID_CORRECTIONS = (
    "bonferroni", "sidak", "holm-sidak", "holm", "simes-hochberg",
    "hommel", "fdr_bh", "fdr_by", "fdr_tsbh", "fdr_tsbky",
)


def friedman_wilcoxon_by_group(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    replicate_cols: list[str] | None = None,
    alpha: float = 0.05,
    correction: str = "holm",
    alternative: str = "two-sided",
    dropna: str = "any",
    verbose: bool = True,
):
    """
    Compare groups (x_col categories) using:
      1) Friedman test (requires paired/repeated-measures structure)
      2) Pairwise Wilcoxon signed-rank tests with multiple-testing correction

    IMPORTANT:
    - These tests assume pairing across groups. You must define "replicates/tasks"
      so that each replicate has one observation per group.
    - Provide replicate_cols (e.g., ["seed","split","model","train_size"]) that define
      a replicate/task ID. If None, this function raises an error.
    - The Friedman test result does not gate the pairwise tests; post-hoc comparisons
      are always run. This is intentional for ML benchmarking workflows where pairwise
      differences are of interest regardless of the omnibus result.

    Parameters
    ----------
    data : pd.DataFrame
        Long-form dataframe containing x_col, y_col, and replicate identifiers.
    x_col : str
        Column with group labels (e.g., "representation" or "model").
    y_col : str
        Numeric column with metric values (e.g., "mae").
    replicate_cols : list[str] | None
        Columns that uniquely identify a replicate/task (paired measurement).
        Example: ["split", "model", "train_size", "seed"].
        Must produce exactly one row per (replicate, group) combination.
    alpha : float
        Significance threshold for corrected p-values.
    correction : str
        Multipletests method: "holm" (default), "bonferroni", "fdr_bh", etc.
        See statsmodels.stats.multitest.multipletests for all options.
    alternative : str
        Wilcoxon alternative: "two-sided" (default), "greater", "less".
    dropna : str
        "any" (default): drop replicates with any missing group value (recommended).
        "all": drop replicates only if all group values are missing.
    verbose : bool
        Print results to console.

    Returns
    -------
    results : dict
        {
          "friedman": {"statistic": float, "p_value": float, "n": int, "k": int},
          "pairwise": pd.DataFrame,
          "table_used": pd.DataFrame  # wide table used for tests
        }
    """

    # --- Input validation ---
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pd.DataFrame, got {type(data)}")

    if replicate_cols is None:
        raise ValueError(
            "replicate_cols must be provided to form paired replicates for Friedman/Wilcoxon.\n"
            "Example: replicate_cols=['model', 'split', 'train_size', 'seed']"
        )

    if alternative not in _VALID_ALTERNATIVES:
        raise ValueError(
            f"alternative must be one of {_VALID_ALTERNATIVES}, got '{alternative}'"
        )

    if correction not in _VALID_CORRECTIONS:
        raise ValueError(
            f"correction must be one of {_VALID_CORRECTIONS}, got '{correction}'"
        )

    if dropna not in ("any", "all"):
        raise ValueError("dropna must be 'any' or 'all'")

    missing = {x_col, y_col, *replicate_cols} - set(data.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # --- Prepare data ---
    df = data[[x_col, y_col, *replicate_cols]].copy()
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col])

    # Build replicate id
    df["_replicate_id"] = df[replicate_cols].astype(str).agg("__".join, axis=1)

    # Detect duplicates: replicate_cols must uniquely identify each (replicate, group) pair
    duplicates = df.duplicated(subset=["_replicate_id", x_col], keep=False)
    if duplicates.any():
        n_dupes = duplicates.sum()
        example = df.loc[duplicates, ["_replicate_id", x_col, y_col]].head(5)
        raise ValueError(
            f"replicate_cols do not uniquely identify each (replicate, group) pair. "
            f"Found {n_dupes} duplicate rows. "
            f"Check your replicate_cols definition.\nExample duplicates:\n{example}"
        )

    # Pivot to wide: rows=replicates, cols=groups
    # Use .pivot() instead of .pivot_table() so duplicate entries raise rather than aggregate
    table = df.pivot(index="_replicate_id", columns=x_col, values=y_col)
    table.columns.name = None

    if dropna == "any":
        n_before = len(table)
        table = table.dropna(axis=0, how="any")
        n_dropped = n_before - len(table)
        if n_dropped > 0 and verbose:
            print(
                f"[Info] Dropped {n_dropped} replicate(s) with missing values for "
                f"at least one group (dropna='any')."
            )
    else:
        table = table.dropna(axis=0, how="all")

    if table.shape[1] < 2:
        raise ValueError("Need at least 2 groups to compare.")
    if table.shape[0] < 2:
        raise ValueError("Need at least 2 paired replicates after filtering.")

    groups = list(table.columns)
    n, k = table.shape

    # --- Friedman test ---
    fried_stat, fried_p = friedmanchisquare(
        *[table[g].to_numpy() for g in groups]
    )

    # --- Pairwise Wilcoxon signed-rank tests ---
    pairs = list(combinations(groups, 2))
    raw_pvals = []
    rows = []

    for a, b in pairs:
        xa = table[a].to_numpy()
        xb = table[b].to_numpy()
        diffs = xa - xb

        if np.all(diffs == 0):
            warnings.warn(
                f"All differences are zero for pair ('{a}', '{b}'). "
                "Wilcoxon test is not applicable; p-value set to 1.0.",
                UserWarning,
                stacklevel=2,
            )
            stat, p = np.nan, 1.0
        else:
            stat, p = wilcoxon(xa, xb, alternative=alternative, zero_method="wilcox")

        raw_pvals.append(p)
        rows.append(
            {
                "group_a": a,
                "group_b": b,
                "n": int(len(xa)),
                "wilcoxon_stat": float(stat) if np.isfinite(stat) else np.nan,
                "p_value": float(p),
                "median_diff_a_minus_b": float(np.median(diffs)),
            }
        )

    reject, p_corr, _, _ = multipletests(raw_pvals, alpha=alpha, method=correction)

    pairwise = pd.DataFrame(rows)
    pairwise["p_value_corrected"] = p_corr
    pairwise["reject_null"] = reject
    pairwise = pairwise.sort_values(
        ["p_value_corrected", "p_value"]
    ).reset_index(drop=True)

    out = {
        "friedman": {
            "statistic": float(fried_stat),
            "p_value": float(fried_p),
            "n": int(n),
            "k": int(k),
        },
        "pairwise": pairwise,
        "table_used": table,
    }

    if verbose:
        print("\n=== Friedman test ===")
        print(f"Groups     : {groups}")
        print(f"Replicates : n={n}, groups k={k}")
        print(f"Statistic  = {fried_stat:.6g}")
        print(f"p-value    = {fried_p:.6g}")

        print(
            f"\n=== Pairwise Wilcoxon ({correction} corrected, alpha={alpha}) ==="
        )
        for _, r in pairwise.iterrows():
            sig = "YES" if r["reject_null"] else "no"
            print(
                f"  {r['group_a']:18s} vs {r['group_b']:18s} | "
                f"p_raw={r['p_value']:.4g} | p_corr={r['p_value_corrected']:.4g} | "
                f"median(a-b)={r['median_diff_a_minus_b']:.4g} | significant={sig}"
            )
        print("=" * 41 + "\n")

    return out