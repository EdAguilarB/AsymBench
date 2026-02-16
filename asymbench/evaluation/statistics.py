from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests


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
      a replicate/task ID. If None, this function assumes `data` is already in wide
      format with each column a group (see note below).

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
        If None, will raise unless data is already wide (see below).
    alpha : float
        Significance threshold for corrected p-values.
    correction : str
        Multipletests method: "holm", "bonferroni", "fdr_bh", ...
    alternative : str
        Wilcoxon alternative: "two-sided", "greater", "less".
    dropna : str
        "any" (default): drop replicates with any missing group value (recommended).
        "all": drop replicates only if all are missing.
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

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pd.DataFrame, got {type(data)}")

    # If replicate_cols not provided, we can't construct paired table reliably.
    if replicate_cols is None:
        raise ValueError(
            "replicate_cols must be provided to form paired replicates for Friedman/Wilcoxon.\n"
            "Example: replicate_cols=['model','split','train_size','seed']"
        )

    missing = {x_col, y_col, *replicate_cols} - set(data.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = data[[x_col, y_col, *replicate_cols]].copy()
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col])

    # Build replicate id
    df["_replicate_id"] = df[replicate_cols].astype(str).agg("__".join, axis=1)

    # Pivot to wide: rows=replicates, cols=groups
    table = df.pivot_table(
        index="_replicate_id", columns=x_col, values=y_col, aggfunc="mean"
    )

    if dropna == "any":
        table = table.dropna(axis=0, how="any")
    elif dropna == "all":
        table = table.dropna(axis=0, how="all")
    else:
        raise ValueError("dropna must be 'any' or 'all'")

    if table.shape[1] < 2:
        raise ValueError("Need at least 2 groups to compare.")
    if table.shape[0] < 2:
        raise ValueError("Need at least 2 paired replicates after filtering.")

    groups = list(table.columns)
    n, k = table.shape

    # --- Friedman ---
    fried_stat, fried_p = friedmanchisquare(
        *[table[g].to_numpy() for g in groups]
    )

    # --- Pairwise Wilcoxon ---
    pairs = list(combinations(groups, 2))
    raw_pvals = []
    rows = []

    for a, b in pairs:
        xa = table[a].to_numpy()
        xb = table[b].to_numpy()
        diffs = xa - xb

        # Wilcoxon can error if all diffs are 0; handle gracefully
        try:
            stat, p = wilcoxon(
                xa, xb, alternative=alternative, zero_method="wilcox"
            )
        except ValueError:
            stat, p = np.nan, 1.0

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

    reject, p_corr, _, _ = multipletests(
        raw_pvals, alpha=alpha, method=correction
    )

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
        print(f"Groups: {groups}")
        print(f"Paired replicates used: n={n}, groups k={k}")
        print(f"statistic = {fried_stat:.6g}")
        print(f"p-value   = {fried_p:.6g}")

        print(
            f"\n=== Pairwise Wilcoxon ({correction} corrected, alpha={alpha}) ==="
        )
        for _, r in pairwise.iterrows():
            sig = "YES" if r["reject_null"] else "no"
            print(
                f"{r['group_a']:18s} vs {r['group_b']:18s} | "
                f"p_raw={r['p_value']:.4g} | p_corr={r['p_value_corrected']:.4g} | "
                f"median(a-b)={r['median_diff_a_minus_b']:.4g} | significant={sig}"
            )
        print("=========================================\n")

    return out
