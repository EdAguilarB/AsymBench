"""SHAP-style beeswarm plot for GNN fragment importances (Integrated Gradients).

The plot mirrors the visual language of SHAP summary beeswarm plots:

* **Horizontal layout** — x-axis = importance value, one row per fragment/source.
* **Signed colour** — red dots push the prediction up; blue dots push it down.
* **Sorted** — rows ordered by mean absolute importance (highest at top).
* **Jittered** — overlapping dots are spread vertically so density is visible.

Usage
-----
::

    from asymbench.visualization.gnn_beeswarm import plot_gnn_fragment_beeswarm
    import pandas as pd

    df = pd.read_csv("fragment_importances.csv")  # fragment, source, importance
    plot_gnn_fragment_beeswarm(df, outpath="beeswarm.png", top_k=20)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SMILES_TRUNCATE = 28   # max characters shown in a fragment label


def _shorten_smiles(smi: str, max_len: int = _SMILES_TRUNCATE) -> str:
    """Truncate long SMILES so y-axis labels stay readable."""
    if len(smi) <= max_len:
        return smi
    return smi[:max_len - 1] + "…"


def _clean_source(col: str) -> str:
    """Strip common suffixes (_smiles, _SMILES) for a compact axis label."""
    for suffix in ("_smiles", "_SMILES", "_smi", "_SMI"):
        if col.endswith(suffix):
            return col[: -len(suffix)]
    return col


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def plot_gnn_fragment_beeswarm(
    df: pd.DataFrame,
    outpath: Path,
    top_k: int = 5,
    figsize: tuple[int, int] | None = None,
    dpi: int = 350,
    jitter_strength: float = 0.20,
    title: str = "Fragment importances (Integrated Gradients)",
    point_size: float = 18,
    point_alpha: float = 0.75,
) -> Path:
    """Create and save a SHAP-style horizontal beeswarm for GNN fragment importances.

    Shows the top *k* most positive and top *k* most negative
    ``(fragment, source)`` pairs — up to ``2 * top_k`` rows total — ranked
    by their mean importance score.  This mirrors the directional split used
    in SHAP beeswarm plots, making it easy to read which fragments drive
    predictions up vs. down.

    Parameters
    ----------
    df:
        Long-format DataFrame with columns ``fragment``, ``source``,
        ``importance``.  One row per individual IG score.
    outpath:
        Destination path for the PNG file.
    top_k:
        Number of ``(fragment, source)`` pairs to show on each side
        (positive and negative).  The plot will display at most
        ``2 * top_k`` rows.
    figsize:
        ``(width, height)`` in inches.  Auto-sized from the number of rows
        shown if not provided.
    dpi:
        Output resolution.
    jitter_strength:
        Half-width of the uniform vertical jitter applied to each dot.
        Increase if dots overlap too much.
    title:
        Figure title.
    point_size:
        Scatter marker area (``s`` parameter of ``ax.scatter``).
    point_alpha:
        Scatter marker alpha.

    Returns
    -------
    Path
        The path where the figure was saved.
    """
    if df.empty:
        return Path(outpath)

    df = df.copy()

    # Build a human-readable y-axis label: "fragment (source)"
    df["_frag_short"] = df["fragment"].map(_shorten_smiles)
    df["_src_short"] = df["source"].map(_clean_source)
    df["_label"] = df["_frag_short"] + "\n(" + df["_src_short"] + ")"

    # Mean importance per (fragment, source) pair — signed
    mean_imp = (
        df.groupby("_label")["importance"]
        .mean()
        .rename("mean_imp")
    )

    # Top-k most positive + top-k most negative, union of both sets
    top_pos = mean_imp.nlargest(top_k).index.tolist()
    top_neg = mean_imp.nsmallest(top_k).index.tolist()
    selected = list(dict.fromkeys(top_pos + top_neg))  # preserves order, deduplicates

    df = df[df["_label"].isin(selected)].copy()

    # Final display order: most negative at bottom → most positive at top
    order = mean_imp.loc[selected].sort_values(ascending=True).index.tolist()
    label_to_y = {label: i for i, label in enumerate(order)}

    # Symmetric colour scale centred on zero  (red = positive, blue = negative)
    vmax = df["importance"].abs().max()
    if vmax == 0:
        vmax = 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.bwr   # blue → white → red

    # Auto-size figure height — up to 2*top_k rows now
    n_rows = len(order)
    if figsize is None:
        figsize = (10, max(4, 0.5 * n_rows + 2.0))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    rng = np.random.default_rng(42)

    for label in order:
        y_center = label_to_y[label]
        vals = df.loc[df["_label"] == label, "importance"].to_numpy()

        # Vertical jitter — proportional to density for a true beeswarm feel
        jitter = rng.uniform(-jitter_strength, jitter_strength, size=len(vals))

        colors = cmap(norm(vals))
        ax.scatter(
            vals,
            np.full(len(vals), y_center) + jitter,
            c=colors,
            s=point_size,
            alpha=point_alpha,
            edgecolors="black",
            zorder=3,
        )

    # Zero line
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--", zorder=2, alpha=0.6)

    # Y-axis labels
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_ylim(-0.7, n_rows - 0.3)

    ax.set_xlabel("IG Fragment Importance", fontsize=13)
    ax.set_title(title, fontsize=13, pad=10)

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("Importance", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # Styling — match existing package aesthetics
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.7)
    ax.tick_params(axis="both", labelsize=9)

    fig.tight_layout()

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return outpath


# ---------------------------------------------------------------------------
# Convenience: load CSV and plot in one call
# ---------------------------------------------------------------------------

def plot_gnn_fragment_beeswarm_from_csv(
    csv_path: Path,
    outpath: Path | None = None,
    **kwargs,
) -> Path:
    """Load a ``fragment_importances.csv`` and call :func:`plot_gnn_fragment_beeswarm`.

    Parameters
    ----------
    csv_path:
        Path to the long-format CSV (columns: fragment, source, importance).
    outpath:
        Destination PNG.  Defaults to ``csv_path.parent / "fragment_beeswarm.png"``.
    **kwargs:
        Forwarded to :func:`plot_gnn_fragment_beeswarm`.
    """
    csv_path = Path(csv_path)
    if outpath is None:
        outpath = csv_path.parent / "fragment_beeswarm.png"

    df = pd.read_csv(csv_path)
    return plot_gnn_fragment_beeswarm(df, outpath=outpath, **kwargs)
