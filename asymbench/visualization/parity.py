from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def make_parity_plot(
    y_true,
    y_pred,
    metrics: Dict[str, Any],
    title: str,
    outpath: Path,
    subtitle: Optional[str] = None,
    dpi: int = 300,
) -> Path:
    """
    Create a parity plot (y_true vs y_pred) with identity line + metrics, and save to disk.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted values (1D).
    metrics : dict
        Typically output from evaluate_predictions (rmse, mae, r2, etc.)
    title : str
        Main title shown on plot.
    outpath : Path
        Full file path to save (including filename).
    subtitle : str | None
        Optional secondary line (e.g. run settings).
    dpi : int
        Save DPI.

    Returns
    -------
    outpath : Path
        Where the figure was saved.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # axis limits (nice padding)
    vmin = np.nanmin(np.concatenate([y_true, y_pred]))
    vmax = np.nanmax(np.concatenate([y_true, y_pred]))
    pad = 0.05 * (vmax - vmin) if vmax > vmin else 1.0
    lo, hi = vmin - pad, vmax + pad

    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=300)

    ax.scatter(y_true, y_pred, alpha=0.75, s=25, c="teal")
    ax.plot([lo, hi], [lo, hi], linewidth=1.5)  # identity line

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(
        "True $\Delta \Delta G^{\ddagger} / kJ \; mol^{-1}$", fontdict={"size": 15}
    )
    ax.set_ylabel(
        "Predicted $\Delta \Delta G^{\ddagger} / kJ \; mol^{-1}$", fontdict={"size": 15}
    )

    ax.set_title(title, pad=12)
    if subtitle:
        ax.text(
            0.5,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # metrics box
    rmse = _safe_float(metrics.get("rmse"))
    mae = _safe_float(metrics.get("mae"))
    r2 = _safe_float(metrics.get("r2"))
    n = len(y_true)

    text = f"N = {n}\nRMSE = {rmse:.4g}\nMAE  = {mae:.4g}\nR²   = {r2:.4g}"
    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.2),
    )

    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return outpath
