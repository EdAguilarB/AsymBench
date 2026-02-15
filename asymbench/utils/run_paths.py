from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def _slug(x: Any) -> str:
    s = str(x).strip().replace(" ", "")
    # keep it filesystem-friendly
    for ch in [
        "/",
        "\\",
        ":",
        ";",
        ",",
        "{",
        "}",
        "[",
        "]",
        "(",
        ")",
        "'",
        '"',
    ]:
        s = s.replace(ch, "_")
    return s


def build_run_dir(
    base_dir: str | Path,
    dataset_name: str,
    rep_type: str,
    model_type: str,
    split_type: str,
    split_by_mol_col: str,
    train_size: float,
    seed: int,
    extra: Dict[str, Any] | None = None,
) -> Path:
    """
    Create a directory path that makes runs easy to find.

    Example:
      experiments/plots/datasetX/rep=morgan/model=rf/split=scaffold/by=substrate/train=0.8/seed=0/
    """
    base_dir = Path(base_dir)
    extra = extra or {}

    parts = [
        _slug(dataset_name),
        f"rep={_slug(rep_type)}",
        f"model={_slug(model_type)}",
        f"split={_slug(split_type)}",
        f"by={_slug(split_by_mol_col)}",
        f"train={_slug(train_size)}",
        f"seed={_slug(seed)}",
    ]

    # optional additional knobs (e.g. scaler=standard, corr=0.95)
    for k, v in extra.items():
        parts.append(f"{_slug(k)}={_slug(v)}")

    outdir = base_dir.joinpath(*parts)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir
