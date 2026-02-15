from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
    tmp.replace(path)


def _slug(x: Any) -> str:
    """
    Make a filesystem-friendly string.
    """
    s = str(x).strip()
    if s == "":
        return "none"

    # replace common problematic chars
    bad = [
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
        '"',
        "'",
        "|",
        "?",
        "*",
        "<",
        ">",
        "=",
    ]
    for ch in bad:
        s = s.replace(ch, "_")

    # collapse whitespace
    s = "_".join(s.split())
    return s


def _format_train_size(ts: Any) -> str:
    """
    Stable formatting for train_size.
    """
    try:
        f = float(ts)
        # 0.8 -> "0p80" to avoid "." in folder names
        return f"{f:.2f}".replace(".", "p")
    except Exception:
        return _slug(ts)


@dataclass
class RunStore:
    """
    Human-readable run store that builds directory paths from signature values.

    Expected signature format (minimal):
      signature["representation"]["type"]
      signature["representation"]["feature_name"]
      signature["model"]["type"]
      signature["split"]["sampler"]
      signature["split"]["train_size"]
      signature["seed"]
    """

    base_dir: Path = Path("experiments/runs")

    # -------------------------
    # Directory construction
    # -------------------------

    def run_dir_from_signature(self, signature: Dict[str, Any]) -> Path:
        rep = signature.get("representation", {})
        model = signature.get("model", {})
        split = signature.get("split", {})

        rep_type = _slug(rep.get("type", "unknown_rep"))
        model_type = _slug(model.get("type", "unknown_model"))
        sampler = _slug(split.get("sampler", "unknown_sampler"))
        train_size = _format_train_size(split.get("train_size", "na"))
        seed = _slug(signature.get("seed", "na"))

        d = (
            Path(self.base_dir)
            / rep_type
            / model_type
            / sampler
            / f"train_{train_size}"
            / f"seed_{seed}"
        )
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -------------------------
    # File paths
    # -------------------------

    def signature_path(self, signature: Dict[str, Any]) -> Path:
        return self.run_dir_from_signature(signature) / "signature.json"

    def status_path(self, signature: Dict[str, Any]) -> Path:
        return self.run_dir_from_signature(signature) / "status.json"

    def metrics_path(self, signature: Dict[str, Any]) -> Path:
        return self.run_dir_from_signature(signature) / "metrics.json"

    # -------------------------
    # Status / load / save
    # -------------------------

    def get_status(self, signature: Dict[str, Any]) -> Dict[str, Any]:
        p = self.status_path(signature)
        if not p.exists():
            return {"state": "missing"}
        with open(p, "r") as f:
            return json.load(f)

    def exists_complete(self, signature: Dict[str, Any]) -> bool:
        if not self.metrics_path(signature).exists():
            return False
        status = self.get_status(signature)
        return status.get("state") == "completed"

    def load_metrics(self, signature: Dict[str, Any]) -> Dict[str, Any]:
        with open(self.metrics_path(signature), "r") as f:
            return json.load(f)

    def mark_started(self, signature: Dict[str, Any]) -> None:
        _atomic_write_json(self.signature_path(signature), signature)
        _atomic_write_json(self.status_path(signature), {"state": "started"})

    def mark_completed(
        self, signature: Dict[str, Any], metrics: Dict[str, Any]
    ) -> None:
        _atomic_write_json(self.metrics_path(signature), metrics)
        _atomic_write_json(self.status_path(signature), {"state": "completed"})

    def mark_failed(
        self, signature: Dict[str, Any], exc: BaseException
    ) -> None:
        _atomic_write_json(
            self.status_path(signature),
            {
                "state": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
