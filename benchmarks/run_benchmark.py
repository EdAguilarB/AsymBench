import logging
import os

# Must be set before any OpenMP-linked library (numpy, torch, unimol_tools) is
# imported.  On macOS, PyTorch's libomp and Intel MKL's libiomp5 are both
# pulled in by the dependency stack; without this flag the second library to
# initialise triggers a segfault during model-weight loading.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Some dependencies (e.g. unimol_tools) call logging.basicConfig(level=DEBUG),
# which causes every matplotlib sub-logger to emit internal debug lines
# (font scanning, backend loading, tick locators, …).
# Silencing the top-level "matplotlib" logger suppresses all of them at once.
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import argparse
from pathlib import Path

import yaml

from asymbench.core.benchmark_runner import BenchmarkRunner
from asymbench.core.result_aggregator import aggregate_results


def main():

    parser = argparse.ArgumentParser(
        description="Run AsymBench benchmark from YAML config."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="benchmarks/external_config.yaml",
        help="Path to YAML configuration file.",
    )

    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    runner = BenchmarkRunner(config)
    results = runner.run()

    summary = aggregate_results(results)

    out_dir = Path(config["log_dirs"]["benchmark"])
    out_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(out_dir / "summary.csv", index=False)


if __name__ == "__main__":
    main()
