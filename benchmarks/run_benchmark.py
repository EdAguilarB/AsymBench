import argparse
import yaml
from pathlib import Path

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