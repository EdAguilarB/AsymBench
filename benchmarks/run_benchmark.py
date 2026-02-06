import yaml
from asymbench.core.benchmark_runner import BenchmarkRunner
from asymbench.core.result_aggregator import aggregate_results


def main():
    with open("benchmarks/benchmark_config.yaml") as f:
        config = yaml.safe_load(f)

    runner = BenchmarkRunner(config)
    results = runner.run()

    summary = aggregate_results(results)
    summary.to_csv("experiments/results/summary.csv", index=False)


if __name__ == "__main__":
    main()
