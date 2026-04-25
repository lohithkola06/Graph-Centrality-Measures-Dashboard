"""Entry point for running synthetic centrality benchmarks."""

from src.experiments import run_default_benchmarks


def main() -> None:
    result = run_default_benchmarks()
    print("Benchmark suite completed successfully.")
    print(f"CSV output: {result['csv_path']}")
    print(f"JSON output: {result['json_path']}")
    print(f"Log output: {result['log_path']}")
    print(f"Rows written: {result['result_count']}")


if __name__ == "__main__":
    main()
