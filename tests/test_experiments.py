"""Tests for experiment and benchmark export utilities."""

from __future__ import annotations

import csv
import json

from src.experiments import BenchmarkSpec, run_benchmark_suite
from src.graph_utils import generate_path_graph, generate_star_graph


def test_run_benchmark_suite_writes_csv_json_and_log(tmp_path):
    specs = [
        BenchmarkSpec(
            family="path",
            label="path_n12",
            config={"n": 12},
            builder=lambda: generate_path_graph(12),
        ),
        BenchmarkSpec(
            family="star",
            label="star_n12",
            config={"n": 12},
            builder=lambda: generate_star_graph(12),
        ),
    ]

    csv_path = tmp_path / "benchmark_results.csv"
    json_path = tmp_path / "benchmark_results.json"
    log_dir = tmp_path / "logs"

    result = run_benchmark_suite(
        specs=specs,
        csv_path=csv_path,
        json_path=json_path,
        log_dir=log_dir,
    )

    assert result["suite_size"] == len(specs)
    assert result["result_count"] == len(specs) * 3
    assert csv_path.exists()
    assert json_path.exists()
    assert result["log_path"].exists()

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert len(rows) == len(specs) * 3
    assert {row["metric_name"] for row in rows} == {"closeness", "betweenness", "eigenvalue"}

    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["suite_size"] == len(specs)
    assert payload["result_count"] == len(specs) * 3
    assert len(payload["results"]) == len(specs) * 3
    assert {row["graph_label"] for row in payload["results"]} == {"path_n12", "star_n12"}
