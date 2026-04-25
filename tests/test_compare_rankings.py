"""Tests for comparative ranking analysis utilities."""

from __future__ import annotations

import csv
import json
import math

from src.compare_rankings import (
    build_comparison_summary,
    export_comparison_results,
    jaccard_similarity,
    overlap_count,
    spearman_rank_correlation,
    top_k_nodes,
)


def test_top_k_and_overlap_helpers():
    scores_a = {"a": 0.9, "b": 0.5, "c": 0.1, "d": 0.0}
    scores_b = {"a": 0.3, "b": 0.2, "d": 0.1, "c": 0.0}

    assert top_k_nodes(scores_a, 2) == ["a", "b"]
    assert top_k_nodes(scores_b, 2) == ["a", "b"]
    assert overlap_count(scores_a, scores_b, 2) == 2

    assert math.isclose(jaccard_similarity({"a", "b"}, {"b", "c"}), 1 / 3)


def test_spearman_rank_correlation_extremes():
    scores_a = {"a": 3.0, "b": 2.0, "c": 1.0}
    scores_b = {"a": 3.0, "b": 2.0, "c": 1.0}
    scores_c = {"a": 1.0, "b": 2.0, "c": 3.0}

    assert math.isclose(spearman_rank_correlation(scores_a, scores_b), 1.0, abs_tol=1e-12)
    assert math.isclose(spearman_rank_correlation(scores_a, scores_c), -1.0, abs_tol=1e-12)


def test_build_summary_and_export(tmp_path):
    metric_scores = {
        "closeness": {0: 1.0, 1: 0.5, 2: 0.2, 3: 0.1},
        "betweenness": {0: 0.9, 1: 0.6, 2: 0.15, 3: 0.1},
        "eigenvalue": {0: 0.95, 1: 0.4, 2: 0.3, 3: 0.05},
    }

    summary = build_comparison_summary(metric_scores, k_values=(2, 3))
    assert set(summary["metrics"]) == {"closeness", "betweenness", "eigenvalue"}
    assert len(summary["pairwise"]) == 3

    csv_path = tmp_path / "comparison_results.csv"
    json_path = tmp_path / "comparison_results.json"

    result = export_comparison_results(
        metric_scores,
        csv_path=csv_path,
        json_path=json_path,
        k_values=(2, 3),
    )

    assert result["csv_path"] == csv_path
    assert result["json_path"] == json_path
    assert csv_path.exists()
    assert json_path.exists()

    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 6

    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert set(payload["metrics"]) == {"closeness", "betweenness", "eigenvalue"}
    assert len(payload["pairwise"]) == 3
