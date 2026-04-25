"""Comparative ranking analysis for graph centrality score dictionaries."""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

NodeScoreDict = Mapping[object, float]


def top_k_nodes(scores: NodeScoreDict, k: int) -> List[object]:
    """Return top-k nodes sorted by descending score.

    Ties are broken deterministically using the string representation of nodes.
    """
    if k <= 0:
        return []

    ordered = sorted(scores.items(), key=lambda item: (-float(item[1]), repr(item[0])))
    return [node for node, _ in ordered[:k]]


def top_k_set(scores: NodeScoreDict, k: int) -> set[object]:
    """Return top-k node set."""
    return set(top_k_nodes(scores, k))


def overlap_count(scores_a: NodeScoreDict, scores_b: NodeScoreDict, k: int) -> int:
    """Return cardinality of intersection between two top-k node sets."""
    return len(top_k_set(scores_a, k).intersection(top_k_set(scores_b, k)))


def jaccard_similarity(set_a: set[object], set_b: set[object]) -> float:
    """Return Jaccard similarity of two sets.

    If both sets are empty, the function returns 1.0.
    """
    if not set_a and not set_b:
        return 1.0
    union = set_a.union(set_b)
    if not union:
        return 0.0
    return len(set_a.intersection(set_b)) / len(union)


def _average_tie_ranks(scores: NodeScoreDict) -> Dict[object, float]:
    """Assign ranks with average tie handling (Spearman-compatible ranks)."""
    ordered = sorted(scores.items(), key=lambda item: (-float(item[1]), repr(item[0])))
    ranks: Dict[object, float] = {}

    position = 1
    index = 0
    while index < len(ordered):
        score = float(ordered[index][1])
        tie_nodes = [ordered[index][0]]
        end = index + 1

        while end < len(ordered) and float(ordered[end][1]) == score:
            tie_nodes.append(ordered[end][0])
            end += 1

        start_rank = position
        end_rank = position + len(tie_nodes) - 1
        average_rank = (start_rank + end_rank) / 2.0

        for node in tie_nodes:
            ranks[node] = average_rank

        position = end_rank + 1
        index = end

    return ranks


def spearman_rank_correlation(scores_a: NodeScoreDict, scores_b: NodeScoreDict) -> float:
    """Compute Spearman rank correlation on common nodes.

    Ranks are computed with average tie handling. If fewer than two common nodes
    exist, or rank variance is zero for either vector, the function returns 0.0.
    """
    common_nodes = list(set(scores_a.keys()).intersection(scores_b.keys()))
    n = len(common_nodes)
    if n < 2:
        return 0.0

    ranks_a = _average_tie_ranks(scores_a)
    ranks_b = _average_tie_ranks(scores_b)

    values_a = [ranks_a[node] for node in common_nodes]
    values_b = [ranks_b[node] for node in common_nodes]

    mean_a = sum(values_a) / n
    mean_b = sum(values_b) / n

    centered_a = [value - mean_a for value in values_a]
    centered_b = [value - mean_b for value in values_b]

    var_a = sum(value * value for value in centered_a)
    var_b = sum(value * value for value in centered_b)

    if var_a <= 0.0 or var_b <= 0.0:
        return 0.0

    covariance = sum(a * b for a, b in zip(centered_a, centered_b))
    return covariance / ((var_a ** 0.5) * (var_b ** 0.5))


def build_comparison_summary(
    metric_scores: Mapping[str, NodeScoreDict],
    k_values: Sequence[int] = (5, 10),
) -> Dict[str, object]:
    """Build structured comparison summary across multiple metrics."""
    if not metric_scores:
        raise ValueError("metric_scores must not be empty.")

    unique_k_values = sorted({int(k) for k in k_values if int(k) > 0})
    if not unique_k_values:
        raise ValueError("k_values must include at least one positive integer.")

    metric_names = sorted(metric_scores.keys())

    top_k_by_metric: Dict[str, Dict[str, List[object]]] = {}
    for metric_name in metric_names:
        top_k_by_metric[metric_name] = {
            str(k): top_k_nodes(metric_scores[metric_name], k)
            for k in unique_k_values
        }

    pairwise: List[Dict[str, object]] = []

    for metric_a, metric_b in combinations(metric_names, 2):
        scores_a = metric_scores[metric_a]
        scores_b = metric_scores[metric_b]

        pair_item: Dict[str, object] = {
            "metric_a": metric_a,
            "metric_b": metric_b,
            "spearman_rank_correlation": float(spearman_rank_correlation(scores_a, scores_b)),
            "top_k": [],
        }

        for k in unique_k_values:
            top_nodes_a = top_k_nodes(scores_a, k)
            top_nodes_b = top_k_nodes(scores_b, k)
            set_a = set(top_nodes_a)
            set_b = set(top_nodes_b)
            overlap = len(set_a.intersection(set_b))

            pair_item["top_k"].append(
                {
                    "k": k,
                    "metric_a_top_nodes": top_nodes_a,
                    "metric_b_top_nodes": top_nodes_b,
                    "overlap_count": overlap,
                    "overlap_fraction": float(overlap / k),
                    "jaccard_similarity": float(jaccard_similarity(set_a, set_b)),
                }
            )

        pairwise.append(pair_item)

    return {
        "metrics": metric_names,
        "k_values": unique_k_values,
        "top_k_by_metric": top_k_by_metric,
        "pairwise": pairwise,
    }


def _flatten_summary_rows(summary: Mapping[str, object]) -> List[Dict[str, object]]:
    """Flatten structured summary into tabular rows for CSV export."""
    rows: List[Dict[str, object]] = []
    pairwise = summary["pairwise"]

    for pair_item in pairwise:  # type: ignore[assignment]
        metric_a = pair_item["metric_a"]
        metric_b = pair_item["metric_b"]
        spearman = pair_item["spearman_rank_correlation"]

        for top_k_item in pair_item["top_k"]:
            rows.append(
                {
                    "metric_a": metric_a,
                    "metric_b": metric_b,
                    "k": top_k_item["k"],
                    "overlap_count": top_k_item["overlap_count"],
                    "overlap_fraction": round(float(top_k_item["overlap_fraction"]), 10),
                    "jaccard_similarity": round(float(top_k_item["jaccard_similarity"]), 10),
                    "spearman_rank_correlation": round(float(spearman), 10),
                    "metric_a_top_nodes": json.dumps(top_k_item["metric_a_top_nodes"]),
                    "metric_b_top_nodes": json.dumps(top_k_item["metric_b_top_nodes"]),
                }
            )

    return rows


def export_comparison_results(
    metric_scores: Mapping[str, NodeScoreDict],
    csv_path: str | Path = "outputs/comparison_results.csv",
    json_path: str | Path = "outputs/comparison_results.json",
    k_values: Sequence[int] = (5, 10),
) -> Dict[str, object]:
    """Generate comparison summary and export it to CSV and JSON."""
    summary = build_comparison_summary(metric_scores=metric_scores, k_values=k_values)
    rows = _flatten_summary_rows(summary)

    csv_output = Path(csv_path)
    json_output = Path(json_path)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "metric_a",
        "metric_b",
        "k",
        "overlap_count",
        "overlap_fraction",
        "jaccard_similarity",
        "spearman_rank_correlation",
        "metric_a_top_nodes",
        "metric_b_top_nodes",
    ]

    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return {
        "summary": summary,
        "rows": rows,
        "csv_path": csv_output,
        "json_path": json_output,
    }


__all__ = [
    "build_comparison_summary",
    "export_comparison_results",
    "jaccard_similarity",
    "overlap_count",
    "spearman_rank_correlation",
    "top_k_nodes",
    "top_k_set",
]
