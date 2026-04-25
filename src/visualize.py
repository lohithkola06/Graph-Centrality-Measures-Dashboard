"""Plotting utilities for benchmark and centrality analysis outputs."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", str((Path("outputs") / ".mplconfig").resolve()))
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.graph_utils import UndirectedGraph, to_networkx_graph


def load_benchmark_rows(csv_path: str | Path) -> List[Dict[str, str]]:
    """Load benchmark rows from CSV output."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _valid_runtime_rows(rows: Iterable[Mapping[str, str]]) -> List[Mapping[str, str]]:
    valid = []
    for row in rows:
        status = row.get("status", "")
        if status not in {"ok", "warning"}:
            continue
        valid.append(row)
    return valid


def plot_runtime_vs_graph_size(rows: Sequence[Mapping[str, str]], output_path: str | Path) -> Path:
    """Create runtime-vs-size scatter plots for each metric."""
    valid_rows = _valid_runtime_rows(rows)
    metrics = sorted({row["metric_name"] for row in valid_rows})
    families = sorted({row["graph_family"] for row in valid_rows})

    if not metrics:
        raise ValueError("No valid benchmark rows available for runtime plot.")

    figure, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4), sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    colors = plt.get_cmap("tab10")

    for axis, metric in zip(axes, metrics):
        for idx, family in enumerate(families):
            family_rows = [
                row for row in valid_rows
                if row["metric_name"] == metric and row["graph_family"] == family
            ]
            if not family_rows:
                continue

            x_values = [int(row["num_nodes"]) for row in family_rows]
            y_values = [float(row["runtime_seconds"]) for row in family_rows]

            axis.scatter(
                x_values,
                y_values,
                alpha=0.8,
                color=colors(idx % 10),
                label=family,
            )

            order = np.argsort(x_values)
            axis.plot(
                [x_values[i] for i in order],
                [y_values[i] for i in order],
                color=colors(idx % 10),
                alpha=0.5,
                linewidth=1.2,
            )

        axis.set_title(f"Runtime vs Graph Size ({metric})")
        axis.set_xlabel("Number of Nodes")
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Runtime (seconds)")
    axes[-1].legend(loc="best", fontsize=8)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_runtime_by_family(rows: Sequence[Mapping[str, str]], output_path: str | Path) -> Path:
    """Create grouped-bar comparison of average runtime by graph family."""
    valid_rows = _valid_runtime_rows(rows)
    metrics = sorted({row["metric_name"] for row in valid_rows})
    families = sorted({row["graph_family"] for row in valid_rows})

    if not metrics or not families:
        raise ValueError("No valid benchmark rows available for family runtime plot.")

    family_metric_means: Dict[str, Dict[str, float]] = {}
    for family in families:
        family_metric_means[family] = {}
        for metric in metrics:
            values = [
                float(row["runtime_seconds"])
                for row in valid_rows
                if row["graph_family"] == family and row["metric_name"] == metric
            ]
            family_metric_means[family][metric] = float(np.mean(values)) if values else np.nan

    x = np.arange(len(families))
    width = 0.8 / len(metrics)

    figure, axis = plt.subplots(figsize=(12, 5))
    colors = plt.get_cmap("tab10")

    for index, metric in enumerate(metrics):
        offsets = x + (index - (len(metrics) - 1) / 2.0) * width
        heights = [family_metric_means[family][metric] for family in families]
        axis.bar(offsets, heights, width=width, label=metric, color=colors(index % 10), alpha=0.9)

    axis.set_title("Average Runtime by Graph Family")
    axis.set_xlabel("Graph Family")
    axis.set_ylabel("Average Runtime (seconds)")
    axis.set_xticks(x)
    axis.set_xticklabels(families, rotation=20, ha="right")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_centrality_histograms(
    metric_scores: Mapping[str, Mapping[object, float]],
    output_path: str | Path,
    bins: int = 20,
) -> Path:
    """Create histogram panels for centrality score distributions."""
    metric_names = sorted(metric_scores.keys())
    if not metric_names:
        raise ValueError("metric_scores must not be empty for histogram plot.")

    figure, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 4))
    if len(metric_names) == 1:
        axes = [axes]

    colors = plt.get_cmap("Set2")

    for idx, metric in enumerate(metric_names):
        values = list(metric_scores[metric].values())
        axes[idx].hist(values, bins=bins, color=colors(idx % 8), alpha=0.85, edgecolor="black")
        axes[idx].set_title(f"{metric.title()} Score Distribution")
        axes[idx].set_xlabel("Centrality Score")
        axes[idx].set_ylabel("Frequency")
        axes[idx].grid(alpha=0.2)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_topk_overlap_bars(
    comparison_summary: Mapping[str, object],
    output_path: str | Path,
    k: int,
) -> Path:
    """Create bar chart of top-k overlap metrics across centrality pairs."""
    pairwise = comparison_summary.get("pairwise", [])
    if not pairwise:
        raise ValueError("comparison_summary does not include pairwise results.")

    labels: List[str] = []
    overlap_values: List[float] = []
    jaccard_values: List[float] = []

    for pair_item in pairwise:
        metric_a = pair_item["metric_a"]
        metric_b = pair_item["metric_b"]
        match = None
        for top_k_item in pair_item["top_k"]:
            if int(top_k_item["k"]) == int(k):
                match = top_k_item
                break
        if match is None:
            continue

        labels.append(f"{metric_a} vs {metric_b}")
        overlap_values.append(float(match["overlap_fraction"]))
        jaccard_values.append(float(match["jaccard_similarity"]))

    if not labels:
        raise ValueError(f"No top-k entries found for k={k}.")

    x = np.arange(len(labels))
    width = 0.36

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(x - width / 2, overlap_values, width=width, label=f"Overlap Fraction (k={k})")
    axis.bar(x + width / 2, jaccard_values, width=width, label=f"Jaccard Similarity (k={k})")

    axis.set_title(f"Top-{k} Ranking Overlap Across Metrics")
    axis.set_xlabel("Metric Pair")
    axis.set_ylabel("Similarity")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_ylim(0, 1.05)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_rank_correlation_heatmap(
    comparison_summary: Mapping[str, object],
    output_path: str | Path,
) -> Path:
    """Create heatmap of pairwise Spearman rank correlations."""
    metrics = list(comparison_summary.get("metrics", []))
    pairwise = comparison_summary.get("pairwise", [])

    if not metrics or not pairwise:
        raise ValueError("comparison_summary must include metrics and pairwise fields.")

    index = {metric: idx for idx, metric in enumerate(metrics)}
    matrix = np.eye(len(metrics), dtype=float)

    for pair_item in pairwise:
        i = index[pair_item["metric_a"]]
        j = index[pair_item["metric_b"]]
        value = float(pair_item["spearman_rank_correlation"])
        matrix[i, j] = value
        matrix[j, i] = value

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_title("Spearman Rank Correlation Heatmap")
    axis.set_xticks(range(len(metrics)))
    axis.set_yticks(range(len(metrics)))
    axis.set_xticklabels(metrics, rotation=30, ha="right")
    axis.set_yticklabels(metrics)

    for i in range(len(metrics)):
        for j in range(len(metrics)):
            axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black")

    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_graph_highlighting_metric(
    graph: UndirectedGraph,
    metric_scores: Mapping[object, float],
    metric_name: str,
    output_path: str | Path,
    top_k: int = 8,
    seed: int = 42,
) -> Path:
    """Draw graph with node size/color proportional to centrality score."""
    nx_graph = to_networkx_graph(graph)

    nodes = list(nx_graph.nodes())
    scores = np.array([float(metric_scores.get(node, 0.0)) for node in nodes], dtype=float)
    if scores.size == 0:
        raise ValueError("metric_scores produced no nodes for graph visualization.")

    max_score = float(np.max(scores)) if np.max(scores) > 0 else 1.0
    normalized = scores / max_score
    node_sizes = 220 + (normalized * 1400)

    top_nodes = {
        node for node, _ in sorted(metric_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    }

    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover
        raise ImportError("networkx is required for graph drawing utilities.") from exc

    positions = nx.spring_layout(nx_graph, seed=seed)

    figure, axis = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_edges(nx_graph, positions, alpha=0.25, ax=axis)

    node_collection = nx.draw_networkx_nodes(
        nx_graph,
        positions,
        node_size=node_sizes,
        node_color=normalized,
        cmap="viridis",
        ax=axis,
        linewidths=[2.0 if node in top_nodes else 0.4 for node in nodes],
        edgecolors=["black" if node in top_nodes else "white" for node in nodes],
    )

    axis.set_title(f"Graph Visualization Highlighted by {metric_name.title()} Centrality")
    axis.set_axis_off()
    colorbar = figure.colorbar(node_collection, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Normalized Centrality Score")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


__all__ = [
    "load_benchmark_rows",
    "plot_centrality_histograms",
    "plot_graph_highlighting_metric",
    "plot_rank_correlation_heatmap",
    "plot_runtime_by_family",
    "plot_runtime_vs_graph_size",
    "plot_topk_overlap_bars",
]
