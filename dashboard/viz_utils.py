"""Utilities for Streamlit dashboard graph loading and visualization."""

from __future__ import annotations

import csv
from io import StringIO
from typing import Dict, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.betweenness import compute_betweenness_centrality
from src.closeness import compute_closeness_centrality
from src.eigenvalue import compute_eigenvalue_centrality
from src.graph_utils import (
    UndirectedGraph,
    generate_barabasi_albert_graph,
    generate_bridge_community_graph,
    generate_cycle_graph,
    generate_erdos_renyi_graph,
    generate_grid_graph,
    generate_path_graph,
    generate_star_graph,
    summarize_graph,
    to_networkx_graph,
)


def parse_edge_list_text(content: str, delimiter: str | None = None) -> UndirectedGraph:
    """Parse edge-list text content into an undirected graph."""
    graph = UndirectedGraph()

    for line_no, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if delimiter is None or delimiter == "":
            parts = [token for token in line.replace(",", " ").split() if token]
        else:
            parts = [token.strip() for token in line.split(delimiter) if token.strip()]

        if len(parts) != 2:
            raise ValueError(
                f"Malformed edge-list line {line_no}: expected 2 columns, got {len(parts)}"
            )

        u_label, v_label = parts
        graph.add_edge(u_label, v_label)

    return graph


def generate_synthetic_graph(family: str, params: Mapping[str, object]) -> UndirectedGraph:
    """Generate a synthetic graph from dashboard-selected family/config."""
    if family == "path":
        return generate_path_graph(int(params["n"]))
    if family == "cycle":
        return generate_cycle_graph(int(params["n"]))
    if family == "star":
        return generate_star_graph(int(params["n"]))
    if family == "grid":
        return generate_grid_graph(int(params["rows"]), int(params["cols"]))
    if family == "erdos_renyi":
        return generate_erdos_renyi_graph(
            n=int(params["n"]),
            p=float(params["p"]),
            seed=int(params["seed"]),
        )
    if family == "barabasi_albert":
        return generate_barabasi_albert_graph(
            n=int(params["n"]),
            m=int(params["m"]),
            seed=int(params["seed"]),
        )
    if family == "bridge_community":
        return generate_bridge_community_graph(
            size_left=int(params["size_left"]),
            size_right=int(params["size_right"]),
            p_left=float(params["p_left"]),
            p_right=float(params["p_right"]),
            bridge_mode=str(params["bridge_mode"]),
            seed=int(params["seed"]),
        )

    raise ValueError(f"Unknown graph family: {family}")


def compute_metrics(graph: UndirectedGraph, selected_metrics: Iterable[str]) -> Dict[str, Dict[object, float]]:
    """Compute selected centrality metrics."""
    selected = set(selected_metrics)
    results: Dict[str, Dict[object, float]] = {}

    if "closeness" in selected:
        results["closeness"] = compute_closeness_centrality(
            graph,
            normalized=True,
            handle_disconnected="reachable_fraction",
        )

    if "betweenness" in selected:
        results["betweenness"] = compute_betweenness_centrality(
            graph,
            normalized=True,
            endpoints=False,
        )

    if "eigenvalue" in selected:
        eigen_scores, _ = compute_eigenvalue_centrality(
            graph,
            max_iter=2_500,
            tol=1e-8,
            normalized=True,
            return_metadata=True,
        )
        results["eigenvalue"] = eigen_scores

    return results


def metrics_to_table_rows(metric_scores: Mapping[str, Mapping[object, float]]) -> list[dict[str, object]]:
    """Convert multi-metric scores to tabular rows keyed by node."""
    nodes = sorted({node for scores in metric_scores.values() for node in scores.keys()}, key=repr)
    metrics = sorted(metric_scores.keys())

    rows = []
    for node in nodes:
        row: dict[str, object] = {"node": repr(node)}
        for metric in metrics:
            row[metric] = float(metric_scores[metric].get(node, 0.0))
        rows.append(row)
    return rows


def metric_scores_to_csv(metric_scores: Mapping[str, Mapping[object, float]]) -> str:
    """Serialize computed metric scores to CSV content."""
    rows = metrics_to_table_rows(metric_scores)
    if not rows:
        return ""

    output = StringIO()
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def build_graph_figure(
    graph: UndirectedGraph,
    metric_scores: Mapping[object, float],
    metric_name: str,
    top_k: int = 10,
    seed: int = 42,
):
    """Build matplotlib figure of graph colored by selected metric."""
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover
        raise ImportError("networkx is required for dashboard graph drawing.") from exc

    nx_graph = to_networkx_graph(graph)
    nodes = list(nx_graph.nodes())
    values = np.array([float(metric_scores.get(node, 0.0)) for node in nodes], dtype=float)

    max_value = float(np.max(values)) if values.size and float(np.max(values)) > 0 else 1.0
    normalized = values / max_value

    sizes = 250 + normalized * 1200
    top_nodes = {
        node for node, _ in sorted(metric_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    }

    positions = nx.spring_layout(nx_graph, seed=seed)

    figure, axis = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_edges(nx_graph, positions, alpha=0.25, ax=axis)

    node_collection = nx.draw_networkx_nodes(
        nx_graph,
        positions,
        node_size=sizes,
        node_color=normalized,
        cmap="plasma",
        ax=axis,
        linewidths=[2.0 if node in top_nodes else 0.4 for node in nodes],
        edgecolors=["black" if node in top_nodes else "white" for node in nodes],
    )

    axis.set_title(f"Graph Highlighted by {metric_name.title()} Centrality")
    axis.set_axis_off()
    colorbar = figure.colorbar(node_collection, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Normalized Score")
    figure.tight_layout()
    return figure


def summary_to_dict(graph: UndirectedGraph) -> dict[str, object]:
    """Convert graph summary dataclass into dictionary for display."""
    summary = summarize_graph(graph)
    return {
        "num_nodes": summary.num_nodes,
        "num_edges": summary.num_edges,
        "min_degree": summary.min_degree,
        "max_degree": summary.max_degree,
        "average_degree": round(summary.average_degree, 6),
        "connected": summary.connected,
    }


__all__ = [
    "build_graph_figure",
    "compute_metrics",
    "generate_synthetic_graph",
    "metric_scores_to_csv",
    "metrics_to_table_rows",
    "parse_edge_list_text",
    "summary_to_dict",
]
