"""Tests for plotting utilities."""

from __future__ import annotations

from src.compare_rankings import build_comparison_summary
from src.graph_utils import generate_bridge_community_graph
from src.visualize import (
    plot_centrality_histograms,
    plot_graph_highlighting_metric,
    plot_rank_correlation_heatmap,
    plot_runtime_by_family,
    plot_runtime_vs_graph_size,
    plot_topk_overlap_bars,
)


def _benchmark_rows_fixture():
    rows = []
    for family in ["path", "star"]:
        for n in [20, 40]:
            for metric, runtime in [
                ("closeness", 0.001 * n),
                ("betweenness", 0.0015 * n),
                ("eigenvalue", 0.0008 * n),
            ]:
                rows.append(
                    {
                        "graph_family": family,
                        "graph_label": f"{family}_{n}",
                        "metric_name": metric,
                        "num_nodes": str(n),
                        "runtime_seconds": str(runtime),
                        "status": "ok",
                    }
                )
    return rows


def test_plot_functions_generate_files(tmp_path):
    rows = _benchmark_rows_fixture()

    runtime_vs_size = plot_runtime_vs_graph_size(rows, tmp_path / "runtime_vs_size.png")
    runtime_by_family = plot_runtime_by_family(rows, tmp_path / "runtime_by_family.png")

    metric_scores = {
        "closeness": {0: 1.0, 1: 0.8, 2: 0.2, 3: 0.1},
        "betweenness": {0: 0.9, 1: 0.7, 2: 0.3, 3: 0.05},
        "eigenvalue": {0: 0.95, 1: 0.6, 2: 0.25, 3: 0.08},
    }

    hist_path = plot_centrality_histograms(metric_scores, tmp_path / "hist.png")

    summary = build_comparison_summary(metric_scores, k_values=(2,))
    overlap_path = plot_topk_overlap_bars(summary, tmp_path / "overlap.png", k=2)
    heatmap_path = plot_rank_correlation_heatmap(summary, tmp_path / "corr.png")

    graph = generate_bridge_community_graph(
        size_left=5,
        size_right=5,
        p_left=0.6,
        p_right=0.6,
        bridge_mode="single",
        seed=5,
    )
    graph_path = plot_graph_highlighting_metric(
        graph,
        metric_scores={node: float(index + 1) for index, node in enumerate(graph.node_labels())},
        metric_name="betweenness",
        output_path=tmp_path / "graph.png",
        top_k=4,
    )

    for path in [
        runtime_vs_size,
        runtime_by_family,
        hist_path,
        overlap_path,
        heatmap_path,
        graph_path,
    ]:
        assert path.exists()
        assert path.stat().st_size > 0
