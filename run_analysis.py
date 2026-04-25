"""Run comparative ranking analysis and generate publication-ready plots."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from typing import Dict

os.environ.setdefault("MPLCONFIGDIR", str((Path("outputs") / ".mplconfig").resolve()))

from src.betweenness import compute_betweenness_centrality
from src.closeness import compute_closeness_centrality
from src.compare_rankings import export_comparison_results
from src.eigenvalue import compute_eigenvalue_centrality
from src.experiments import run_default_benchmarks
from src.graph_utils import generate_bridge_community_graph
from src.visualize import (
    load_benchmark_rows,
    plot_centrality_histograms,
    plot_graph_highlighting_metric,
    plot_rank_correlation_heatmap,
    plot_runtime_by_family,
    plot_runtime_vs_graph_size,
    plot_topk_overlap_bars,
)


def run_analysis() -> Dict[str, object]:
    """Execute ranking comparison and generate analysis outputs."""
    benchmark_csv = Path("outputs/benchmark_results.csv")
    benchmark_json = Path("outputs/benchmark_results.json")

    if not benchmark_csv.exists() or not benchmark_json.exists():
        run_default_benchmarks()

    benchmark_rows = load_benchmark_rows(benchmark_csv)

    analysis_graph = generate_bridge_community_graph(
        size_left=20,
        size_right=20,
        p_left=0.22,
        p_right=0.22,
        bridge_mode="single",
        seed=101,
    )

    closeness_scores = compute_closeness_centrality(
        analysis_graph,
        normalized=True,
        handle_disconnected="reachable_fraction",
    )
    betweenness_scores = compute_betweenness_centrality(
        analysis_graph,
        normalized=True,
        endpoints=False,
    )
    eigenvalue_scores, eigenvalue_meta = compute_eigenvalue_centrality(
        analysis_graph,
        max_iter=2_500,
        tol=1e-9,
        normalized=True,
        return_metadata=True,
    )

    metric_scores = {
        "closeness": closeness_scores,
        "betweenness": betweenness_scores,
        "eigenvalue": eigenvalue_scores,
    }

    comparison_export = export_comparison_results(
        metric_scores=metric_scores,
        csv_path="outputs/comparison_results.csv",
        json_path="outputs/comparison_results.json",
        k_values=(5, 10),
    )
    comparison_summary = comparison_export["summary"]

    plots_dir = Path("outputs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = {
        "runtime_vs_size": plot_runtime_vs_graph_size(
            benchmark_rows,
            plots_dir / "runtime_vs_graph_size.png",
        ),
        "runtime_by_family": plot_runtime_by_family(
            benchmark_rows,
            plots_dir / "runtime_by_family.png",
        ),
        "centrality_histograms": plot_centrality_histograms(
            metric_scores,
            plots_dir / "centrality_histograms.png",
            bins=18,
        ),
        "topk_overlaps": plot_topk_overlap_bars(
            comparison_summary,
            plots_dir / "topk_overlap_bars.png",
            k=10,
        ),
        "rank_correlations": plot_rank_correlation_heatmap(
            comparison_summary,
            plots_dir / "rank_correlation_heatmap.png",
        ),
        "graph_highlight": plot_graph_highlighting_metric(
            analysis_graph,
            betweenness_scores,
            metric_name="betweenness",
            output_path=plots_dir / "graph_highlight_betweenness.png",
            top_k=10,
            seed=77,
        ),
    }

    report_figures_dir = Path("report/figures")
    report_figures_dir.mkdir(parents=True, exist_ok=True)
    copied_figures = []
    for key, plot_path in plot_paths.items():
        target = report_figures_dir / plot_path.name
        shutil.copy2(plot_path, target)
        copied_figures.append(target)

    logs_dir = Path("outputs/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"analysis_run_{timestamp}.log"

    log_lines = [
        "=== Centrality Analysis Run ===",
        f"timestamp={datetime.now().isoformat(timespec='seconds')}",
        f"benchmark_rows={len(benchmark_rows)}",
        f"comparison_csv={comparison_export['csv_path']}",
        f"comparison_json={comparison_export['json_path']}",
        f"eigenvalue_converged={eigenvalue_meta['converged']}",
        f"eigenvalue_iterations={eigenvalue_meta['iterations']}",
        "plots_generated:",
    ]
    log_lines.extend([f"  {name}: {path}" for name, path in plot_paths.items()])
    log_lines.append("figures_copied:")
    log_lines.extend([f"  {path}" for path in copied_figures])

    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    return {
        "comparison_csv": comparison_export["csv_path"],
        "comparison_json": comparison_export["json_path"],
        "plots": plot_paths,
        "copied_figures": copied_figures,
        "log_path": log_path,
    }


def main() -> None:
    result = run_analysis()
    print("Analysis run completed successfully.")
    print(f"Comparison CSV: {result['comparison_csv']}")
    print(f"Comparison JSON: {result['comparison_json']}")
    print(f"Analysis log: {result['log_path']}")
    print("Plots:")
    for name, path in result["plots"].items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
