"""Demo runner for centrality project deliverables.

Generates sample graphs, prints summaries, computes closeness/eigenvalue/betweenness
rankings, and writes a sample log file under outputs/logs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from src.betweenness import compute_betweenness_centrality
from src.closeness import compute_closeness_centrality, rank_centrality
from src.eigenvalue import compute_eigenvalue_centrality
from src.graph_utils import (
    UndirectedGraph,
    generate_barabasi_albert_graph,
    generate_bridge_community_graph,
    generate_cycle_graph,
    generate_path_graph,
    generate_star_graph,
    summarize_graph,
)
from src.validators import run_small_sanity_checks


def _format_top(scores: Dict[object, float], top_k: int = 5) -> List[str]:
    ordered = rank_centrality(scores, top_k=top_k)
    return [f"{label}: {value:.6f}" for label, value in ordered]


def _emit_section(lines: List[str], title: str, rows: Iterable[str]) -> None:
    lines.append(title)
    for row in rows:
        lines.append(f"  {row}")


def run_project_demo(log_dir: str | Path = "outputs/logs") -> Path:
    """Execute project demo and write a timestamped log file."""
    graphs: Dict[str, UndirectedGraph] = {
        "path_n6": generate_path_graph(6),
        "cycle_n8": generate_cycle_graph(8),
        "star_n7": generate_star_graph(7),
        "bridge_community": generate_bridge_community_graph(
            size_left=8,
            size_right=8,
            p_left=0.30,
            p_right=0.25,
            bridge_mode="single",
            seed=7,
        ),
        "barabasi_albert_n20_m2": generate_barabasi_albert_graph(n=20, m=2, seed=19),
    }

    report_lines: List[str] = []
    report_lines.append("=== Centrality Project Demo ===")
    report_lines.append(f"Run timestamp: {datetime.now().isoformat(timespec='seconds')}")

    print("=== Centrality Project Demo ===")

    for graph_name, graph in graphs.items():
        summary = summarize_graph(graph)
        closeness_scores = compute_closeness_centrality(
            graph,
            normalized=True,
            handle_disconnected="reachable_fraction",
        )
        betweenness_scores = compute_betweenness_centrality(
            graph,
            normalized=True,
            endpoints=False,
        )
        eigen_scores, eigen_meta = compute_eigenvalue_centrality(
            graph,
            max_iter=2_000,
            tol=1e-10,
            normalized=True,
            return_metadata=True,
        )

        print(f"\n[{graph_name}]")
        print(
            "summary:",
            {
                "nodes": summary.num_nodes,
                "edges": summary.num_edges,
                "min_degree": summary.min_degree,
                "max_degree": summary.max_degree,
                "avg_degree": round(summary.average_degree, 4),
                "connected": summary.connected,
            },
        )
        print("top closeness:", _format_top(closeness_scores, top_k=3))
        print("top betweenness:", _format_top(betweenness_scores, top_k=3))
        print("top eigenvalue:", _format_top(eigen_scores, top_k=3))
        print(
            "eigen metadata:",
            {
                "converged": eigen_meta["converged"],
                "iterations": eigen_meta["iterations"],
                "final_delta": f"{eigen_meta['final_delta']:.3e}",
                "eigenvalue_estimate": f"{eigen_meta['eigenvalue_estimate']:.6f}",
            },
        )

        _emit_section(report_lines, f"\n[{graph_name}]", [])
        _emit_section(
            report_lines,
            "summary",
            [
                f"nodes={summary.num_nodes}",
                f"edges={summary.num_edges}",
                f"min_degree={summary.min_degree}",
                f"max_degree={summary.max_degree}",
                f"average_degree={summary.average_degree:.6f}",
                f"connected={summary.connected}",
            ],
        )
        _emit_section(report_lines, "top_closeness", _format_top(closeness_scores, top_k=5))
        _emit_section(report_lines, "top_betweenness", _format_top(betweenness_scores, top_k=5))
        _emit_section(report_lines, "top_eigenvalue", _format_top(eigen_scores, top_k=5))
        _emit_section(
            report_lines,
            "eigen_metadata",
            [
                f"converged={eigen_meta['converged']}",
                f"iterations={eigen_meta['iterations']}",
                f"final_delta={eigen_meta['final_delta']:.6e}",
                f"eigenvalue_estimate={eigen_meta['eigenvalue_estimate']:.6f}",
            ],
        )

    checks = run_small_sanity_checks()
    _emit_section(report_lines, "\n[validator_checks]", [f"{k}={v}" for k, v in checks.items()])

    print("\nvalidator checks:", checks)

    target_dir = Path(log_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = target_dir / f"centrality_demo_{timestamp}.log"
    log_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    sample_path = Path("outputs/samples") / "latest_centrality_demo.log"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"\nDemo log written to: {log_path}")
    return log_path


if __name__ == "__main__":
    run_project_demo()
