"""Benchmarking utilities for synthetic graph centrality experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence
import warnings

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
)


@dataclass(frozen=True)
class BenchmarkSpec:
    """Single benchmark graph specification."""

    family: str
    label: str
    config: Dict[str, Any]
    builder: Callable[[], UndirectedGraph]


def default_benchmark_specs(seed: int = 42) -> List[BenchmarkSpec]:
    """Return a default synthetic benchmark suite.

    Sizes are intentionally moderate so runs complete on typical student laptops.
    """
    specs: List[BenchmarkSpec] = []

    for n in [25, 50, 100, 200]:
        specs.append(
            BenchmarkSpec(
                family="path",
                label=f"path_n{n}",
                config={"n": n},
                builder=lambda n=n: generate_path_graph(n),
            )
        )

    for n in [25, 50, 100, 200]:
        specs.append(
            BenchmarkSpec(
                family="cycle",
                label=f"cycle_n{n}",
                config={"n": n},
                builder=lambda n=n: generate_cycle_graph(n),
            )
        )

    for n in [25, 50, 100, 200]:
        specs.append(
            BenchmarkSpec(
                family="star",
                label=f"star_n{n}",
                config={"n": n},
                builder=lambda n=n: generate_star_graph(n),
            )
        )

    for rows, cols in [(5, 5), (8, 8), (10, 10)]:
        specs.append(
            BenchmarkSpec(
                family="grid",
                label=f"grid_{rows}x{cols}",
                config={"rows": rows, "cols": cols},
                builder=lambda rows=rows, cols=cols: generate_grid_graph(rows, cols),
            )
        )

    er_configs = [(60, 0.08), (120, 0.05), (180, 0.03)]
    for index, (n, p) in enumerate(er_configs):
        er_seed = seed + 100 + index
        specs.append(
            BenchmarkSpec(
                family="erdos_renyi",
                label=f"erdos_renyi_n{n}_p{p}",
                config={"n": n, "p": p, "seed": er_seed},
                builder=lambda n=n, p=p, er_seed=er_seed: generate_erdos_renyi_graph(n=n, p=p, seed=er_seed),
            )
        )

    ba_configs = [(60, 2), (120, 2), (180, 3)]
    for index, (n, m) in enumerate(ba_configs):
        ba_seed = seed + 200 + index
        specs.append(
            BenchmarkSpec(
                family="barabasi_albert",
                label=f"barabasi_albert_n{n}_m{m}",
                config={"n": n, "m": m, "seed": ba_seed},
                builder=lambda n=n, m=m, ba_seed=ba_seed: generate_barabasi_albert_graph(n=n, m=m, seed=ba_seed),
            )
        )

    bridge_configs = [
        (15, 15, 0.35, 0.35, "single"),
        (25, 25, 0.25, 0.25, "single"),
        (35, 35, 0.18, 0.18, "single"),
    ]
    for index, (size_left, size_right, p_left, p_right, bridge_mode) in enumerate(bridge_configs):
        bridge_seed = seed + 300 + index
        specs.append(
            BenchmarkSpec(
                family="bridge_community",
                label=f"bridge_community_l{size_left}_r{size_right}",
                config={
                    "size_left": size_left,
                    "size_right": size_right,
                    "p_left": p_left,
                    "p_right": p_right,
                    "bridge_mode": bridge_mode,
                    "seed": bridge_seed,
                },
                builder=lambda size_left=size_left, size_right=size_right, p_left=p_left, p_right=p_right, bridge_mode=bridge_mode, bridge_seed=bridge_seed: generate_bridge_community_graph(
                    size_left=size_left,
                    size_right=size_right,
                    p_left=p_left,
                    p_right=p_right,
                    bridge_mode=bridge_mode,
                    seed=bridge_seed,
                ),
            )
        )

    return specs


def _metric_closeness(graph: UndirectedGraph) -> tuple[Dict[object, float], str, str]:
    scores = compute_closeness_centrality(
        graph,
        normalized=True,
        handle_disconnected="reachable_fraction",
    )
    return scores, "ok", ""


def _metric_betweenness(graph: UndirectedGraph) -> tuple[Dict[object, float], str, str]:
    scores = compute_betweenness_centrality(
        graph,
        normalized=True,
        endpoints=False,
    )
    return scores, "ok", ""


def _metric_eigenvalue(graph: UndirectedGraph) -> tuple[Dict[object, float], str, str]:
    # Benchmark output captures convergence via structured status/notes.
    # Suppress direct RuntimeWarning noise in console output.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        scores, metadata = compute_eigenvalue_centrality(
            graph,
            max_iter=2_000,
            tol=1e-8,
            normalized=True,
            return_metadata=True,
        )
    status = "ok" if metadata["converged"] else "warning"
    notes = (
        f"converged={metadata['converged']};"
        f"iterations={metadata['iterations']};"
        f"final_delta={metadata['final_delta']:.6e};"
        f"eigenvalue_estimate={metadata['eigenvalue_estimate']:.6f}"
    )
    return scores, status, notes


def _serialize_top_node(scores: Dict[object, float]) -> tuple[str, float]:
    if not scores:
        return "", 0.0
    top_node, top_score = max(scores.items(), key=lambda item: item[1])
    return repr(top_node), float(top_score)


def run_benchmark_suite(
    specs: Optional[Sequence[BenchmarkSpec]] = None,
    csv_path: str | Path = "outputs/benchmark_results.csv",
    json_path: str | Path = "outputs/benchmark_results.json",
    log_dir: str | Path = "outputs/logs",
) -> Dict[str, Any]:
    """Run benchmark suite and export runtime results to CSV/JSON.

    Each benchmark row captures graph metadata and execution details for one
    metric/graph pair.
    """
    suite = list(specs) if specs is not None else default_benchmark_specs()

    run_started_utc = datetime.now(timezone.utc)
    run_started_iso = run_started_utc.isoformat(timespec="seconds")

    csv_output = Path(csv_path)
    json_output = Path(json_path)
    logs_output_dir = Path(log_dir)

    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    logs_output_dir.mkdir(parents=True, exist_ok=True)

    metric_runners: Dict[str, Callable[[UndirectedGraph], tuple[Dict[object, float], str, str]]] = {
        "closeness": _metric_closeness,
        "betweenness": _metric_betweenness,
        "eigenvalue": _metric_eigenvalue,
    }

    rows: List[Dict[str, Any]] = []
    log_lines: List[str] = [
        "=== Centrality Benchmark Run ===",
        f"started_utc={run_started_iso}",
        f"suite_size={len(suite)}",
    ]

    for spec in suite:
        graph: UndirectedGraph | None = None
        graph_error: str | None = None

        try:
            graph = spec.builder()
            summary = summarize_graph(graph)
        except Exception as exc:  # pragma: no cover - defensive failure path
            graph_error = f"graph_build_error: {exc}"
            summary = None

        for metric_name, runner in metric_runners.items():
            runtime_seconds = 0.0
            status = "ok"
            notes = ""
            top_node = ""
            top_score = 0.0
            num_nodes = summary.num_nodes if summary else -1
            num_edges = summary.num_edges if summary else -1
            is_connected = summary.connected if summary else False

            if graph_error is not None:
                status = "graph_build_error"
                notes = graph_error
            else:
                started = perf_counter()
                try:
                    scores, status, notes = runner(graph)  # type: ignore[arg-type]
                    top_node, top_score = _serialize_top_node(scores)
                except Exception as exc:  # pragma: no cover - defensive failure path
                    status = "error"
                    notes = f"{type(exc).__name__}: {exc}"
                runtime_seconds = perf_counter() - started

            row = {
                "run_started_utc": run_started_iso,
                "graph_family": spec.family,
                "graph_label": spec.label,
                "graph_config": json.dumps(spec.config, sort_keys=True),
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "is_connected": is_connected,
                "metric_name": metric_name,
                "runtime_seconds": round(float(runtime_seconds), 10),
                "status": status,
                "top_node": top_node,
                "top_score": round(float(top_score), 10),
                "notes": notes,
            }
            rows.append(row)

            log_lines.append(
                " | ".join(
                    [
                        f"graph={spec.label}",
                        f"metric={metric_name}",
                        f"status={status}",
                        f"runtime={runtime_seconds:.6f}s",
                        f"top_node={top_node}",
                    ]
                )
            )

    fieldnames = [
        "run_started_utc",
        "graph_family",
        "graph_label",
        "graph_config",
        "num_nodes",
        "num_edges",
        "is_connected",
        "metric_name",
        "runtime_seconds",
        "status",
        "top_node",
        "top_score",
        "notes",
    ]

    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_output.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_started_utc": run_started_iso,
                "suite_size": len(suite),
                "result_count": len(rows),
                "results": rows,
            },
            handle,
            indent=2,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_output_dir / f"benchmark_run_{timestamp}.log"
    log_lines.append(f"rows_written={len(rows)}")
    log_lines.append(f"csv_output={csv_output}")
    log_lines.append(f"json_output={json_output}")
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    return {
        "run_started_utc": run_started_iso,
        "suite_size": len(suite),
        "result_count": len(rows),
        "csv_path": csv_output,
        "json_path": json_output,
        "log_path": log_path,
    }


def run_default_benchmarks() -> Dict[str, Any]:
    """Run the default benchmark configuration."""
    return run_benchmark_suite(specs=default_benchmark_specs())


__all__ = [
    "BenchmarkSpec",
    "default_benchmark_specs",
    "run_benchmark_suite",
    "run_default_benchmarks",
]
