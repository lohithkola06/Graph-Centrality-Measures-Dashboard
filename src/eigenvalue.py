"""Eigenvalue centrality implementation via power iteration."""

from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np

from src.graph_utils import NodeLabel, UndirectedGraph


def adjacency_matvec(graph: UndirectedGraph, vector: np.ndarray) -> np.ndarray:
    """Compute y = A x using adjacency-list traversal.

    A is the unweighted undirected adjacency matrix.
    """
    result = np.zeros_like(vector, dtype=float)
    for node_id in graph.node_ids():
        total = 0.0
        for neighbor_id in graph.neighbors(node_id):
            total += float(vector[neighbor_id])
        result[node_id] = total
    return result


def compute_eigenvalue_centrality(
    graph: UndirectedGraph,
    max_iter: int = 1_000,
    tol: float = 1e-8,
    stability_shift: float = 1.0,
    normalized: bool = True,
    return_metadata: bool = True,
):
    """Compute eigenvalue centrality using power iteration.

    Method:
        Starting from a positive vector x^(0), repeat:
            x^(k+1) = (A + alpha I) x^(k) / ||(A + alpha I) x^(k)||_2
        until ||x^(k+1) - x^(k)||_2 < tol or max_iter is reached.
        The default alpha=1.0 stabilizes convergence on bipartite graphs
        without changing eigenvectors of A.

    Args:
        graph: Undirected unweighted graph.
        max_iter: Maximum iterations for power method.
        tol: Convergence threshold on L2 difference between consecutive vectors.
        stability_shift: Scalar alpha in (A + alpha I) iteration.
        normalized: If True, rescale final scores so max score is 1.
        return_metadata: If True, return (scores, metadata), else scores only.

    Returns:
        scores dict keyed by original node label.
        If return_metadata=True, also returns metadata dict with:
            - converged
            - iterations
            - final_delta
            - tolerance
            - max_iter
            - stability_shift
            - eigenvalue_estimate
            - delta_history
    """
    if max_iter <= 0:
        raise ValueError(f"max_iter must be > 0, got {max_iter}.")
    if tol <= 0:
        raise ValueError(f"tol must be > 0, got {tol}.")
    if stability_shift < 0:
        raise ValueError(f"stability_shift must be >= 0, got {stability_shift}.")

    n = graph.number_of_nodes()
    if n == 0:
        empty_result: Dict[NodeLabel, float] = {}
        metadata = {
            "converged": True,
            "iterations": 0,
            "final_delta": 0.0,
            "tolerance": float(tol),
            "max_iter": int(max_iter),
            "stability_shift": float(stability_shift),
            "eigenvalue_estimate": 0.0,
            "delta_history": [],
        }
        return (empty_result, metadata) if return_metadata else empty_result

    x = np.ones(n, dtype=float)
    x /= np.linalg.norm(x, ord=2)

    delta_history: List[float] = []
    converged = False

    for iteration in range(1, max_iter + 1):
        x_next = adjacency_matvec(graph, x)
        if stability_shift > 0.0:
            x_next = x_next + (stability_shift * x)
        norm = np.linalg.norm(x_next, ord=2)

        if norm == 0.0:
            x = np.zeros(n, dtype=float)
            delta_history.append(0.0)
            converged = True
            break

        x_next = x_next / norm
        delta = float(np.linalg.norm(x_next - x, ord=2))
        delta_history.append(delta)
        x = x_next

        if delta < tol:
            converged = True
            break

    if not converged:
        warnings.warn(
            "Power iteration did not converge within max_iter; results may be approximate.",
            RuntimeWarning,
            stacklevel=2,
        )

    ax = adjacency_matvec(graph, x)
    denom = float(np.dot(x, x))
    eigenvalue_estimate = float(np.dot(x, ax) / denom) if denom > 0 else 0.0

    score_vector = x.copy()
    if normalized:
        max_value = float(np.max(score_vector))
        if max_value > 0.0:
            score_vector = score_vector / max_value

    scores: Dict[NodeLabel, float] = {}
    for node_id in graph.node_ids():
        scores[graph.label_from_id(node_id)] = float(score_vector[node_id])

    metadata = {
        "converged": converged,
        "iterations": len(delta_history),
        "final_delta": delta_history[-1] if delta_history else 0.0,
        "tolerance": float(tol),
        "max_iter": int(max_iter),
        "stability_shift": float(stability_shift),
        "eigenvalue_estimate": eigenvalue_estimate,
        "delta_history": delta_history,
    }

    return (scores, metadata) if return_metadata else scores


__all__ = ["adjacency_matvec", "compute_eigenvalue_centrality"]
