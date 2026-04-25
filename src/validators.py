"""Validation and sanity-check helpers for implemented centrality metrics."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from src.betweenness import compute_betweenness_centrality
from src.closeness import compute_closeness_centrality
from src.eigenvalue import compute_eigenvalue_centrality
from src.graph_utils import (
    UndirectedGraph,
    generate_bridge_community_graph,
    generate_cycle_graph,
    generate_path_graph,
    generate_star_graph,
    to_networkx_graph,
)

try:
    import networkx as nx
except ImportError:  # pragma: no cover - optional dependency at runtime
    nx = None


def sort_scores(scores: Dict[object, float]) -> List[Tuple[object, float]]:
    """Sort a centrality dictionary in descending order."""
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _is_uniform(values: List[float], tol: float) -> bool:
    if not values:
        return True
    return (max(values) - min(values)) <= tol


def validate_closeness_path_pattern(n: int = 7) -> bool:
    """Path graph check: the middle node should be top-ranked for odd n."""
    graph = generate_path_graph(n)
    scores = compute_closeness_centrality(graph)
    ordered = sort_scores(scores)
    return ordered[0][0] == n // 2


def validate_closeness_star_pattern(n: int = 6) -> bool:
    """Star graph check: center node (0) should be top-ranked."""
    graph = generate_star_graph(n)
    scores = compute_closeness_centrality(graph)
    ordered = sort_scores(scores)
    return ordered[0][0] == 0


def validate_closeness_cycle_uniform(n: int = 8, tol: float = 1e-12) -> bool:
    """Cycle graph check: all closeness scores should be equal."""
    graph = generate_cycle_graph(n)
    scores = compute_closeness_centrality(graph)
    return _is_uniform(list(scores.values()), tol=tol)


def validate_closeness_disconnected_safe() -> bool:
    """Disconnected graph check: no crash and finite scores."""
    graph = UndirectedGraph(edges=[(0, 1), (2, 3)])

    scores_reachable = compute_closeness_centrality(
        graph,
        normalized=True,
        handle_disconnected="reachable_fraction",
    )
    scores_strict = compute_closeness_centrality(
        graph,
        normalized=True,
        handle_disconnected="strict_zero",
    )

    return (
        len(scores_reachable) == 4
        and all(math.isfinite(value) for value in scores_reachable.values())
        and all(math.isfinite(value) for value in scores_strict.values())
    )


def validate_betweenness_path_pattern(n: int = 7) -> bool:
    """Path graph check: middle node should have highest betweenness."""
    graph = generate_path_graph(n)
    scores = compute_betweenness_centrality(graph, normalized=True)
    ordered = sort_scores(scores)
    center = n // 2
    return ordered[0][0] == center and math.isclose(scores[0], 0.0, abs_tol=1e-12)


def validate_betweenness_star_pattern(n: int = 7) -> bool:
    """Star graph check: center should dominate betweenness."""
    graph = generate_star_graph(n)
    scores = compute_betweenness_centrality(graph, normalized=True)
    center = scores[0]
    leaves = [scores[node] for node in range(1, n)]

    return all(center > leaf for leaf in leaves) and all(
        math.isclose(leaf, 0.0, abs_tol=1e-12) for leaf in leaves
    )


def validate_betweenness_cycle_uniform(n: int = 8, tol: float = 1e-12) -> bool:
    """Cycle graph check: all nodes should have equal betweenness by symmetry."""
    graph = generate_cycle_graph(n)
    scores = compute_betweenness_centrality(graph, normalized=True)
    return _is_uniform(list(scores.values()), tol=tol)


def validate_betweenness_bridge_pattern(size_left: int = 4, size_right: int = 4) -> bool:
    """Bridge-community check: bridge nodes should dominate betweenness.

    Uses two dense communities (cliques) connected by a single bridge edge.
    """
    graph = generate_bridge_community_graph(
        size_left=size_left,
        size_right=size_right,
        p_left=1.0,
        p_right=1.0,
        bridge_mode="single",
        seed=3,
    )

    scores = compute_betweenness_centrality(graph, normalized=True)
    ordered = sort_scores(scores)

    expected_bridge_nodes = {size_left - 1, size_left}
    top_two_nodes = {ordered[0][0], ordered[1][0]}

    return top_two_nodes == expected_bridge_nodes


def validate_betweenness_disconnected_safe() -> bool:
    """Disconnected graph check: no crash and expected zero betweenness."""
    graph = UndirectedGraph(edges=[(0, 1), (2, 3)])
    scores = compute_betweenness_centrality(graph, normalized=True)
    return len(scores) == 4 and all(math.isclose(value, 0.0, abs_tol=1e-12) for value in scores.values())


def validate_eigenvalue_star_pattern(n: int = 7) -> bool:
    """Star graph check: center node (0) should dominate eigenvalue centrality."""
    graph = generate_star_graph(n)
    scores, metadata = compute_eigenvalue_centrality(graph, return_metadata=True)
    if not metadata["converged"]:
        return False

    ordered = sort_scores(scores)
    return ordered[0][0] == 0


def validate_eigenvalue_hub_pattern() -> bool:
    """Hub-heavy graph check: hub node should have highest eigenvalue centrality."""
    graph = UndirectedGraph(
        edges=[
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 2),
            (2, 3),
            (3, 4),
        ]
    )
    scores, metadata = compute_eigenvalue_centrality(graph, return_metadata=True)
    if not metadata["converged"]:
        return False

    ordered = sort_scores(scores)
    return ordered[0][0] == 0


def validate_eigenvalue_disconnected_safe() -> bool:
    """Disconnected graph check: no crash and finite values are produced."""
    graph = UndirectedGraph(edges=[(0, 1), (2, 3)])
    scores, metadata = compute_eigenvalue_centrality(graph, return_metadata=True)

    return (
        len(scores) == 4
        and all(math.isfinite(value) for value in scores.values())
        and metadata["iterations"] >= 0
    )


def compare_closeness_with_networkx(graph: UndirectedGraph) -> Optional[float]:
    """Return max absolute difference vs NetworkX closeness on tiny graphs.

    Returns:
        None if networkx is unavailable.
        Float max absolute difference otherwise.
    """
    if nx is None:
        return None

    nx_graph = to_networkx_graph(graph)
    ours = compute_closeness_centrality(
        graph,
        normalized=True,
        handle_disconnected="reachable_fraction",
    )
    theirs = nx.closeness_centrality(nx_graph, wf_improved=True)

    return max(abs(float(ours[node]) - float(theirs[node])) for node in ours.keys())


def compare_eigenvalue_with_networkx(graph: UndirectedGraph, tol: float = 1e-8) -> Optional[float]:
    """Return max absolute difference vs NetworkX eigenvector centrality on tiny graphs.

    Results are compared after scaling each score vector by its max value.

    Returns:
        None if networkx is unavailable.
    """
    if nx is None:
        return None

    nx_graph = to_networkx_graph(graph)

    ours, _ = compute_eigenvalue_centrality(
        graph,
        max_iter=2_000,
        tol=tol,
        normalized=True,
        return_metadata=True,
    )
    theirs = nx.eigenvector_centrality(nx_graph, max_iter=2_000, tol=tol)

    max_ours = max(ours.values()) if ours else 1.0
    max_theirs = max(theirs.values()) if theirs else 1.0

    ours_scaled = {node: value / max_ours for node, value in ours.items()}
    theirs_scaled = {node: value / max_theirs for node, value in theirs.items()}

    return max(abs(float(ours_scaled[node]) - float(theirs_scaled[node])) for node in ours.keys())


def compare_betweenness_with_networkx(graph: UndirectedGraph) -> Optional[float]:
    """Return max absolute difference vs NetworkX betweenness on tiny graphs.

    Returns:
        None if networkx is unavailable.
    """
    if nx is None:
        return None

    nx_graph = to_networkx_graph(graph)

    ours = compute_betweenness_centrality(
        graph,
        normalized=True,
        endpoints=False,
    )
    theirs = nx.betweenness_centrality(nx_graph, normalized=True, endpoints=False)

    return max(abs(float(ours[node]) - float(theirs[node])) for node in ours.keys())


def run_small_sanity_checks() -> Dict[str, bool]:
    """Run quick built-in sanity checks for all implemented centrality methods."""
    checks = {
        "closeness_path_center": validate_closeness_path_pattern(),
        "closeness_star_center": validate_closeness_star_pattern(),
        "closeness_cycle_uniform": validate_closeness_cycle_uniform(),
        "closeness_disconnected_safe": validate_closeness_disconnected_safe(),
        "betweenness_path_center": validate_betweenness_path_pattern(),
        "betweenness_star_center": validate_betweenness_star_pattern(),
        "betweenness_cycle_uniform": validate_betweenness_cycle_uniform(),
        "betweenness_bridge_nodes": validate_betweenness_bridge_pattern(),
        "betweenness_disconnected_safe": validate_betweenness_disconnected_safe(),
        "eigenvalue_star_center": validate_eigenvalue_star_pattern(),
        "eigenvalue_hub_center": validate_eigenvalue_hub_pattern(),
        "eigenvalue_disconnected_safe": validate_eigenvalue_disconnected_safe(),
    }
    checks["all_passed"] = all(checks.values())
    return checks


__all__ = [
    "compare_betweenness_with_networkx",
    "compare_closeness_with_networkx",
    "compare_eigenvalue_with_networkx",
    "run_small_sanity_checks",
    "sort_scores",
    "validate_betweenness_bridge_pattern",
    "validate_betweenness_cycle_uniform",
    "validate_betweenness_disconnected_safe",
    "validate_betweenness_path_pattern",
    "validate_betweenness_star_pattern",
    "validate_closeness_cycle_uniform",
    "validate_closeness_disconnected_safe",
    "validate_closeness_path_pattern",
    "validate_closeness_star_pattern",
    "validate_eigenvalue_disconnected_safe",
    "validate_eigenvalue_hub_pattern",
    "validate_eigenvalue_star_pattern",
]
