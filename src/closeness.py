"""Closeness centrality implementation for unweighted undirected graphs."""

from __future__ import annotations

from collections import deque
from typing import Dict, Literal

from src.graph_utils import NodeLabel, UndirectedGraph

DisconnectedPolicy = Literal["reachable_fraction", "strict_zero"]


def bfs_shortest_paths(graph: UndirectedGraph, source: NodeLabel) -> Dict[int, int]:
    """Compute shortest-path distances from source using BFS.

    Args:
        graph: Undirected unweighted graph.
        source: Node label or internal node ID.

    Returns:
        Mapping internal-node-id -> shortest path distance from source.
    """
    source_id = graph.resolve_node(source)

    distances: Dict[int, int] = {source_id: 0}
    queue = deque([source_id])

    while queue:
        current = queue.popleft()
        for neighbor in graph.neighbors(current):
            if neighbor in distances:
                continue
            distances[neighbor] = distances[current] + 1
            queue.append(neighbor)

    return distances


def compute_closeness_centrality(
    graph: UndirectedGraph,
    normalized: bool = True,
    handle_disconnected: DisconnectedPolicy = "reachable_fraction",
) -> Dict[NodeLabel, float]:
    """Compute exact closeness centrality by BFS from every node.

    Formula used:
        Let R(v) be nodes reachable from v (excluding v), r = |R(v)|,
        and S(v) = sum_{u in R(v)} dist(v, u).

        Base score:
            C_base(v) = r / S(v), if r > 0 else 0.

        If normalized=True:
            - reachable_fraction policy:
                C(v) = C_base(v) * (r / (n - 1))
                This matches the Wasserman-Faust style correction and
                safely handles disconnected graphs.
            - strict_zero policy:
                C(v) = C_base(v) if r == n - 1 else 0.

        If normalized=False:
            C(v) = C_base(v).

    Args:
        graph: Undirected unweighted graph.
        normalized: Whether to apply normalization.
        handle_disconnected: Disconnected-graph handling policy.

    Returns:
        Dictionary keyed by original node labels.
    """
    if handle_disconnected not in {"reachable_fraction", "strict_zero"}:
        raise ValueError(
            "handle_disconnected must be either 'reachable_fraction' or 'strict_zero'."
        )

    n = graph.number_of_nodes()
    if n == 0:
        return {}

    scores: Dict[NodeLabel, float] = {}

    for node_id in graph.node_ids():
        distances = bfs_shortest_paths(graph, node_id)
        reachable_count = len(distances) - 1

        if reachable_count <= 0:
            scores[graph.label_from_id(node_id)] = 0.0
            continue

        distance_sum = sum(distance for nid, distance in distances.items() if nid != node_id)
        if distance_sum <= 0:
            scores[graph.label_from_id(node_id)] = 0.0
            continue

        base_score = reachable_count / distance_sum

        if not normalized:
            score = base_score
        elif reachable_count == (n - 1):
            score = base_score
        elif handle_disconnected == "reachable_fraction":
            score = base_score * (reachable_count / (n - 1))
        else:
            score = 0.0

        scores[graph.label_from_id(node_id)] = float(score)

    return scores


def rank_centrality(scores: Dict[NodeLabel, float], top_k: int | None = None):
    """Return nodes sorted by centrality (descending score)."""
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if top_k is None:
        return ordered
    return ordered[:top_k]


__all__ = [
    "DisconnectedPolicy",
    "bfs_shortest_paths",
    "compute_closeness_centrality",
    "rank_centrality",
]
