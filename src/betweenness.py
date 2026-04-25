"""Betweenness centrality for unweighted undirected graphs.

This module implements the exact Brandes algorithm using BFS from each source.
"""

from __future__ import annotations

from collections import deque
from typing import Dict

from src.graph_utils import NodeLabel, UndirectedGraph


def compute_betweenness_centrality(
    graph: UndirectedGraph,
    normalized: bool = True,
    endpoints: bool = False,
) -> Dict[NodeLabel, float]:
    """Compute exact betweenness centrality via Brandes algorithm.

    The implementation targets unweighted, undirected graphs.

    Algorithm summary:
    1. For each source node s, run BFS to discover shortest-path levels.
    2. Track:
       - sigma[v]: number of shortest paths from s to v,
       - pred[v]: predecessors of v on shortest paths from s,
       - stack order of BFS visitation for reverse accumulation.
    3. Traverse nodes in reverse BFS order to accumulate dependencies
       and update betweenness scores.

    Endpoint policy:
    - ``endpoints=False`` (supported): standard node betweenness where
      source/target endpoints are not counted as intermediates.
    - ``endpoints=True`` is intentionally not implemented in this project.

    Normalization:
    - Raw undirected Brandes accumulation counts each unordered pair twice,
      so values are divided by 2.
    - If ``normalized=True`` and n > 2, scores are scaled by
      ``2 / ((n - 1) * (n - 2))``.

    Args:
        graph: Undirected unweighted graph.
        normalized: Whether to return normalized scores.
        endpoints: Whether to include path endpoints in centrality values.

    Returns:
        Dictionary keyed by original node labels.

    Raises:
        ValueError: if endpoints=True (not implemented policy).
    """
    if endpoints:
        raise ValueError(
            "endpoints=True is not supported in this implementation; "
            "use endpoints=False for standard betweenness centrality."
        )

    node_ids = graph.node_ids()
    n = len(node_ids)
    if n == 0:
        return {}

    betweenness_by_id: Dict[int, float] = {node_id: 0.0 for node_id in node_ids}

    for source in node_ids:
        stack = []
        predecessors = {node_id: [] for node_id in node_ids}
        shortest_path_count = {node_id: 0.0 for node_id in node_ids}
        shortest_path_count[source] = 1.0
        distance = {node_id: -1 for node_id in node_ids}
        distance[source] = 0

        queue = deque([source])

        # BFS discovers shortest-path DAG rooted at source.
        while queue:
            current = queue.popleft()
            stack.append(current)

            for neighbor in graph.neighbors(current):
                if distance[neighbor] < 0:
                    queue.append(neighbor)
                    distance[neighbor] = distance[current] + 1

                if distance[neighbor] == distance[current] + 1:
                    shortest_path_count[neighbor] += shortest_path_count[current]
                    predecessors[neighbor].append(current)

        dependency = {node_id: 0.0 for node_id in node_ids}

        # Reverse traversal accumulates dependencies from farthest nodes back.
        while stack:
            node = stack.pop()
            sigma_node = shortest_path_count[node]
            if sigma_node > 0.0:
                coeff = (1.0 + dependency[node]) / sigma_node
                for parent in predecessors[node]:
                    dependency[parent] += shortest_path_count[parent] * coeff

            if node != source:
                betweenness_by_id[node] += dependency[node]

    # Undirected accumulation counts each pair twice: (s, t) and (t, s).
    for node_id in betweenness_by_id:
        betweenness_by_id[node_id] *= 0.5

    if normalized:
        if n > 2:
            scale = 2.0 / ((n - 1) * (n - 2))
            for node_id in betweenness_by_id:
                betweenness_by_id[node_id] *= scale
        else:
            for node_id in betweenness_by_id:
                betweenness_by_id[node_id] = 0.0

    return {
        graph.label_from_id(node_id): float(score)
        for node_id, score in betweenness_by_id.items()
    }


__all__ = ["compute_betweenness_centrality"]
