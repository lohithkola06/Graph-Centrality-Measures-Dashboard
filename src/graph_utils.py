"""Graph infrastructure and synthetic graph generators.

Main scope: unweighted, undirected graphs with adjacency-list storage.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from random import Random
import re
from typing import Dict, Hashable, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

NodeLabel = Hashable
Edge = Tuple[NodeLabel, NodeLabel]


class UndirectedGraph:
    """Unweighted undirected graph backed by an adjacency list.

    Nodes are stored internally as contiguous integer IDs for algorithmic speed,
    while preserving a mapping to original labels for I/O and reporting.
    """

    def __init__(self, edges: Optional[Iterable[Edge]] = None, allow_self_loops: bool = False) -> None:
        self._adj: Dict[int, Set[int]] = {}
        self._label_to_id: Dict[NodeLabel, int] = {}
        self._id_to_label: List[NodeLabel] = []
        self.allow_self_loops = allow_self_loops

        if edges is not None:
            for u_label, v_label in edges:
                self.add_edge(u_label, v_label)

    def add_node(self, label: NodeLabel) -> int:
        """Add a node if missing and return its internal integer ID."""
        if label in self._label_to_id:
            return self._label_to_id[label]

        node_id = len(self._id_to_label)
        self._label_to_id[label] = node_id
        self._id_to_label.append(label)
        self._adj[node_id] = set()
        return node_id

    def add_edge(self, u_label: NodeLabel, v_label: NodeLabel) -> None:
        """Add an undirected edge between node labels.

        Duplicate edges are ignored automatically due to set-based adjacency.
        """
        u = self.add_node(u_label)
        v = self.add_node(v_label)

        if u == v and not self.allow_self_loops:
            raise ValueError(f"Self-loop detected for label {u_label!r}; self-loops are disabled.")

        self._adj[u].add(v)
        self._adj[v].add(u)

    def number_of_nodes(self) -> int:
        return len(self._adj)

    def number_of_edges(self) -> int:
        return sum(len(neigh) for neigh in self._adj.values()) // 2

    def node_ids(self) -> List[int]:
        return list(self._adj.keys())

    def node_labels(self) -> List[NodeLabel]:
        return list(self._id_to_label)

    def resolve_node(self, node: NodeLabel) -> int:
        """Resolve either an internal ID or an external label to an internal ID.

        Raises:
            KeyError: if node is unknown.
            ValueError: if an integer value is ambiguous as both a label and ID.
        """
        id_match = isinstance(node, int) and node in self._adj

        label_match = False
        label_id = -1
        try:
            label_id = self._label_to_id[node]
            label_match = True
        except (KeyError, TypeError):
            label_match = False

        if id_match and label_match and label_id != node:
            raise ValueError(
                f"Ambiguous node reference {node!r}: valid both as ID and as label mapped to ID {label_id}."
            )
        if id_match:
            return node
        if label_match:
            return label_id

        raise KeyError(f"Unknown node reference: {node!r}")

    def label_from_id(self, node_id: int) -> NodeLabel:
        if node_id not in self._adj:
            raise KeyError(f"Unknown node ID: {node_id}")
        return self._id_to_label[node_id]

    def neighbors(self, node_id: int) -> Set[int]:
        if node_id not in self._adj:
            raise KeyError(f"Unknown node ID: {node_id}")
        return set(self._adj[node_id])

    def degree(self, node_id: int) -> int:
        return len(self.neighbors(node_id))

    def has_edge(self, u_id: int, v_id: int) -> bool:
        if u_id not in self._adj or v_id not in self._adj:
            return False
        return v_id in self._adj[u_id]

    def adjacency_by_labels(self) -> Dict[NodeLabel, List[NodeLabel]]:
        """Return adjacency list keyed by original labels."""
        result: Dict[NodeLabel, List[NodeLabel]] = {}
        for node_id, neighbors in self._adj.items():
            label = self.label_from_id(node_id)
            result[label] = [self.label_from_id(nei_id) for nei_id in sorted(neighbors)]
        return result

    def iter_edges(self) -> Iterator[Edge]:
        """Yield each undirected edge once, using original labels."""
        for u_id, neighbors in self._adj.items():
            for v_id in neighbors:
                if u_id < v_id:
                    yield (self.label_from_id(u_id), self.label_from_id(v_id))


@dataclass(frozen=True)
class GraphSummary:
    num_nodes: int
    num_edges: int
    min_degree: int
    max_degree: int
    average_degree: float
    connected: bool


def _require_int(name: str, value: int, *, min_value: int = 1) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}.")


def _require_probability(name: str, value: float) -> None:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}.")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")


def build_graph_from_edges(edges: Iterable[Edge], allow_self_loops: bool = False) -> UndirectedGraph:
    """Build a graph from an iterable of edge tuples."""
    return UndirectedGraph(edges=edges, allow_self_loops=allow_self_loops)


def load_graph_from_edge_list(
    path: Path | str,
    delimiter: Optional[str] = None,
    comment_prefix: str = "#",
    allow_self_loops: bool = False,
) -> UndirectedGraph:
    """Load an unweighted undirected graph from an edge-list text file.

    Accepted formats per line:
    - whitespace separated, e.g. "u v"
    - comma separated when delimiter is None, e.g. "u,v"
    - custom delimiter if provided explicitly

    Blank lines and comment lines (prefix '#') are ignored.

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError: for malformed lines or invalid self-loops.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Edge-list file not found: {file_path}")

    graph = UndirectedGraph(allow_self_loops=allow_self_loops)

    with file_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if comment_prefix and line.startswith(comment_prefix):
                continue

            if delimiter is None:
                parts = [part for part in re.split(r"[\s,]+", line) if part]
            else:
                parts = [part.strip() for part in line.split(delimiter) if part.strip()]

            if len(parts) != 2:
                raise ValueError(
                    f"Malformed edge-list line {line_no}: expected exactly 2 columns, got {len(parts)} -> {raw_line.rstrip()!r}"
                )

            u_label, v_label = parts
            if u_label == v_label and not allow_self_loops:
                raise ValueError(
                    f"Invalid self-loop at line {line_no}: {u_label!r} {v_label!r}."
                )

            graph.add_edge(u_label, v_label)

    return graph


def is_connected(graph: UndirectedGraph) -> bool:
    """Return True if graph is connected (empty graph is treated as connected)."""
    n = graph.number_of_nodes()
    if n == 0:
        return True

    start = graph.node_ids()[0]
    visited = {start}
    queue = deque([start])

    while queue:
        current = queue.popleft()
        for neighbor in graph.neighbors(current):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(neighbor)

    return len(visited) == n


def summarize_graph(graph: UndirectedGraph) -> GraphSummary:
    """Compute basic graph summary statistics."""
    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    if n == 0:
        return GraphSummary(
            num_nodes=0,
            num_edges=0,
            min_degree=0,
            max_degree=0,
            average_degree=0.0,
            connected=True,
        )

    degrees = [graph.degree(node_id) for node_id in graph.node_ids()]
    return GraphSummary(
        num_nodes=n,
        num_edges=m,
        min_degree=min(degrees),
        max_degree=max(degrees),
        average_degree=(2.0 * m) / n,
        connected=is_connected(graph),
    )


def generate_path_graph(n: int) -> UndirectedGraph:
    """Generate a path graph with nodes 0..n-1."""
    _require_int("n", n, min_value=1)
    graph = UndirectedGraph()

    for i in range(n):
        graph.add_node(i)
    for i in range(n - 1):
        graph.add_edge(i, i + 1)
    return graph


def generate_cycle_graph(n: int) -> UndirectedGraph:
    """Generate a cycle graph with nodes 0..n-1."""
    _require_int("n", n, min_value=3)
    graph = generate_path_graph(n)
    graph.add_edge(0, n - 1)
    return graph


def generate_star_graph(n: int) -> UndirectedGraph:
    """Generate a star graph with node 0 as center and n-1 leaves."""
    _require_int("n", n, min_value=2)
    graph = UndirectedGraph()

    for i in range(n):
        graph.add_node(i)
    for leaf in range(1, n):
        graph.add_edge(0, leaf)
    return graph


def generate_grid_graph(rows: int, cols: int) -> UndirectedGraph:
    """Generate a 2D grid graph; labels are (row, col) tuples."""
    _require_int("rows", rows, min_value=1)
    _require_int("cols", cols, min_value=1)

    graph = UndirectedGraph()

    for r in range(rows):
        for c in range(cols):
            graph.add_node((r, c))

    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                graph.add_edge((r, c), (r + 1, c))
            if c + 1 < cols:
                graph.add_edge((r, c), (r, c + 1))

    return graph


def generate_erdos_renyi_graph(n: int, p: float, seed: Optional[int] = None) -> UndirectedGraph:
    """Generate a G(n, p) Erdos-Renyi random graph."""
    _require_int("n", n, min_value=1)
    _require_probability("p", p)

    rng = Random(seed)
    graph = UndirectedGraph()
    for i in range(n):
        graph.add_node(i)

    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() <= p:
                graph.add_edge(u, v)

    return graph


def generate_barabasi_albert_graph(n: int, m: int, seed: Optional[int] = None) -> UndirectedGraph:
    """Generate a Barabasi-Albert preferential attachment graph.

    Args:
        n: Total number of nodes.
        m: Number of edges added for each new node.
    """
    _require_int("n", n, min_value=2)
    _require_int("m", m, min_value=1)
    if m >= n:
        raise ValueError(f"m must satisfy m < n, got m={m}, n={n}.")

    rng = Random(seed)

    initial_size = m + 1
    graph = UndirectedGraph()
    for i in range(initial_size):
        graph.add_node(i)

    # Start from a clique so every node has non-zero degree.
    for u in range(initial_size):
        for v in range(u + 1, initial_size):
            graph.add_edge(u, v)

    repeated_nodes: List[int] = []
    for node_id in graph.node_ids():
        repeated_nodes.extend([node_id] * graph.degree(node_id))

    for new_label in range(initial_size, n):
        graph.add_node(new_label)
        targets: Set[int] = set()

        while len(targets) < m:
            targets.add(rng.choice(repeated_nodes))

        for target_id in targets:
            graph.add_edge(new_label, graph.label_from_id(target_id))

        # Maintain a degree-proportional sampling list.
        new_id = graph.resolve_node(new_label)
        repeated_nodes.extend(targets)
        repeated_nodes.extend([new_id] * m)

    return graph


def _add_connected_community(
    graph: UndirectedGraph,
    node_labels: Sequence[int],
    p: float,
    rng: Random,
) -> None:
    """Create a connected community with random extra edges.

    A path backbone ensures internal connectivity even when p=0.
    """
    for label in node_labels:
        graph.add_node(label)

    for idx in range(len(node_labels) - 1):
        graph.add_edge(node_labels[idx], node_labels[idx + 1])

    for i in range(len(node_labels)):
        for j in range(i + 1, len(node_labels)):
            u = node_labels[i]
            v = node_labels[j]
            u_id = graph.resolve_node(u)
            v_id = graph.resolve_node(v)
            if graph.has_edge(u_id, v_id):
                continue
            if rng.random() <= p:
                graph.add_edge(u, v)


def generate_bridge_community_graph(
    size_left: int,
    size_right: int,
    p_left: float,
    p_right: float,
    bridge_mode: str = "single",
    seed: Optional[int] = None,
) -> UndirectedGraph:
    """Generate two communities connected by bridge edge(s).

    Communities are guaranteed internally connected via path backbones,
    then densified by random intra-community edges.

    Args:
        size_left: Number of nodes in the left community.
        size_right: Number of nodes in the right community.
        p_left: Extra-edge probability in left community.
        p_right: Extra-edge probability in right community.
        bridge_mode: 'single' (default) or 'double'.
    """
    _require_int("size_left", size_left, min_value=1)
    _require_int("size_right", size_right, min_value=1)
    _require_probability("p_left", p_left)
    _require_probability("p_right", p_right)

    if bridge_mode not in {"single", "double"}:
        raise ValueError("bridge_mode must be 'single' or 'double'.")

    rng = Random(seed)
    graph = UndirectedGraph()

    left_nodes = list(range(size_left))
    right_nodes = list(range(size_left, size_left + size_right))

    _add_connected_community(graph, left_nodes, p_left, rng)
    _add_connected_community(graph, right_nodes, p_right, rng)

    graph.add_edge(left_nodes[-1], right_nodes[0])
    if bridge_mode == "double" and len(left_nodes) > 1 and len(right_nodes) > 1:
        graph.add_edge(left_nodes[0], right_nodes[-1])

    return graph


def to_networkx_graph(graph: UndirectedGraph):
    """Convert to a NetworkX graph (optional utility)."""
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError("networkx is required for conversion utilities.") from exc

    nx_graph = nx.Graph()
    for node_label in graph.node_labels():
        nx_graph.add_node(node_label)
    for u_label, v_label in graph.iter_edges():
        nx_graph.add_edge(u_label, v_label)
    return nx_graph


__all__ = [
    "Edge",
    "GraphSummary",
    "NodeLabel",
    "UndirectedGraph",
    "build_graph_from_edges",
    "generate_barabasi_albert_graph",
    "generate_bridge_community_graph",
    "generate_cycle_graph",
    "generate_erdos_renyi_graph",
    "generate_grid_graph",
    "generate_path_graph",
    "generate_star_graph",
    "is_connected",
    "load_graph_from_edge_list",
    "summarize_graph",
    "to_networkx_graph",
]
