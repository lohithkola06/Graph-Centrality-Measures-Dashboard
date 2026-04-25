"""Unit tests for graph infrastructure and centrality implementations."""

from __future__ import annotations

import math

import pytest

from src.betweenness import compute_betweenness_centrality
from src.closeness import compute_closeness_centrality
from src.eigenvalue import compute_eigenvalue_centrality
from src.graph_utils import (
    UndirectedGraph,
    build_graph_from_edges,
    generate_bridge_community_graph,
    generate_cycle_graph,
    generate_grid_graph,
    generate_path_graph,
    generate_star_graph,
    is_connected,
    load_graph_from_edge_list,
    summarize_graph,
)
from src.validators import (
    compare_betweenness_with_networkx,
    compare_closeness_with_networkx,
    compare_eigenvalue_with_networkx,
    run_small_sanity_checks,
)


def test_edge_list_loader_and_summary(tmp_path):
    edge_file = tmp_path / "triangle.edgelist"
    edge_file.write_text("A B\nB C\nC A\n", encoding="utf-8")

    graph = load_graph_from_edge_list(edge_file)
    summary = summarize_graph(graph)

    assert summary.num_nodes == 3
    assert summary.num_edges == 3
    assert summary.min_degree == 2
    assert summary.max_degree == 2
    assert math.isclose(summary.average_degree, 2.0)
    assert summary.connected is True


def test_edge_list_loader_rejects_malformed_line(tmp_path):
    bad_file = tmp_path / "bad.edgelist"
    bad_file.write_text("u v w\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_graph_from_edge_list(bad_file)


def test_generators_and_connectedness_basic_properties():
    path_graph = generate_path_graph(5)
    cycle_graph = generate_cycle_graph(6)
    star_graph = generate_star_graph(6)
    grid_graph = generate_grid_graph(2, 3)

    assert path_graph.number_of_nodes() == 5
    assert path_graph.number_of_edges() == 4
    assert is_connected(path_graph)

    assert cycle_graph.number_of_nodes() == 6
    assert cycle_graph.number_of_edges() == 6
    assert is_connected(cycle_graph)

    assert star_graph.number_of_nodes() == 6
    assert star_graph.number_of_edges() == 5
    assert is_connected(star_graph)

    assert grid_graph.number_of_nodes() == 6
    assert grid_graph.number_of_edges() == 7
    assert is_connected(grid_graph)


def test_bridge_community_generator_connected():
    graph = generate_bridge_community_graph(
        size_left=6,
        size_right=7,
        p_left=0.2,
        p_right=0.2,
        bridge_mode="single",
        seed=123,
    )
    assert graph.number_of_nodes() == 13
    assert is_connected(graph)


def test_connectedness_detects_disconnected_graph():
    graph = build_graph_from_edges([(0, 1), (2, 3)])
    assert is_connected(graph) is False


def test_closeness_path_graph_behavior():
    graph = generate_path_graph(5)
    scores = compute_closeness_centrality(graph)

    assert scores[2] > scores[1] > scores[0]
    assert scores[2] > scores[3] > scores[4]
    assert math.isclose(scores[0], scores[4], rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(scores[1], scores[3], rel_tol=0.0, abs_tol=1e-12)

    expected = {0: 0.4, 1: 4.0 / 7.0, 2: 2.0 / 3.0, 3: 4.0 / 7.0, 4: 0.4}
    for node, value in expected.items():
        assert math.isclose(scores[node], value, rel_tol=1e-9, abs_tol=1e-9)


def test_closeness_star_graph_behavior():
    graph = generate_star_graph(6)
    scores = compute_closeness_centrality(graph)

    center = scores[0]
    leaves = [scores[i] for i in range(1, 6)]

    assert all(math.isclose(leaves[0], leaf, rel_tol=0.0, abs_tol=1e-12) for leaf in leaves)
    assert all(center > leaf for leaf in leaves)
    assert math.isclose(center, 1.0, rel_tol=1e-9, abs_tol=1e-9)


def test_closeness_cycle_graph_uniformity():
    graph = generate_cycle_graph(6)
    scores = compute_closeness_centrality(graph)

    values = list(scores.values())
    assert max(values) - min(values) <= 1e-12


def test_closeness_disconnected_policy_behavior():
    graph = UndirectedGraph(edges=[(0, 1), (2, 3)])

    reachable_fraction_scores = compute_closeness_centrality(
        graph,
        normalized=True,
        handle_disconnected="reachable_fraction",
    )
    strict_zero_scores = compute_closeness_centrality(
        graph,
        normalized=True,
        handle_disconnected="strict_zero",
    )

    assert all(
        math.isclose(v, 1.0 / 3.0, rel_tol=1e-9, abs_tol=1e-9)
        for v in reachable_fraction_scores.values()
    )
    assert all(math.isclose(v, 0.0, rel_tol=0.0, abs_tol=1e-12) for v in strict_zero_scores.values())


def test_betweenness_path_graph_behavior():
    graph = generate_path_graph(5)
    scores = compute_betweenness_centrality(graph, normalized=True)

    assert scores[2] > scores[1] > scores[0]
    assert scores[2] > scores[3] > scores[4]
    assert math.isclose(scores[0], 0.0, abs_tol=1e-12)
    assert math.isclose(scores[4], 0.0, abs_tol=1e-12)

    expected = {0: 0.0, 1: 0.5, 2: 2.0 / 3.0, 3: 0.5, 4: 0.0}
    for node, value in expected.items():
        assert math.isclose(scores[node], value, rel_tol=1e-9, abs_tol=1e-9)


def test_betweenness_star_graph_behavior():
    graph = generate_star_graph(6)
    scores = compute_betweenness_centrality(graph, normalized=True)

    center = scores[0]
    leaves = [scores[i] for i in range(1, 6)]

    assert math.isclose(center, 1.0, rel_tol=1e-9, abs_tol=1e-9)
    assert all(math.isclose(leaf, 0.0, abs_tol=1e-12) for leaf in leaves)


def test_betweenness_cycle_graph_uniformity():
    graph = generate_cycle_graph(8)
    scores = compute_betweenness_centrality(graph, normalized=True)

    values = list(scores.values())
    assert max(values) - min(values) <= 1e-12


def test_betweenness_bridge_nodes_dominate():
    graph = generate_bridge_community_graph(
        size_left=4,
        size_right=4,
        p_left=1.0,
        p_right=1.0,
        bridge_mode="single",
        seed=7,
    )
    scores = compute_betweenness_centrality(graph, normalized=True)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_two = {ordered[0][0], ordered[1][0]}
    assert top_two == {3, 4}


def test_betweenness_disconnected_graph_safe():
    graph = UndirectedGraph(edges=[(0, 1), (2, 3)])
    scores = compute_betweenness_centrality(graph, normalized=True)

    assert len(scores) == 4
    assert all(math.isclose(value, 0.0, abs_tol=1e-12) for value in scores.values())


def test_betweenness_endpoints_policy_explicit():
    graph = generate_star_graph(6)
    with pytest.raises(ValueError):
        compute_betweenness_centrality(graph, endpoints=True)


def test_eigenvalue_star_graph_behavior():
    graph = generate_star_graph(7)
    scores, metadata = compute_eigenvalue_centrality(graph, max_iter=2_000, tol=1e-10, return_metadata=True)

    assert metadata["converged"] is True
    assert metadata["iterations"] > 0
    assert metadata["final_delta"] <= 1e-10

    center = scores[0]
    leaves = [scores[i] for i in range(1, 7)]

    assert all(math.isclose(leaves[0], leaf, rel_tol=0.0, abs_tol=1e-8) for leaf in leaves)
    assert all(center > leaf for leaf in leaves)


def test_eigenvalue_hub_heavy_graph_behavior():
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
    scores, metadata = compute_eigenvalue_centrality(graph, max_iter=2_000, tol=1e-10, return_metadata=True)

    assert metadata["converged"] is True
    assert scores[0] == max(scores.values())


def test_eigenvalue_disconnected_graph_safe():
    graph = UndirectedGraph(edges=[(0, 1), (2, 3)])
    scores, metadata = compute_eigenvalue_centrality(graph, max_iter=2_000, tol=1e-10, return_metadata=True)

    assert len(scores) == 4
    assert metadata["iterations"] >= 0
    assert all(math.isfinite(value) for value in scores.values())


def test_optional_networkx_comparisons_on_tiny_graphs():
    graph = generate_path_graph(6)

    closeness_diff = compare_closeness_with_networkx(graph)
    eigenvalue_diff = compare_eigenvalue_with_networkx(graph)
    betweenness_diff = compare_betweenness_with_networkx(graph)

    if closeness_diff is None or eigenvalue_diff is None or betweenness_diff is None:
        pytest.skip("networkx unavailable in runtime environment")

    assert closeness_diff <= 1e-9
    assert eigenvalue_diff <= 1e-6
    assert betweenness_diff <= 1e-9


def test_validator_bundle_runs_cleanly():
    checks = run_small_sanity_checks()
    assert checks["all_passed"] is True
