"""Streamlit dashboard for centrality computation and comparison."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
import sys
from typing import Dict

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.viz_utils import (
    build_graph_figure,
    compute_metrics,
    generate_synthetic_graph,
    metric_scores_to_csv,
    metrics_to_table_rows,
    parse_edge_list_text,
    summary_to_dict,
)
from src.compare_rankings import build_comparison_summary, top_k_nodes


st.set_page_config(page_title="Graph Centrality Dashboard", layout="wide")
st.title("Graph Centrality Dashboard")
st.caption("Interactive exploration of closeness, betweenness, and eigenvalue centrality.")


GRAPH_FAMILIES = [
    "path",
    "cycle",
    "star",
    "grid",
    "erdos_renyi",
    "barabasi_albert",
    "bridge_community",
]
METRIC_OPTIONS = ["closeness", "betweenness", "eigenvalue"]


@st.cache_data(show_spinner=False)
def _compute_cached(metric_names: tuple[str, ...], graph_payload: dict) -> Dict[str, Dict[object, float]]:
    graph = generate_synthetic_graph(graph_payload["family"], graph_payload["params"])
    return compute_metrics(graph, metric_names)


def _synthetic_controls() -> tuple[dict, Dict[str, object]]:
    family = st.sidebar.selectbox("Graph family", GRAPH_FAMILIES)
    params: Dict[str, object] = {}

    if family in {"path", "cycle", "star"}:
        params["n"] = st.sidebar.slider("n", min_value=5, max_value=250, value=50, step=1)
    elif family == "grid":
        params["rows"] = st.sidebar.slider("rows", min_value=2, max_value=20, value=8, step=1)
        params["cols"] = st.sidebar.slider("cols", min_value=2, max_value=20, value=8, step=1)
    elif family == "erdos_renyi":
        params["n"] = st.sidebar.slider("n", min_value=10, max_value=250, value=80, step=1)
        params["p"] = st.sidebar.slider("p", min_value=0.01, max_value=0.6, value=0.08, step=0.01)
        params["seed"] = st.sidebar.number_input("seed", min_value=0, max_value=10_000, value=42)
    elif family == "barabasi_albert":
        params["n"] = st.sidebar.slider("n", min_value=10, max_value=250, value=80, step=1)
        params["m"] = st.sidebar.slider("m", min_value=1, max_value=10, value=2, step=1)
        params["seed"] = st.sidebar.number_input("seed", min_value=0, max_value=10_000, value=42)
    else:
        params["size_left"] = st.sidebar.slider("size_left", min_value=4, max_value=100, value=20, step=1)
        params["size_right"] = st.sidebar.slider("size_right", min_value=4, max_value=100, value=20, step=1)
        params["p_left"] = st.sidebar.slider("p_left", min_value=0.01, max_value=1.0, value=0.25, step=0.01)
        params["p_right"] = st.sidebar.slider("p_right", min_value=0.01, max_value=1.0, value=0.25, step=0.01)
        params["bridge_mode"] = st.sidebar.selectbox("bridge_mode", ["single", "double"])
        params["seed"] = st.sidebar.number_input("seed", min_value=0, max_value=10_000, value=42)

    return {"family": family, "params": params}, params


def _render_summary(graph_summary: dict[str, object]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", int(graph_summary["num_nodes"]))
    c2.metric("Edges", int(graph_summary["num_edges"]))
    c3.metric("Avg Degree", f"{graph_summary['average_degree']:.3f}")
    c4.metric("Connected", "Yes" if graph_summary["connected"] else "No")

    st.markdown(
        f"- Min degree: `{graph_summary['min_degree']}`\n"
        f"- Max degree: `{graph_summary['max_degree']}`"
    )


def main() -> None:
    st.sidebar.header("Graph Input")
    source = st.sidebar.radio("Source", ["Synthetic graph", "Edge-list upload"])

    graph = None
    graph_payload = None

    if source == "Synthetic graph":
        graph_payload, _ = _synthetic_controls()
        graph = generate_synthetic_graph(graph_payload["family"], graph_payload["params"])
    else:
        uploaded = st.sidebar.file_uploader("Upload edge-list file", type=["txt", "edgelist", "csv"])
        delimiter = st.sidebar.text_input("Delimiter (optional)", value="")
        if uploaded is not None:
            content = StringIO(uploaded.getvalue().decode("utf-8", errors="replace")).read()
            try:
                graph = parse_edge_list_text(content, delimiter=delimiter or None)
            except Exception as exc:
                st.error(f"Failed to parse edge-list file: {exc}")

    if graph is None:
        st.info("Provide a graph using synthetic controls or edge-list upload.")
        return

    st.subheader("Graph Summary")
    graph_summary = summary_to_dict(graph)
    _render_summary(graph_summary)

    st.subheader("Centrality Computation")
    selected_metrics = st.multiselect(
        "Metrics",
        METRIC_OPTIONS,
        default=METRIC_OPTIONS,
    )

    if not selected_metrics:
        st.warning("Select at least one metric to continue.")
        return

    with st.spinner("Computing centrality scores..."):
        if source == "Synthetic graph" and graph_payload is not None:
            metric_scores = _compute_cached(tuple(sorted(selected_metrics)), graph_payload)
        else:
            metric_scores = compute_metrics(graph, selected_metrics)

    st.success("Centrality computation completed.")

    st.subheader("Graph Visualization")
    display_metric = st.selectbox("Metric for visualization", sorted(metric_scores.keys()))
    top_k = st.slider("Highlight top-k nodes", min_value=3, max_value=30, value=10, step=1)

    figure = build_graph_figure(
        graph,
        metric_scores[display_metric],
        metric_name=display_metric,
        top_k=top_k,
    )
    st.pyplot(figure, width="stretch")

    st.subheader("Node Scores")
    rows = metrics_to_table_rows(metric_scores)
    try:
        st.dataframe(rows, width="stretch")
    except Exception:
        st.warning("Interactive table rendering is unavailable in this environment. Showing CSV preview instead.")
        preview_csv = metric_scores_to_csv(metric_scores)
        st.code("\n".join(preview_csv.splitlines()[:30]))

    st.subheader("Top-k Rankings")
    topk_cols = st.columns(len(metric_scores))
    for index, metric in enumerate(sorted(metric_scores.keys())):
        top_nodes = top_k_nodes(metric_scores[metric], top_k)
        topk_cols[index].write(f"**{metric.title()}**")
        topk_cols[index].markdown("\n".join(f"- {repr(node)}" for node in top_nodes))

    if len(metric_scores) >= 2:
        st.subheader("Comparison Panel")
        comparison = build_comparison_summary(metric_scores, k_values=(top_k,))
        for pair_item in comparison["pairwise"]:
            top_item = pair_item["top_k"][0]
            st.markdown(
                f"**{pair_item['metric_a']} vs {pair_item['metric_b']}**  \n"
                f"Spearman: `{float(pair_item['spearman_rank_correlation']):.6f}`  \n"
                f"Top-{top_k} overlap count: `{int(top_item['overlap_count'])}`  \n"
                f"Top-{top_k} overlap fraction: `{float(top_item['overlap_fraction']):.6f}`  \n"
                f"Top-{top_k} Jaccard: `{float(top_item['jaccard_similarity']):.6f}`"
            )

    st.subheader("Export")
    csv_content = metric_scores_to_csv(metric_scores)
    st.download_button(
        label="Download metric scores as CSV",
        data=csv_content,
        file_name="centrality_scores.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
