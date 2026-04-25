# Efficient Implementation and Comparative Analysis of Graph Centrality Measures

A modular Python project for implementing, validating, benchmarking, comparing, and visualizing graph centrality measures on unweighted undirected graphs.

## Implemented Features

- Graph infrastructure with adjacency-list representation and label-to-ID mapping
- Robust edge-list loading and validation
- Synthetic graph generators:
  - path
  - cycle
  - star
  - grid
  - Erdos-Renyi
  - Barabasi-Albert
  - bridge-community
- Centrality algorithms:
  - closeness centrality (exact BFS-based)
  - betweenness centrality (exact Brandes algorithm)
  - eigenvalue centrality (power iteration with convergence metadata)
- Validation layer and unit tests for structural and ranking behavior
- Synthetic benchmark suite with CSV/JSON/log export
- Ranking comparison module:
  - top-k extraction
  - top-k overlap
  - Jaccard similarity
  - Spearman rank correlation
- Plot generation for runtime, score distribution, and ranking comparison
- Streamlit dashboard for interactive exploration and edge-list upload
- LaTeX architecture note and full LaTeX final report template/content

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── run_project.py
├── run_experiments.py
├── run_analysis.py
├── data/
│   ├── raw/
│   └── processed/
├── dashboard/
│   ├── app.py
│   └── viz_utils.py
├── docs/
│   └── architecture.tex
├── outputs/
│   ├── logs/
│   ├── plots/
│   ├── samples/
│   ├── benchmark_results.csv
│   ├── benchmark_results.json
│   ├── comparison_results.csv
│   └── comparison_results.json
├── report/
│   ├── final_report.tex
│   └── figures/
├── src/
│   ├── __init__.py
│   ├── graph_utils.py
│   ├── closeness.py
│   ├── betweenness.py
│   ├── eigenvalue.py
│   ├── validators.py
│   ├── experiments.py
│   ├── compare_rankings.py
│   ├── visualize.py
│   └── demo.py
└── tests/
    ├── test_small_graphs.py
    ├── test_experiments.py
    ├── test_compare_rankings.py
    └── test_visualize.py
```

## Setup

### 1) Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

## Run Commands

### Main demo

```bash
python3 run_project.py
```

Generates centrality outputs on representative synthetic graphs and writes a demo log to `outputs/logs/`.

### Tests

```bash
python3 -m pytest -q
```

### Benchmark suite

```bash
python3 run_experiments.py
```

Outputs:
- `outputs/benchmark_results.csv`
- `outputs/benchmark_results.json`
- `outputs/logs/benchmark_run_<timestamp>.log`

### Comparative analysis + plots

```bash
python3 run_analysis.py
```

Outputs:
- `outputs/comparison_results.csv`
- `outputs/comparison_results.json`
- `outputs/plots/runtime_vs_graph_size.png`
- `outputs/plots/runtime_by_family.png`
- `outputs/plots/centrality_histograms.png`
- `outputs/plots/topk_overlap_bars.png`
- `outputs/plots/rank_correlation_heatmap.png`
- `outputs/plots/graph_highlight_betweenness.png`
- `outputs/logs/analysis_run_<timestamp>.log`
- Copied figures under `report/figures/`

### Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard capabilities:
- synthetic graph selection or edge-list upload
- graph summary statistics
- metric computation (single or all)
- graph visualization by selected metric
- top-k ranking comparison, overlap, and correlation display
- CSV export of computed scores

## LaTeX Compilation

### Architecture document

```bash
cd docs
pdflatex architecture.tex
```

### Final report

```bash
cd report
pdflatex final_report.tex
```

If `pdflatex` is missing, install a TeX distribution (e.g., TeX Live, MacTeX, MiKTeX).

## Input and Output Formats

### Input
- Edge-list text file (`.txt`, `.edgelist`, `.csv`) with one edge per line:
  - whitespace-separated: `u v`
  - comma-separated: `u,v`

### Output
- Centrality scores: Python dictionaries (node -> score), CSV export in dashboard
- Benchmark output: row-oriented CSV/JSON with graph metadata, metric, runtime, status
- Comparison output: CSV/JSON with top-k overlap, Jaccard, and Spearman summaries
- Plots: PNG files under `outputs/plots/`

## Limitations

- Primary support is limited to unweighted undirected graphs.
- Benchmarking is synthetic-only in the current repository.
- Dashboard is designed for small and medium graph sizes and is not optimized for very large interactive workloads.
- Comparative analytics focuses on overlap and correlation summaries; advanced statistical significance analysis is not included.

## Future Work

- Add weighted and directed graph variants
- Extend comparative analytics with richer statistical and stability analyses
- Integrate real-world graph datasets and ingestion pipelines
- Expand dashboard interactions for larger-scale graph exploration
# Graph-Centrality-Measures-Dashboard
