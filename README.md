# Async Web Crawler & Graph Analyzer

This project implements an asynchronous web crawler in Python using `asyncio` and `aiohttp` to explore a given domain, save HTML documents, and build a link graph. It also includes scripts to analyze the generated graph structure and PageRank scores.

## Requirements

- Python 3.13+
- Required libraries (install via `pip` or `uv`):

  ```bash
  # Using pip
  pip install aiohttp beautifulsoup4 lxml networkx aiofiles requests matplotlib numpy scipy

  # Or using uv
  uv add aiohttp beautifulsoup4 lxml networkx aiofiles requests matplotlib numpy scipy
  ```

## Quick Start

1.  **Run the Crawler:**

    ```bash
    # Crawl osu.edu, max 3000 pages, 20 concurrent tasks, save to ./artifacts
    uv run src/crawler.py https://www.osu.edu -m 3000 -c 20 -o ./artifacts
    ```

    This will generate `artifacts/link_graph.graphml` and save HTML files in `artifacts/html_docs/`.

2.  **Analyze the Graph:**

    ```bash
    # Analyze the generated graph, save results/plots to ./graph_results
    uv run src/graph_analysis.py artifacts/link_graph.graphml -o ./artifacts/graph_results
    ```

3.  **Analyze PageRank:**

    ```bash
    # Calculate and analyze PageRank, save results/plots to ./pagerank_results
    uv run src/pagerank_analysis.py artifacts/link_graph.graphml -o ./artifacts/pagerank_results
    ```

4.  **Test Crawler Performance:**

    ```bash
    # Make the script executable
    chmod +x src/test_performance.sh

    # Run performance tests (uses MAX_PAGES_TEST=1000 by default)
    ./src/test_performance.sh
    ```

    This saves timing results to `./artifacts/perf_test/performance_results.csv`.
