import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import os
from collections import Counter

# --- Helper Functions (copied/adapted from analyze_graph.py) ---


def plot_distribution(data, title, xlabel, ylabel, filename, bins=50, log_scale=False):
    """Helper function to create and save histograms."""
    if not data:
        print(f"Warning: No data provided for plot '{title}'. Skipping.")
        return
    data = [x for x in data if isinstance(x, (int, float)) and np.isfinite(x)]
    if not data:
        print(
            f"Warning: No valid numeric data left for plot '{title}' after filtering. Skipping."
        )
        return

    plt.figure(figsize=(10, 6))
    counts, bin_edges = np.histogram(data, bins=bins)
    plt.hist(data, bins=bin_edges, alpha=0.75)

    if log_scale:
        plt.xscale("log")
        plt.yscale("log")
        positive_data = [x for x in data if x > 0]
        min_val = (
            min(positive_data) if positive_data else 1e-9
        )  # Use small value if no positive data
        plt.xlim(left=min_val * 0.9)
        min_y, max_y = plt.ylim()
        if min_y <= 0:
            positive_counts = counts[counts > 0]
            min_y_data = min(positive_counts) if positive_counts.size > 0 else 1
            plt.ylim(bottom=min_y_data * 0.5)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    try:
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
    except Exception as e:
        print(f"Error saving plot {filename}: {e}")
    plt.close()


def fit_power_law(values):
    """Fits a power law to the distribution of values using log-log regression."""
    if not values:
        return None, None, None
    # Use values directly (e.g., PageRank scores), not degrees here
    # Filter for positive values for log
    positive_values = sorted(
        [v for v in values if isinstance(v, (int, float)) and v > 0]
    )
    if not positive_values:
        return None, None, None

    # Create histogram-like data for fitting: value vs frequency (or rank)
    # Using rank-frequency plot is common for PageRank power law analysis
    n = len(positive_values)
    ranks = np.arange(1, n + 1)  # Rank from 1 to N
    values_sorted_desc = sorted(positive_values, reverse=True)

    log_ranks = np.log10(ranks)
    log_values = np.log10(values_sorted_desc)

    if len(log_ranks) < 2:
        return None, None, None

    try:
        coeffs = np.polyfit(log_ranks, log_values, 1)  # Fit log(value) vs log(rank)
        # For rank-frequency, the exponent is often denoted differently, but derived from slope
        # If P(r) ~ r^-beta (rank vs value), then log(P) ~ -beta*log(r)
        # If V(r) ~ r^-alpha (rank vs value), then log(V) ~ -alpha*log(r)
        alpha_exponent = -coeffs[0]  # Slope is negative exponent
        slope, intercept = coeffs
        y_pred = slope * log_ranks + intercept
        residuals = log_values - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((log_values - np.mean(log_values)) ** 2)
        if ss_tot == 0:
            r_squared = 1.0 if ss_res == 0 else 0.0
        else:
            r_squared = 1 - (ss_res / ss_tot)

        # Note: 'gamma' usually refers to degree distribution exponent P(k) ~ k^-gamma
        # For PageRank value distribution P(x) ~ x^-gamma or Rank-Value R(v) ~ v^-beta
        # We are fitting log(value) vs log(rank), so slope relates to exponent of rank vs value.
        # Let's call it 'rank_exponent' for clarity.
        return alpha_exponent, coeffs, r_squared
    except Exception as e:
        print(f"Error during power-law fitting (rank-value): {e}")
        return None, None, None


# --- PageRank Implementation ---


def calculate_pagerank(G, alpha=0.85, max_iter=100, tol=1.0e-6):
    """
    Calculates PageRank iteratively.

    Args:
        G (nx.DiGraph): The input graph.
        alpha (float): Damping factor (probability of following links).
                       alpha=1.0 corresponds to no damping (basic PageRank).
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence (L1 norm of change).

    Returns:
        tuple: (dict: PageRank scores, int: iterations performed)
    """
    N = G.number_of_nodes()
    if N == 0:
        return {}, 0

    # Initialisation
    ranks = {node: 1.0 / N for node in G.nodes()}
    dangling_nodes = [node for node in G if G.out_degree(node) == 0]
    iterations = 0

    for i in range(max_iter):
        iterations += 1
        old_ranks = ranks.copy()
        new_ranks = {node: 0.0 for node in G.nodes()}  # Start fresh for new calculation

        # Calculate contribution from dangling nodes in the *previous* iteration
        dangling_sum = sum(old_ranks[node] for node in dangling_nodes)

        # Distribute rank based on links and dangling nodes
        for node in G.nodes():
            # Rank from incoming links
            in_rank_sum = sum(
                old_ranks[predecessor] / G.out_degree(predecessor)
                for predecessor in G.predecessors(node)
                if G.out_degree(predecessor) > 0
            )  # Avoid division by zero just in case

            # Combine base rank, link rank, and dangling rank
            new_ranks[node] = ((1.0 - alpha) / N) + alpha * (
                in_rank_sum + dangling_sum / N
            )

        # --- Normalization ---
        # Necessary especially for alpha=1 or graphs with sinks/disconnections
        # to ensure ranks sum to 1 and handle potential rank loss/gain issues.
        current_sum = sum(new_ranks.values())
        if (
            abs(current_sum - 1.0) > 1e-9
        ):  # Check if sum is significantly different from 1
            # print(f"Debug: Iter {i+1}, Sum before norm: {current_sum}") # Optional debug
            if current_sum <= 0:  # Avoid division by zero or negative ranks
                print(
                    f"Warning: Rank sum became zero or negative at iter {i + 1}. Resetting to uniform."
                )
                # Reset to uniform as a fallback, though shouldn't happen with correct formula
                ranks = {node: 1.0 / N for node in G.nodes()}
                continue  # Skip convergence check for this iteration
            factor = 1.0 / current_sum
            ranks = {node: rank * factor for node, rank in new_ranks.items()}
        else:
            ranks = new_ranks  # Assign directly if sum is close enough to 1

        # --- Check Convergence ---
        # L1 norm difference
        diff = sum(abs(ranks[node] - old_ranks[node]) for node in G.nodes())
        if diff < tol:
            # print(f"Converged after {iterations} iterations.")
            break
    else:  # Executed if loop finishes without break
        print(
            f"Warning: PageRank did not converge within {max_iter} iterations (diff={diff:.2e})."
        )

    return ranks, iterations


# --- Main Analysis Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate and analyze PageRank for a web graph."
    )
    parser.add_argument(
        "graphml_file", help="Path to the input GraphML file (e.g., link_graph.graphml)"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="./pagerank_analysis_results",
        help="Directory to save analysis results (plots)",
    )
    parser.add_argument(
        "-a",
        "--alphas",
        nargs="+",
        type=float,
        default=[0.5, 0.7, 0.85, 0.9, 0.95, 0.99, 1.0],
        help="List of alpha (damping factor) values to test for convergence.",
    )
    parser.add_argument(
        "--max_iter", type=int, default=100, help="Max iterations for PageRank."
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0e-6,
        help="Convergence tolerance for PageRank.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.graphml_file):
        print(f"Error: Input file not found: {args.graphml_file}")
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Analysis results will be saved in: {args.output_dir}")

    # Load graph
    print(f"Loading graph from {args.graphml_file}...")
    start_load_time = time.time()
    G = None
    try:
        G = nx.read_graphml(args.graphml_file)
        if not isinstance(G, nx.DiGraph):
            print(
                "Warning: Graph read from file is not directed (DiGraph). Converting..."
            )
            G = nx.DiGraph(G)
    except Exception as e:
        print(f"Error loading graph: {e}")
        exit(1)

    if G is None or G.number_of_nodes() == 0:
        print("Loaded graph is empty or failed to load properly. Exiting.")
        exit(1)
    print(
        f"Graph loaded successfully with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges in {time.time() - start_load_time:.2f} seconds."
    )

    # --- 1. Calculate PageRank (Damped and Undamped) ---
    print("\n--- 1. Calculating PageRank ---")
    default_alpha = 0.85
    if (
        default_alpha not in args.alphas
    ):  # Ensure the default is calculated if not in list
        args.alphas.append(default_alpha)
        args.alphas.sort()

    print(f"Calculating PageRank with default damping (alpha={default_alpha})...")
    start_time = time.time()
    pagerank_damped, iters_damped = calculate_pagerank(
        G, alpha=default_alpha, max_iter=args.max_iter, tol=args.tolerance
    )
    print(
        f"  Completed in {time.time() - start_time:.2f}s ({iters_damped} iterations)."
    )

    print(f"\nCalculating PageRank without damping (alpha=1.0)...")
    start_time = time.time()
    pagerank_undamped, iters_undamped = calculate_pagerank(
        G, alpha=1.0, max_iter=args.max_iter, tol=args.tolerance
    )
    print(
        f"  Completed in {time.time() - start_time:.2f}s ({iters_undamped} iterations)."
    )
    if iters_undamped >= args.max_iter:
        print(
            "  Note: Undamped PageRank might not converge properly on graphs that aren't strongly connected."
        )

    # Display top ranked pages (example)
    if pagerank_damped:
        top_n = 10
        sorted_pr_damped = sorted(
            pagerank_damped.items(), key=lambda item: item[1], reverse=True
        )
        print(f"\nTop {top_n} pages (Damped PageRank, alpha={default_alpha}):")
        for i, (node, rank) in enumerate(sorted_pr_damped[:top_n]):
            print(f"  {i + 1}. Rank={rank:.2e} : {node}")

    if pagerank_undamped:
        top_n = 10
        sorted_pr_undamped = sorted(
            pagerank_undamped.items(), key=lambda item: item[1], reverse=True
        )
        print(f"\nTop {top_n} pages (Undamped PageRank, alpha=1.0):")
        for i, (node, rank) in enumerate(sorted_pr_undamped[:top_n]):
            print(f"  {i + 1}. Rank={rank:.2e} : {node}")

    # --- 2. Analyze PageRank Distribution (Damped) ---
    print(
        f"\n--- 2. Analyzing Damped PageRank (alpha={default_alpha}) Distribution ---"
    )
    if pagerank_damped:
        pr_values = list(pagerank_damped.values())

        # Linear Histogram
        plot_distribution(
            pr_values,
            f"PageRank Distribution (alpha={default_alpha})",
            "PageRank Value",
            "Frequency",
            os.path.join(args.output_dir, f"pagerank_{default_alpha}_hist.png"),
        )

        # Log-Log Histogram
        plot_distribution(
            pr_values,
            f"PageRank Distribution (alpha={default_alpha}, Log-Log)",
            "PageRank Value",
            "Frequency",
            os.path.join(args.output_dir, f"pagerank_{default_alpha}_loglog.png"),
            log_scale=True,
        )

        # Fit Power Law (Rank-Value)
        rank_exponent, _, r_sq = fit_power_law(pr_values)
        if rank_exponent is not None:
            print(
                f"Estimated Power Law Exponent for PageRank (Rank vs Value): {rank_exponent:.4f} (R^2={r_sq:.4f})"
            )
            if r_sq < 0.8:
                print(
                    "  Note: R^2 value suggests the power law fit (rank-value) may not be very accurate."
                )
        else:
            print("Could not fit power law to PageRank distribution.")
    else:
        print(
            "Skipping PageRank distribution analysis as damped PageRank could not be calculated."
        )

    # --- 3. Analyze Convergence ---
    print("\n--- 3. Analyzing PageRank Convergence vs. Alpha ---")
    convergence_results = {}
    alphas_to_test = sorted(
        [a for a in args.alphas if 0.0 <= a <= 1.0]
    )  # Ensure valid alphas
    print(f"Testing convergence for alpha values: {alphas_to_test}")

    for alpha_val in alphas_to_test:
        print(f"  Running for alpha={alpha_val}...")
        start_time = time.time()
        _, iters = calculate_pagerank(
            G,
            alpha=alpha_val,
            max_iter=args.max_iter * 2,  # Allow more iterations for convergence test
            tol=args.tolerance,
        )
        convergence_results[alpha_val] = iters
        print(f"    Finished in {time.time() - start_time:.2f}s, Iterations: {iters}")

    # Plot results
    if convergence_results:
        alphas = list(convergence_results.keys())
        iterations = list(convergence_results.values())

        plt.figure(figsize=(10, 6))
        plt.plot(alphas, iterations, marker="o", linestyle="-")
        plt.xlabel("Damping Factor (alpha)")
        plt.ylabel("Iterations to Converge")
        plt.title("PageRank Convergence Speed vs. Damping Factor")
        plt.grid(True)
        plt.tight_layout()
        plot_filename = os.path.join(args.output_dir, "pagerank_convergence.png")
        try:
            plt.savefig(plot_filename)
            print(f"\nConvergence plot saved to {plot_filename}")
        except Exception as e:
            print(f"Error saving convergence plot {plot_filename}: {e}")
        plt.close()
    else:
        print("No convergence results to plot.")

    print("\n--- PageRank Analysis Complete ---")
