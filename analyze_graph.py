import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import argparse
import os
from collections import Counter
from scipy import stats  # Import for fitting and correlation

NUM_SAMPLES_FOR_PATHS = 500
NODES_TO_REMOVE_PERCENT = 5


def plot_distribution(data, title, xlabel, ylabel, filename, bins=50, log_scale=False):
    if not data:
        print(f"Warning: No data provided for plot '{title}'. Skipping.")
        return
    # Filter out potential non-numeric or NaN/inf values if they somehow occur
    data = [x for x in data if isinstance(x, (int, float)) and np.isfinite(x)]
    if not data:
        print(
            f"Warning: No valid numeric data left for plot '{title}' after filtering. Skipping."
        )
        return

    plt.figure(figsize=(10, 6))
    # Use numpy histogram for potentially better bin calculation
    counts, bin_edges = np.histogram(data, bins=bins)
    plt.hist(data, bins=bin_edges, alpha=0.75)  # Use calculated bin edges

    if log_scale:
        plt.xscale("log")
        plt.yscale("log")
        # Filter positive data for xlim calculation
        positive_data = [x for x in data if x > 0]
        min_val = min(positive_data) if positive_data else 1
        plt.xlim(left=min_val * 0.9)
        min_y, max_y = plt.ylim()
        if min_y <= 0:
            # Use histogram counts to determine min y > 0
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


def fit_power_law(degrees):
    if not degrees:
        return None, None
    # Ensure degrees are numeric and positive for log
    degrees = [d for d in degrees if isinstance(d, (int, float)) and d > 0]
    if not degrees:
        return None, None

    counts = Counter(degrees)
    degree_values = sorted(counts.keys())
    counts_values = [counts[d] for d in degree_values]

    if len(degree_values) < 2:
        return None, None

    log_degrees = np.log10(degree_values)
    log_counts = np.log10(counts_values)

    try:
        # Use nan_policy='omit' if potential NaNs could arise, although filtering should prevent this
        coeffs = np.polyfit(log_degrees, log_counts, 1)
        gamma = -coeffs[0]
        # Add R^2 calculation for goodness of fit assessment
        slope, intercept = coeffs
        y_pred = slope * log_degrees + intercept
        residuals = log_counts - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((log_counts - np.mean(log_counts)) ** 2)
        if ss_tot == 0:  # Avoid division by zero if all counts are the same
            r_squared = 1.0 if ss_res == 0 else 0.0
        else:
            r_squared = 1 - (ss_res / ss_tot)

        return gamma, coeffs, r_squared
    except Exception as e:
        print(f"Error during power-law fitting: {e}")
        return None, None, None


def analyze_basic_properties(G):
    print("\n--- 1. Basic Properties ---")
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    print(f"Number of nodes (V): {num_nodes}")
    print(f"Number of edges (E): {num_edges}")
    return num_nodes, num_edges


def analyze_connectivity(G):
    print("\n--- 2. Connectivity Analysis ---")
    if not G or G.number_of_nodes() == 0:
        print("Graph is empty or has no nodes, cannot analyze connectivity.")
        return [], [], set(), set()  # Return empty structures

    wccs = list(nx.weakly_connected_components(G))
    num_wcc = len(wccs)
    largest_wcc = max(wccs, key=len) if wccs else set()
    size_largest_wcc = len(largest_wcc)
    print(f"Number of Weakly Connected Components (WCC): {num_wcc}")
    if num_wcc > 0:
        print(
            f"Size of the largest WCC: {size_largest_wcc} nodes ({size_largest_wcc / G.number_of_nodes():.2%})"
        )
        wcc_sizes = sorted([len(c) for c in wccs], reverse=True)
        print(f"WCC sizes (top 5): {wcc_sizes[:5]}")

    sccs = list(nx.strongly_connected_components(G))
    num_scc = len(sccs)
    largest_scc = max(sccs, key=len) if sccs else set()
    size_largest_scc = len(largest_scc)
    print(f"\nNumber of Strongly Connected Components (SCC): {num_scc}")
    if num_scc > 0:
        print(
            f"Size of the largest SCC: {size_largest_scc} nodes ({size_largest_scc / G.number_of_nodes():.2%})"
        )
        scc_sizes = sorted([len(c) for c in sccs], reverse=True)
        print(f"SCC sizes (top 5): {scc_sizes[:5]}")

    if size_largest_scc > 0:
        print(
            "\nNote: Detailed IN/OUT component analysis relative to the largest SCC requires extensive reachability checks and is simplified here."
        )
        try:
            scc_graph = nx.condensation(G, scc=sccs)
            print(
                f"Condensation graph (graph of SCCs) has {scc_graph.number_of_nodes()} nodes and {scc_graph.number_of_edges()} edges."
            )
        except Exception as e:
            print(f"Could not generate condensation graph: {e}")

    return wccs, sccs, largest_wcc, largest_scc


def analyze_degrees(G, output_dir):
    print("\n--- 3. Degree Distribution Analysis ---")
    if not G or G.number_of_nodes() == 0:
        print("Graph is empty or has no nodes, cannot analyze degrees.")
        return

    os.makedirs(output_dir, exist_ok=True)

    in_degrees = [d for n, d in G.in_degree()]
    avg_in_degree = np.mean(in_degrees) if in_degrees else 0
    print(f"Average In-Degree: {avg_in_degree:.4f}")
    plot_distribution(
        in_degrees,
        "In-Degree Distribution",
        "In-Degree",
        "Frequency",
        os.path.join(output_dir, "in_degree_hist.png"),
    )
    plot_distribution(
        in_degrees,
        "In-Degree Distribution (Log-Log)",
        "In-Degree",
        "Frequency",
        os.path.join(output_dir, "in_degree_loglog.png"),
        log_scale=True,
    )
    gamma_in, _, r_sq_in = fit_power_law(in_degrees)
    if gamma_in is not None:
        print(
            f"Estimated Power Law Exponent (Gamma) for In-Degree: {gamma_in:.4f} (R^2={r_sq_in:.4f})"
        )
        if r_sq_in < 0.8:  # Arbitrary threshold for poor fit
            print(
                "  Note: R^2 value suggests the power law fit may not be very accurate."
            )

    out_degrees = [d for n, d in G.out_degree()]
    avg_out_degree = np.mean(out_degrees) if out_degrees else 0
    print(f"\nAverage Out-Degree: {avg_out_degree:.4f}")
    plot_distribution(
        out_degrees,
        "Out-Degree Distribution",
        "Out-Degree",
        "Frequency",
        os.path.join(output_dir, "out_degree_hist.png"),
    )
    plot_distribution(
        out_degrees,
        "Out-Degree Distribution (Log-Log)",
        "Out-Degree",
        "Frequency",
        os.path.join(output_dir, "out_degree_loglog.png"),
        log_scale=True,
    )
    gamma_out, _, r_sq_out = fit_power_law(out_degrees)
    if gamma_out is not None:
        print(
            f"Estimated Power Law Exponent (Gamma) for Out-Degree: {gamma_out:.4f} (R^2={r_sq_out:.4f})"
        )
        if r_sq_out < 0.8:  # Arbitrary threshold for poor fit
            print(
                "  Note: R^2 value suggests the power law fit may not be very accurate."
            )


def analyze_distances(G, largest_wcc, output_dir):
    print("\n--- 4. Shortest Path Analysis (on Largest WCC) ---")
    if not largest_wcc:
        print("Largest WCC is empty, cannot analyze distances.")
        return

    valid_nodes = set(G.nodes())
    largest_wcc_nodes = set(largest_wcc).intersection(valid_nodes)
    if not largest_wcc_nodes:
        print(
            "Largest WCC node set is empty or invalid after intersection with graph nodes. Skipping distance analysis."
        )
        return

    largest_wcc_subgraph = G.subgraph(largest_wcc_nodes).copy()
    num_nodes_wcc = largest_wcc_subgraph.number_of_nodes()

    if num_nodes_wcc == 0:
        print("Largest WCC subgraph has 0 nodes. Skipping distance analysis.")
        return

    if not nx.is_weakly_connected(largest_wcc_subgraph):
        print(
            "Warning: Extracted subgraph is not weakly connected (this might indicate an issue). Skipping distance analysis."
        )
        return

    avg_shortest_path_val = None
    all_shortest_paths = []

    try:
        is_strongly_conn = nx.is_strongly_connected(largest_wcc_subgraph)
        if is_strongly_conn:
            # Calculations for strongly connected WCC
            start_time = time.time()
            avg_shortest_path_val = nx.average_shortest_path_length(
                largest_wcc_subgraph
            )
            print(
                f"Average Shortest Path Length (Strongly Connected WCC): {avg_shortest_path_val:.4f} (calculated in {time.time() - start_time:.2f}s)"
            )
            # Calculate all paths for histogram if feasible
            if num_nodes_wcc <= NUM_SAMPLES_FOR_PATHS * 2:
                print(
                    f"Calculating all shortest path lengths for histogram (nodes <= {NUM_SAMPLES_FOR_PATHS * 2})..."
                )
                calc_start_time = time.time()
                path_lengths_dict = dict(
                    nx.all_pairs_shortest_path_length(largest_wcc_subgraph)
                )
                for source, targets in path_lengths_dict.items():
                    for target, length in targets.items():
                        if source != target:
                            all_shortest_paths.append(length)
                print(
                    f"  (Calculated {len(all_shortest_paths)} path lengths in {time.time() - calc_start_time:.2f}s)"
                )
            else:
                print(
                    "WCC is large, using sampling for path length histogram even though strongly connected."
                )
                # Fall through to sampling code below
        else:
            print(
                "Largest WCC is not strongly connected. Calculating average shortest path length considering only reachable pairs via sampling."
            )
            # Fall through to sampling code below

        # Sampling logic (used if not strongly connected, or if strongly connected but large and full calc not done)
        if not all_shortest_paths:
            total_path_length = 0
            reachable_pairs = 0
            start_time = time.time()
            sample_size = min(NUM_SAMPLES_FOR_PATHS, num_nodes_wcc)
            if sample_size > 0:
                sampled_nodes = random.sample(
                    list(largest_wcc_subgraph.nodes()), sample_size
                )
                print(f"Sampling shortest paths from {len(sampled_nodes)} nodes...")
                for i, source in enumerate(sampled_nodes):
                    if i > 0 and i % 100 == 0:
                        print(
                            f"  Processed paths from {i}/{len(sampled_nodes)} sampled nodes..."
                        )
                    try:
                        lengths = nx.shortest_path_length(
                            largest_wcc_subgraph, source=source
                        )
                        for target, length in lengths.items():
                            if source != target:
                                total_path_length += length
                                reachable_pairs += 1
                                all_shortest_paths.append(length)
                    except nx.NetworkXNoPath:
                        continue
                    except Exception as path_err:
                        print(
                            f"  Error calculating paths from node {source}: {path_err}"
                        )

                avg_shortest_path_val = (
                    total_path_length / reachable_pairs if reachable_pairs > 0 else 0
                )
                print(
                    f"Estimated Average Shortest Path Length (Sampled, Weakly Connected): {avg_shortest_path_val:.4f}"
                )
                print(
                    f"  (Calculated over {reachable_pairs} reachable pairs from sampled nodes in {time.time() - start_time:.2f}s)"
                )
            else:
                print("  No nodes to sample from.")

        # Plot histogram if paths were found
        if all_shortest_paths:
            plot_distribution(
                all_shortest_paths,
                "Shortest Path Length Distribution (Sampled on Largest WCC)",
                "Path Length",
                "Frequency",
                os.path.join(output_dir, "shortest_path_hist.png"),
                bins=max(
                    int(max(all_shortest_paths)) + 1, 10
                ),  # Ensure bins cover range
            )
            # *** NEW: Analyze path length distribution ***
            try:
                mu, std = stats.norm.fit(all_shortest_paths)
                print(f"\nAnalysis of Shortest Path Length Distribution (Example):")
                print(
                    f"  Fitted Normal Distribution Parameters: Mean={mu:.4f}, StdDev={std:.4f}"
                )
                # Test for normality (optional but good practice)
                k2, p = stats.normaltest(all_shortest_paths)
                print(
                    f"  Normality Test (D'Agostino-Pearson): Statistic={k2:.4f}, p-value={p:.4g}"
                )
                if p < 0.05:  # Significance level
                    print(
                        "  Note: Distribution significantly differs from normal based on the test."
                    )
            except Exception as fit_err:
                print(
                    f"  Could not fit/analyze normal distribution for path lengths: {fit_err}"
                )
        else:
            print(
                "No shortest path lengths were calculated or sampled for the histogram."
            )

    except nx.NetworkXError as e:
        print(f"Could not calculate average shortest path length: {e}")
    except Exception as e:
        print(
            f"An unexpected error occurred during average shortest path calculation: {e}"
        )

    # *** NEW: Eccentricity Analysis ***
    print(
        "\nCalculating Eccentricities (can be very slow, requires all-pairs shortest paths)..."
    )
    if nx.is_strongly_connected(largest_wcc_subgraph):
        try:
            start_time = time.time()
            # Use infinity=None to handle potential disconnectedness within SCC if graph is malformed
            eccentricities = nx.eccentricity(largest_wcc_subgraph)
            ecc_values = [
                e for e in eccentricities.values() if e is not None
            ]  # Filter out potential None/inf
            print(f"  (Calculated eccentricities in {time.time() - start_time:.2f}s)")

            if ecc_values:
                radius = min(ecc_values)
                diameter = max(ecc_values)
                avg_ecc = np.mean(ecc_values)
                print(f"Graph Radius (min eccentricity): {radius}")
                print(f"Graph Diameter (max eccentricity): {diameter}")
                print(f"Average Eccentricity: {avg_ecc:.4f}")
                # Use integer bins if eccentricities are integers
                bin_count = (
                    max(1, int(diameter) - int(radius) + 1)
                    if isinstance(radius, int) and isinstance(diameter, int)
                    else 50
                )
                plot_distribution(
                    ecc_values,
                    "Eccentricity Distribution (Strongly Connected WCC)",
                    "Eccentricity",
                    "Frequency",
                    os.path.join(output_dir, "eccentricity_hist.png"),
                    bins=bin_count,
                )
            else:
                print("Eccentricity calculation returned no valid values.")

        except (nx.NetworkXError, nx.NetworkXUnfeasible) as e:  # Catch specific errors
            print(
                f"Could not calculate eccentricities (graph might have issues despite strong check): {e}"
            )
        except Exception as e:
            print(f"An unexpected error occurred during eccentricity calculation: {e}")
    else:
        print(
            "Largest WCC is not strongly connected, eccentricity is ill-defined for directed paths. Calculation skipped."
        )


def analyze_clustering(G, output_dir):
    print("\n--- 5. Clustering Coefficient Analysis ---")
    if not G or G.number_of_nodes() == 0:
        print("Graph is empty or has no nodes, cannot analyze clustering.")
        return

    os.makedirs(output_dir, exist_ok=True)

    global_clustering = None
    try:
        # Handle potential division by zero in transitivity for small/sparse graphs
        if G.number_of_edges() > 0:
            global_clustering = nx.transitivity(G)
            print(
                f"Global Clustering Coefficient (Transitivity): {global_clustering:.4f}"
            )
        else:
            print("Global Clustering Coefficient (Transitivity): N/A (no edges)")
    except Exception as e:
        print(f"Could not calculate global clustering coefficient: {e}")

    print("Calculating average local clustering coefficient (can be slow)...")
    avg_local_clustering = None
    try:
        start_time = time.time()
        avg_local_clustering = nx.average_clustering(G)
        print(
            f"Average Local Clustering Coefficient: {avg_local_clustering:.4f} (calculated in {time.time() - start_time:.2f}s)"
        )
    except Exception as e:
        print(f"Could not calculate average local clustering coefficient: {e}")

    print("Calculating local clustering coefficients for histogram (can be slow)...")
    local_clustering_coeffs = []
    nodes_list_for_clustering = []  # Store node order if calculation succeeds
    try:
        start_time = time.time()
        clustering_dict = nx.clustering(G)
        nodes_list_for_clustering = list(clustering_dict.keys())
        local_clustering_coeffs = list(clustering_dict.values())
        print(f"  (Calculated local coefficients in {time.time() - start_time:.2f}s)")
        if local_clustering_coeffs:
            plot_distribution(
                local_clustering_coeffs,
                "Local Clustering Coefficient Distribution",
                "Local Clustering Coefficient",
                "Frequency",
                os.path.join(output_dir, "local_clustering_hist.png"),
                bins=50,
            )
            # *** NEW: Analyze clustering coefficient distribution ***
            print(f"\nAnalysis of Local Clustering Coefficient Distribution (Example):")
            # Simple correlation with node degree
            try:
                if nodes_list_for_clustering and len(nodes_list_for_clustering) == len(
                    local_clustering_coeffs
                ):
                    degrees = [G.degree(n) for n in nodes_list_for_clustering]
                    coeffs_array = np.array(local_clustering_coeffs)
                    degrees_array = np.array(degrees)

                    valid_indices = degrees_array >= 2
                    if np.sum(valid_indices) > 1:
                        corr_coeffs = coeffs_array[valid_indices]
                        corr_degrees = degrees_array[valid_indices]
                        correlation, p_value = stats.pearsonr(corr_degrees, corr_coeffs)
                        print(
                            f"  Correlation between Node Degree (>=2) and Local CC: {correlation:.4f} (p-value: {p_value:.4g})"
                        )
                    else:
                        print(
                            "  Not enough nodes with degree >= 2 for correlation analysis."
                        )
                else:
                    print(
                        "  Cannot perform degree-clustering correlation due to potential node/coefficient mismatch."
                    )
            except Exception as corr_err:
                print(
                    f"  Could not perform correlation analysis for clustering coefficients: {corr_err}"
                )

        else:
            print("No local clustering coefficients were calculated.")
    except Exception as e:
        print(f"Could not calculate local clustering coefficients: {e}")


def analyze_robustness(G_orig, largest_wcc_orig, output_dir):
    print("\n--- 6. Robustness Analysis ---")
    if not G_orig or G_orig.number_of_nodes() == 0:
        print("Original graph is empty or has no nodes, cannot analyze robustness.")
        return

    num_nodes_orig = G_orig.number_of_nodes()
    nodes_to_remove = int(num_nodes_orig * (NODES_TO_REMOVE_PERCENT / 100.0))
    if nodes_to_remove == 0:
        print(
            f"Warning: Calculated 0 nodes to remove ({NODES_TO_REMOVE_PERCENT}% of {num_nodes_orig}). Skipping robustness analysis."
        )
        return
    print(
        f"Simulating removal of {nodes_to_remove} nodes ({NODES_TO_REMOVE_PERCENT}%)."
    )

    # --- Random Failures ---
    print("\n--- 6a. Random Failures ---")
    G_failed = G_orig.copy()
    nodes_to_remove_random = []
    available_nodes = list(G_failed.nodes())
    # Ensure we don't try to sample more nodes than exist
    actual_remove_count_random = min(nodes_to_remove, len(available_nodes))
    if actual_remove_count_random > 0:
        nodes_to_remove_random = random.sample(
            available_nodes, actual_remove_count_random
        )
        G_failed.remove_nodes_from(nodes_to_remove_random)
        print(f"Removed {len(nodes_to_remove_random)} random nodes.")
    else:
        print(
            f"Warning: Not enough nodes to remove for random failures or 0 nodes requested. Skipping."
        )
        G_failed = None

    if G_failed:
        print("Analyzing graph after random failures:")
        analyze_basic_properties(G_failed)
        # Recalculate connectivity for the modified graph
        wccs_f, sccs_f, largest_wcc_f, largest_scc_f = analyze_connectivity(G_failed)
        analyze_degrees(G_failed, os.path.join(output_dir, "robustness_failure"))
        # Use the newly calculated largest WCC for distance analysis
        analyze_distances(
            G_failed, largest_wcc_f, os.path.join(output_dir, "robustness_failure")
        )
    else:
        print("Skipping analysis after random failures due to previous warning.")

    # --- Targeted Attacks ---
    print("\n--- 6b. Targeted Attacks ---")
    G_attacked = G_orig.copy()
    nodes_to_remove_attack = []
    if G_attacked.number_of_nodes() > 0:
        node_degrees = list(G_attacked.degree())
        if node_degrees:
            nodes_sorted_by_degree = sorted(
                node_degrees, key=lambda item: item[1], reverse=True
            )
            actual_remove_count_attack = min(
                nodes_to_remove, len(nodes_sorted_by_degree)
            )
            nodes_to_remove_attack = [
                node
                for node, degree in nodes_sorted_by_degree[:actual_remove_count_attack]
            ]
            if nodes_to_remove_attack:  # Check if list is not empty
                G_attacked.remove_nodes_from(nodes_to_remove_attack)
                print(f"Removed {len(nodes_to_remove_attack)} highest-degree nodes.")
            else:
                print(
                    "Warning: No nodes selected for targeted attack removal (list was empty). Skipping."
                )
                G_attacked = None
        else:
            print(
                "Warning: Could not retrieve degrees for targeted attack sorting. Skipping."
            )
            G_attacked = None
    else:
        print(
            "Warning: Original graph copy is empty for targeted attack analysis. Skipping."
        )
        G_attacked = None

    if G_attacked:
        print("Analyzing graph after targeted attacks:")
        analyze_basic_properties(G_attacked)
        # Recalculate connectivity for the modified graph
        wccs_a, sccs_a, largest_wcc_a, largest_scc_a = analyze_connectivity(G_attacked)
        analyze_degrees(G_attacked, os.path.join(output_dir, "robustness_attack"))
        # Use the newly calculated largest WCC for distance analysis
        analyze_distances(
            G_attacked, largest_wcc_a, os.path.join(output_dir, "robustness_attack")
        )
    else:
        print("Skipping analysis after targeted attacks due to previous warning.")


def analyze_vertex_connectivity(G, largest_wcc, output_dir):
    print("\n--- 7. Vertex Connectivity Analysis (on Largest WCC) ---")
    if not largest_wcc:
        print("Largest WCC is empty, cannot analyze vertex connectivity.")
        return

    valid_nodes = set(G.nodes())
    largest_wcc_nodes = set(largest_wcc).intersection(valid_nodes)
    if not largest_wcc_nodes:
        print(
            "Largest WCC node set is empty or invalid after intersection. Skipping vertex connectivity."
        )
        return

    G_wcc_subgraph = G.subgraph(largest_wcc_nodes).copy()
    num_nodes_wcc = G_wcc_subgraph.number_of_nodes()

    if num_nodes_wcc == 0:
        print("Largest WCC subgraph has 0 nodes. Skipping vertex connectivity.")
        return

    print(
        "Node connectivity calculation is computationally expensive and skipped by default."
    )

    print(
        "Finding articulation points (cut vertices) on the largest WCC (treating as undirected)..."
    )
    try:
        start_time = time.time()
        # Convert to undirected, handling potential isolates if WCC wasn't perfectly derived
        G_wcc_undirected = nx.Graph()
        G_wcc_undirected.add_nodes_from(G_wcc_subgraph.nodes())
        G_wcc_undirected.add_edges_from(G_wcc_subgraph.edges())
        # Remove isolates as articulation points are defined for connected parts > 1 node
        G_wcc_undirected.remove_nodes_from(list(nx.isolates(G_wcc_undirected)))

        if G_wcc_undirected.number_of_nodes() > 1 and nx.is_connected(G_wcc_undirected):
            articulation_points = list(nx.articulation_points(G_wcc_undirected))
            print(
                f"Number of Articulation Points in Largest WCC (non-isolated part): {len(articulation_points)}"
            )
            if articulation_points:
                sample_size = min(10, len(articulation_points))
                print(
                    f"  Articulation Points (sample): {articulation_points[:sample_size]}"
                )
            print(
                f"  (Calculated articulation points in {time.time() - start_time:.2f}s)"
            )

            if len(articulation_points) == 0 and G_wcc_undirected.number_of_nodes() > 2:
                print(
                    "The main connected component of the largest WCC (undirected) is 2-connected (no articulation points)."
                )
                print(
                    "Finding separating pairs (k=2) requires more advanced algorithms (e.g., block decomposition) and is not implemented here."
                )
            elif len(articulation_points) > 0:
                print(
                    "The main connected component of the largest WCC (undirected) is 1-connected."
                )
        elif G_wcc_undirected.number_of_nodes() <= 1:
            print(
                "Warning: Undirected WCC subgraph has 0 or 1 non-isolated nodes. Articulation analysis not applicable."
            )
        else:  # Not connected
            print(
                "Warning: The largest WCC treated as undirected is not connected after removing isolates. Articulation points analyzed on components if needed (not implemented here)."
            )

    except Exception as e:
        print(f"Could not find articulation points: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze a web graph stored in GraphML format."
    )
    parser.add_argument(
        "graphml_file", help="Path to the input GraphML file (e.g., link_graph.graphml)"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="./graph_analysis_results",
        help="Directory to save analysis results (plots)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.graphml_file):
        print(f"Error: Input file not found: {args.graphml_file}")
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Analysis results will be saved in: {args.output_dir}")

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

    print(f"Graph loaded successfully in {time.time() - start_load_time:.2f} seconds.")

    num_nodes, num_edges = analyze_basic_properties(G)

    wccs, sccs, largest_wcc, largest_scc = analyze_connectivity(G)

    analyze_degrees(G, args.output_dir)

    # Pass the originally captured largest_wcc set
    analyze_distances(G, largest_wcc, args.output_dir)

    analyze_clustering(G, args.output_dir)

    # Pass copies to robustness analysis, handle empty largest_wcc
    analyze_robustness(
        G.copy(), largest_wcc.copy() if largest_wcc else set(), args.output_dir
    )

    # Pass the original largest_wcc set again
    analyze_vertex_connectivity(G, largest_wcc, args.output_dir)

    print("\n--- Analysis Complete ---")
