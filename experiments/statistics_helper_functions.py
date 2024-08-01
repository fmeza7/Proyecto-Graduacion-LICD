# IMPORTS
from positional_encodings.torch_encodings import *
import numpy as np
import random
import networkx as nx
from collections import defaultdict
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pyvis.network import Network
from IPython.display import display, HTML
import hashlib
# ----------------------------

# STATISTICS FUNCTIONS
# -------------------
def calculate_graph_metrics(adj_matrices: np.ndarray):
    """
    Calculate various graph metrics for a dataset of adjacency matrices.

    :param adj_matrices: NumPy array of shape (n_observations, num_nodes, num_nodes)
    :return: List of dictionaries containing metrics for each graph
    """
    metrics = []

    for adj_matrix in adj_matrices:
        G = nx.from_numpy_array(adj_matrix)

        # Basic metrics
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        avg_degree = 2 * num_edges / num_nodes
        density = nx.density(G)

        # Degree distribution
        degree_sequence = [d for n, d in G.degree()]
        degree_counts = np.bincount(degree_sequence)
        degree_distribution = degree_counts / num_nodes

        # Clustering coefficient
        clustering_coeff = nx.average_clustering(G)

        # Average path length and diameter
        if nx.is_connected(G):
            avg_path_length = nx.average_shortest_path_length(G)
            diameter = nx.diameter(G)
        else:
            avg_path_length = np.nan
            diameter = np.nan

        # Connected components
        connected_components = list(nx.connected_components(G))
        num_components = len(connected_components)
        largest_component_size = max(len(c) for c in connected_components)

        # Centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)

        metrics.append({
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "avg_degree": avg_degree,
            "density": density,
            "degree_distribution": degree_distribution.tolist(),
            "clustering_coefficient": clustering_coeff,
            "avg_path_length": avg_path_length,
            "diameter": diameter,
            "num_connected_components": num_components,
            "largest_component_size": largest_component_size,
            "avg_degree_centrality": np.mean(list(degree_centrality.values())),
            "avg_betweenness_centrality": np.mean(list(betweenness_centrality.values())),
            "avg_closeness_centrality": np.mean(list(closeness_centrality.values()))
        })

    return metrics

def summarize_dataset_metrics(metrics):
    """
    Calculate summary statistics for the entire dataset of graphs.

    :param metrics: List of dictionaries containing metrics for each graph
    :return: Dictionary of summary statistics for the dataset
    """
    summary = {}

    # Metrics to summarize (excluding degree_distribution)
    summarize_keys = [
        "num_nodes", "num_edges", "avg_degree", "density", "clustering_coefficient",
        "avg_path_length", "diameter", "num_connected_components", "largest_component_size",
        "avg_degree_centrality", "avg_betweenness_centrality", "avg_closeness_centrality"
    ]

    for key in summarize_keys:
        values = [m[key] for m in metrics if not np.isnan(m[key])]
        summary[f"{key}_mean"] = np.mean(values)
        summary[f"{key}_std"] = np.std(values)
        summary[f"{key}_min"] = np.min(values)
        summary[f"{key}_max"] = np.max(values)

    # Summarize degree distribution
    all_degree_distributions = [m["degree_distribution"] for m in metrics]
    max_degree = max(len(dist) for dist in all_degree_distributions)

    padded_distributions = [np.pad(dist, (0, max_degree - len(dist))) for dist in all_degree_distributions]
    avg_degree_distribution = np.mean(padded_distributions, axis=0)

    summary["avg_degree_distribution"] = avg_degree_distribution.tolist()

    return summary

def create_visualizations(metrics, summary):
    """
    Create and display various visualizations for the graph metrics dataset.

    :param metrics: List of dictionaries containing metrics for each graph
    :param summary: Dictionary of summary statistics for the dataset
    """
    # Convert metrics to a DataFrame for easier plotting
    df = pd.DataFrame(metrics)

    # 1. Histograms of key metrics
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flatten()

    for i, column in enumerate(['num_nodes', 'num_edges', 'avg_degree',
                                'clustering_coefficient', 'avg_path_length']):
        sns.histplot(df[column].dropna(), ax=axes[i], kde=True)
        axes[i].set_title(f'Distribution of {column}')

    axes[-1].remove()  # Remove the last empty subplot
    plt.tight_layout()
    plt.show()

    # 2. Box plots for all numeric metrics
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[numeric_columns])
    plt.xticks(rotation=90)
    plt.title('Box plots of numeric metrics')
    plt.tight_layout()
    plt.show()

    # 3. Scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    sns.scatterplot(data=df, x='num_nodes', y='num_edges', ax=axes[0])
    axes[0].set_title('Number of nodes vs. Number of edges')

    sns.scatterplot(data=df, x='avg_degree', y='clustering_coefficient', ax=axes[1])
    axes[1].set_title('Average degree vs. Clustering coefficient')

    sns.scatterplot(data=df, x='num_nodes', y='avg_path_length', ax=axes[2])
    axes[2].set_title('Number of nodes vs. Average path length')

    plt.tight_layout()
    plt.show()

    # 4. Heatmap of correlation matrix
    plt.figure(figsize=(8, 8))
    correlation_matrix = df[numeric_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Matrix of Numeric Metrics')
    plt.tight_layout()
    plt.show()

    # 5. Degree distribution summary
    degree_distributions = [m['degree_distribution'] for m in metrics]
    max_degree = max(len(dist) for dist in degree_distributions)

    # Calculate average degree distribution
    avg_distribution = np.zeros(max_degree)
    for dist in degree_distributions:
        avg_distribution[:len(dist)] += dist
    avg_distribution /= len(degree_distributions)

    # Plot average degree distribution
    plt.figure(figsize=(8, 4))
    plt.plot(range(max_degree), avg_distribution)
    plt.title('Average Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Average Proportion of Nodes')
    plt.tight_layout()
    plt.show()

    print("All visualizations have been displayed.")

# -------------------