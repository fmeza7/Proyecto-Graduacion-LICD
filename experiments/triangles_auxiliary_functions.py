# IMPORTS
import numpy as np
import random
import networkx as nx
import torch

import matplotlib.pyplot as plt

from pyvis.network import Network
from IPython.display import display, HTML
# ----------------------------

#  Auxiliary Functions from Dataset 1 -> Meeting 25-07-24

# AUXILIARY FUNCTIONS
# -------------------
def has_triangle(matrix: np.ndarray) -> bool:
    """
    Check if a graph has a triangle.

    Parameters:
    np.ndarray

    Return:
    Boolean 
    """
    n = matrix.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if matrix[i,j]:  # if there's an edge between i and j
                for k in range(j+1, n):
                    if matrix[i,k] and matrix[j,k]:
                        return True
    return False

def generate_random_graphs(num_nodes: int, num_graphs: int=1, enforce_triangle: bool=None) -> list:
    """
    Generate random graphs with a specified number of nodes.

    Parameters:
    num_nodes (int): Number of nodes in the graph.
    num_graphs (int): Number of graphs to generate.
    enforce_triangle (bool): Whether to enforce the presence of a triangle in the graph.

    Return:
    list: List of adjacency matrices representing the generated graphs.
    """
    def generate_single_graph():
        # Initialize adjacency matrix
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

        if enforce_triangle is False:
            # Generate a triangle-free graph
            for i in range(num_nodes):
                for j in range(i+1, num_nodes):
                    # Check if adding this edge would create a triangle
                    can_add_edge = True
                    for k in range(num_nodes):
                        if k != i and k != j and adj_matrix[i, k] and adj_matrix[j, k]:
                            can_add_edge = False
                            break
                    if can_add_edge:
                        adj_matrix[i, j] = adj_matrix[j, i] = random.randint(0, 1)
        else:
            # Fill upper triangle (excluding diagonal) with random 0s and 1s
            for i in range(num_nodes):
                for j in range(i+1, num_nodes):
                    adj_matrix[i, j] = random.randint(0, 1)

            # Make the matrix symmetric
            adj_matrix = np.maximum(adj_matrix, adj_matrix.T)

        # Handle triangle enforcement for True case
        if enforce_triangle is True and not has_triangle(adj_matrix):
            # Ensure at least one triangle
            while not has_triangle(adj_matrix):
                i, j, k = random.sample(range(num_nodes), 3)
                adj_matrix[i, j] = adj_matrix[j, i] = 1
                adj_matrix[i, k] = adj_matrix[k, i] = 1
                adj_matrix[j, k] = adj_matrix[k, j] = 1

        return adj_matrix

    # Generate the specified number of graphs
    graphs = [generate_single_graph() for _ in range(num_graphs)]

    return graphs

def show_graph_with_labels(adj_matrix: np.ndarray, labels: list=None) -> None:
    """
    Show a graph visualization of the given adjacency matrix.

    Parameters:
    adj_matrix (np.ndarray): Adjacency matrix of the graph.
    labels (list): List of node labels.

    Return:
    None
    """
    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix)

    # If labels are not provided, use default integer labels
    if labels is None:
        labels = {i: str(i) for i in range(len(adj_matrix))}

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, labels=labels, with_labels=True, node_color='lightblue',
            node_size=500, font_size=16, font_weight='bold')

    # Draw edge labels
    edge_labels = {(i, j): '' for i, j in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Graph Visualization")
    plt.axis('off')
    plt.show()

def show_interactive_graph_with_labels(adj_matrix: np.ndarray, labels: list=None) -> None:
    """
    Show an interactive graph visualization of the given adjacency matrix.

    Parameters:
    adj_matrix (np.ndarray): Adjacency matrix of the graph.
    labels (list): List of node labels.

    Return:
    None
    """
    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix)

    # If labels are not provided, use default integer labels
    if labels is None:
        labels = {i: str(i) for i in range(len(adj_matrix))}

    # Create a Pyvis network
    net = Network(notebook=True, directed=False, height="500px", width="100%", cdn_resources='remote')

    # Add nodes to the network
    for node in G.nodes():
        net.add_node(node, label=labels[node], title=f"Node: {labels[node]}\nConnections: {G.degree(node)}")

    # Add edges to the network
    for edge in G.edges():
        net.add_edge(edge[0], edge[1])

    # Set physics layout
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=200, spring_strength=0.05, damping=0.09)

    # Generate and display the HTML
    html = net.generate_html()
    display(HTML(html))

def verify_graphs(graphs: list, enforce_triangle: bool) -> int:
    """
    Returns the number of errors found

    Parameters:
    graphs (list): List of adjacency matrices representing the generated graphs.
    enforce_triangle (bool): Whether to enforce the presence of a triangle in the graph.

    Return:
    int: Number of errors found
    """
    results = []
    for graph in graphs:
        has_tri = has_triangle(graph)
        if enforce_triangle is False:
            results.append(has_tri)
        elif enforce_triangle is True:
            results.append(not has_tri)
        else:  # enforce_triangle is None
            results.append(True)  # Always true for random graphs
    return np.sum(np.array(results))

def get_unique_adj_matrices(adj_matrices):
  all_graphs = [ tuple([tuple(r.tolist()) for r in g])  for g in adj_matrices]
  unique_graphs = set([ tuple([tuple(r.tolist()) for r in g]) for g in adj_matrices])
  unique_graphs_npy = [ np.array([list(r) for r in g])  for g in unique_graphs]
  return unique_graphs_npy

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def find_triangles(adj_matrices):
  edge_labels = []
  num_nodes = adj_matrices[0].shape[0]
  for adj in adj_matrices:
    triangle_edges = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if adj[i, j] == 1:
                for k in range(j+1, num_nodes):
                    if adj[i, k] == 1 and adj[j, k] == 1:
                        triangle_edges[i, j] = triangle_edges[j, i] = 1
                        triangle_edges[i, k] = triangle_edges[k, i] = 1
                        triangle_edges[j, k] = triangle_edges[k, j] = 1
    edge_labels.append(torch.tensor(triangle_edges, dtype=torch.float32))
  return edge_labels
# -------------------