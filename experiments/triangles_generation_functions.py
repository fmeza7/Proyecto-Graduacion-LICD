# IMPORTS
from positional_encodings.torch_encodings import *
import numpy as np
import random
import networkx as nx
from collections import defaultdict
from itertools import permutations
from collections import defaultdict

import matplotlib.pyplot as plt

from triangles_auxiliary_functions import generate_random_graphs, get_unique_adj_matrices
# ----------------------------

#  Generation Functions from Dataset 1 -> Meeting 25-07-24

# GENERATION FUNCTIONS
# -------------------
# Dataset 1
def generate_dataset(dataset_size_target=6000, node_size_list=range(5, 11)):
  node_sizes = len(node_size_list)
  dataset_size_target // node_sizes
  node_size_target = dataset_size_target // node_sizes

  num_nodes = 4
  2**((num_nodes**2 - num_nodes)//2)

  adj_matrices = []
  labels = []
  sizes = dict()
  total_sizes = {"with": 0, "without": 0}
  for num_nodes in node_size_list:

    current_node_size_target = node_size_target

    if num_nodes <=5:
      max_existent_graphs = 2**((num_nodes**2 - num_nodes)//2)
      current_node_size_target = min(node_size_target, max_existent_graphs)
      node_size_target += node_size_target - current_node_size_target


    graphs_with_triangle = generate_random_graphs(num_nodes=num_nodes, num_graphs=current_node_size_target//2, enforce_triangle=True)
    graphs_without_triangle = generate_random_graphs(num_nodes=num_nodes, num_graphs=current_node_size_target//2, enforce_triangle=False)

    graphs_with_triangle_unique = get_unique_adj_matrices(graphs_with_triangle)
    graphs_without_triangle_unique = get_unique_adj_matrices(graphs_without_triangle)

    current_adj_matrices = graphs_with_triangle_unique + graphs_without_triangle_unique
    current_labels = [1 for _ in range(len(graphs_with_triangle_unique))] + [0 for _ in range(len(graphs_without_triangle_unique))]
    adj_matrices += current_adj_matrices
    labels += current_labels
    sizes[num_nodes] = {"with": len(graphs_with_triangle_unique), "without": len(graphs_without_triangle_unique)}
    total_sizes["with"] += len(graphs_with_triangle_unique)
    total_sizes["without"] += len(graphs_without_triangle_unique)

  dataset_size = len(adj_matrices)


  print("Dataset Size:", dataset_size)
  print("With Triangle Size:", total_sizes["with"])
  print("Without Triange Size:", total_sizes["without"])
  print("Totals for each node size:")
  print(sizes)
  ids = np.random.choice(range(dataset_size), size=dataset_size)
  adj_matrices, labels = zip(*random.sample(list(zip(adj_matrices, labels)), dataset_size))

  return np.array(adj_matrices), labels

# Dataset 2
def generate_non_isomorphic_graphs(num_nodes: int, allow_cycles: bool) -> list:
    """
    Generate non-isomorphic graphs with a specified number of nodes.

    Parameters:
    num_nodes (int): Number of nodes in the graph.
    allow_cycles (bool): Whether to allow cycles in the generated graphs.

    Return:
    list: List of generated graphs.
    """
    def generate_random_tree(n: int) -> nx.Graph:
        """
        Generate a random tree with a specified number of nodes.

        Parameters:
        n (int): Number of nodes in the tree.

        Return:
        nx.Graph: Generated tree.
        """
        tree = nx.random_tree(n)
        return tree

    def generate_random_graph_with_one_triangle(n: int, node_participation: dict) -> nx.Graph:
        """
        Generate a random graph with a specified number of nodes and one triangle.

        Parameters:
        n (int): Number of nodes in the graph.
        node_participation (dict): Dictionary to keep track of the nodes that participate in triangles.

        Return:
        nx.Graph: Generated graph.
        """
        # Create a tree
        graph = generate_random_tree(n)
           # List all nodes with degree 2 or more
        nodes_with_degree_2 = [node for node in graph.nodes() if graph.degree(node) >= 2]
        # Select a random node with degree 2 or more
        node = random.choice(nodes_with_degree_2)

        neighbors = list(graph.neighbors(node))
        # Selected randomly 2 numbers from the neighbors of the node
        a, b = random.sample(neighbors, 2)
        graph.add_edge(a, b)

        triangle_nodes = [node, a, b]

        # Add one triangle (3-node cycle) to the directed graph
        if n >= 3:
            for node in triangle_nodes:
                node_participation[node] += 1

        return graph

    graphs = []
    node_participation = defaultdict(int)
    if allow_cycles:
        for _ in range(1000):  # Generate 1000 different graphs to ensure non-isomorphism
            graphs.append(generate_random_graph_with_one_triangle(num_nodes, node_participation))
    else:
        for _ in range(1000):
            graphs.append(generate_random_tree(num_nodes))

    non_isomorphic_graphs = []
    for g in graphs:
        if all(not nx.is_isomorphic(g, existing_g) for existing_g in non_isomorphic_graphs):
            non_isomorphic_graphs.append(g)
        # non_isomorphic_graphs.append(g)
    print(len(non_isomorphic_graphs))
    # Print the number of edges in each graph and which type of graph it is
    # for g in non_isomorphic_graphs:
    #     print(f"Number of edges: {g.number_of_edges()}, Is tree: {nx.is_tree(g)}")
    return non_isomorphic_graphs, node_participation

def plot_graphs(graphs: list, node_participation: dict, num_nodes: int, allow_cycles: bool) -> None:
    """
    Plot the first 10 graphs in the list.

    Parameters:
    graphs (list): List of graphs to plot.
    node_participation (dict): Dictionary of the nodes that participate in cycles.
    num_nodes (int): Number of nodes in the graphs.
    allow_cycles (bool): Whether the graphs allow cycles

    Return:
    None
    """
    plt.figure(figsize=(20, 10))
    for i, g in enumerate(graphs[:10]):
        plt.subplot(2, 5, i + 1)
        nx.draw(g, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
        plt.title(f'Graph {i + 1}')
    plt.show()

    # if allow_cycles:
    #     total_cycles = sum(node_participation.values()) // 3  # Each cycle involves 3 nodes
    #     for node in range(num_nodes):
    #         percentage = (node_participation[node] / total_cycles) * 100
    #         #print(f'Node {node} is part of {percentage:.2f}% of the total cycles.')


def generate_unique_random_graphs_with_triangles(num_nodes, edge_probability=0.3, num_graphs=1000, max_attempts=10000):
    """
    Generate unique random graphs with annotated triangle edges.

    Args:
    num_nodes (int): Number of nodes in each graph.
    edge_probability (float): Probability of an edge between any two nodes.
    num_graphs (int): Number of unique graphs to generate.
    max_attempts (int): Maximum number of attempts to generate unique graphs.

    Returns:
    tuple: (adj_matrices, edge_labels)
        adj_matrices: List of unique adjacency matrices (torch.Tensor)
        edge_labels: List of corresponding edge label matrices (torch.Tensor)
    """
    adj_matrices = []
    edge_labels = []
    graph_hashes = set()
    attempts = 0

    while len(adj_matrices) < num_graphs and attempts < max_attempts:
        # Generate random adjacency matrix
        adj_matrix = np.random.choice([0, 1], size=(num_nodes, num_nodes), p=[1-edge_probability, edge_probability])
        adj_matrix = np.triu(adj_matrix, 1)  # Upper triangular to avoid self-loops
        adj_matrix = adj_matrix + adj_matrix.T  # Make it symmetric

        # Generate a hash for this graph
        graph_hash = hashlib.md5(adj_matrix.tobytes()).hexdigest()

        # If this is a new, unique graph, process it
        if graph_hash not in graph_hashes:
            graph_hashes.add(graph_hash)

            # Find triangles
            triangle_edges = np.zeros((num_nodes, num_nodes))
            for i in range(num_nodes):
                for j in range(i+1, num_nodes):
                    if adj_matrix[i, j] == 1:
                        for k in range(j+1, num_nodes):
                            if adj_matrix[i, k] == 1 and adj_matrix[j, k] == 1:
                                triangle_edges[i, j] = triangle_edges[j, i] = 1
                                triangle_edges[i, k] = triangle_edges[k, i] = 1
                                triangle_edges[j, k] = triangle_edges[k, j] = 1

            adj_matrices.append(torch.tensor(adj_matrix, dtype=torch.float32))
            edge_labels.append(torch.tensor(triangle_edges, dtype=torch.float32))

        attempts += 1

    if len(adj_matrices) < num_graphs:
        print(f"Warning: Only generated {len(adj_matrices)} unique graphs out of {num_graphs} requested.")

    return adj_matrices, edge_labels


def generate_isomorphic_graphs(adj_matrix):
    n = len(adj_matrix)
    vertices = list(range(n))

    # Generate all possible permutations of vertex labelings
    all_permutations = permutations(vertices)

    isomorphic_matrices = set()

    for perm in all_permutations:
        # Create a new matrix based on the permutation
        new_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                new_matrix[perm[i]][perm[j]] = adj_matrix[i][j]

        # Convert the matrix to a tuple for hashing (set membership)
        matrix_tuple = tuple(map(tuple, new_matrix))
        isomorphic_matrices.add(matrix_tuple)

    # Convert back to list of numpy arrays
    return [np.array(matrix) for matrix in isomorphic_matrices]

# -------------------