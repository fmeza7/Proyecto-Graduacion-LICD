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

# ATTENTION ANALYSIS FUNCTIONS
# -------------------
def visualize_single_edge_attention(attention_weights, layer_index, num_nodes, source_edge, index=0):
    # Extract attention weights for the specified layer
    layer_weights = attention_weights[layer_index]

    # Average over the batch dimension
    #avg_attn_map = layer_weights.mean(dim=0).cpu().detach().numpy()
    avg_attn_map = layer_weights[index, :, :].cpu().detach().numpy()

    # Calculate the index of the source edge
    source_i, source_j = source_edge
    source_edge_index = source_i * num_nodes + source_j

    # Extract attention weights for the specified source edge
    source_edge_attention = avg_attn_map[source_edge_index]

    # Create edge labels
    source_edge_label = f"{source_i}->{source_j}"

    # Reshape attention weights into a 2D grid
    attention_grid = source_edge_attention.reshape(num_nodes, num_nodes)

    # Create a heatmap
    plt.figure(figsize=(8, 8))
    sns.heatmap(attention_grid, annot=True, cmap="YlGnBu", fmt=".2f",
                xticklabels=range(num_nodes), yticklabels=range(num_nodes))
    plt.title(f"Attention from Edge {source_edge_label} - Layer {layer_index + 1}")
    plt.xlabel("Target Node (To)")
    plt.ylabel("Target Node (From)")
    plt.tight_layout()
    plt.show()

def visualize_edge_attention(attention_weights, layer_index, num_nodes, index=0, figsize=(10, 10)):
    # Extract attention weights for the specified layer
    layer_weights = attention_weights[layer_index]

    # Average over the batch dimension
    #avg_attn_map = layer_weights.mean(dim=0).cpu().detach().numpy()
    avg_attn_map = layer_weights[index, :, :].cpu().detach().numpy()

    # Create edge labels
    edge_labels = [f"{i}->{j}" for i in range(num_nodes) for j in range(num_nodes)]

    # Create a heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(avg_attn_map, annot=False, cmap="YlGnBu", xticklabels=edge_labels, yticklabels=edge_labels)
    plt.title(f"Edge Attention Map - Layer {layer_index + 1}")
    plt.xlabel("Target Edge")
    plt.ylabel("Source Edge")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def get_correct_example(model, test_loader, device, desired_label=None):
    model.eval()
    with torch.no_grad():
        for adj, labels in test_loader:
            adj, labels = adj.to(device), labels.to(device)
            outputs, attention_weights = model(adj)
            predicted = (outputs > 0.5).float()

            # Find correct predictions
            correct_mask = (predicted == labels).cpu().numpy()

            # If desired_label is specified, only consider examples with that label
            if desired_label is not None:
                label_mask = (labels == desired_label).cpu().numpy()
                correct_mask = correct_mask & label_mask

            correct_indices = np.where(correct_mask)[0]

            if len(correct_indices) > 0:
                # Choose a random correct example
                idx = np.random.choice(correct_indices)
                return adj[idx], labels[idx], outputs[idx], attention_weights

    # If no correct examples found
    return None, None, None, None
# -------------------