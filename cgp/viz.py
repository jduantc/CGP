"""Plotting helpers for CGP: corner plots, adjacency/correlation matrices, graphs."""

import corner
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def generic_var_labels(num_vars):
    """Default labels X1, X2, ... for a dataset with num_vars columns."""
    return [f"X{i + 1}" for i in range(num_vars)]


def plot_matrix(matrix, var_names, title, cmap="Greys", vmin=0, vmax=1, symmetric=False):
    """Display a square matrix (adjacency, correlation, ...) as a heatmap with a colorbar."""
    matrix = np.copy(matrix)
    if symmetric:
        np.fill_diagonal(matrix, 0)
        vbound = np.nanmax(np.abs(matrix))
        vmin, vmax = -vbound, vbound
        cmap = "coolwarm"

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(var_names)))
    ax.set_yticks(np.arange(len(var_names)))
    ax.set_xticklabels(var_names, rotation=90, ha="center")
    ax.set_yticklabels(var_names)
    ax.set_title(title, fontsize=20)
    fig.colorbar(im, ax=ax)
    plt.show()


def plot_ar_diagnostics(edge_params, residuals, var_names):
    """Show edge-weight estimates and per-node residuals side by side."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    cbound = np.max(np.abs(edge_params))
    im0 = ax[0].imshow(edge_params, cmap="coolwarm", vmin=-cbound, vmax=cbound)
    ax[0].set_xticks(np.arange(len(var_names)))
    ax[0].set_yticks(np.arange(len(var_names)))
    ax[0].set_xticklabels(var_names)
    ax[0].set_yticklabels(var_names)
    ax[0].set_title("Edge weight estimates")
    divider0 = make_axes_locatable(ax[0])
    fig.colorbar(im0, cax=divider0.append_axes("right", size="5%", pad=0.05))

    im1 = ax[1].imshow(residuals, cmap="Greys")
    ax[1].set_xticks(np.arange(len(var_names)))
    ax[1].set_yticks(np.arange(len(var_names)))
    ax[1].set_xticklabels(var_names)
    ax[1].set_yticklabels(var_names)
    ax[1].set_title("Residuals")
    divider1 = make_axes_locatable(ax[1])
    fig.colorbar(im1, cax=divider1.append_axes("right", size="5%", pad=0.05))
    plt.show()


def plot_clusters_corner(clusters, colors=None):
    """Overlay corner.py plots for a list of point clusters."""
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    figure = corner.corner(clusters[0], color=colors[0])
    for i in range(1, len(clusters)):
        figure = corner.corner(clusters[i], fig=figure, color=colors[i % len(colors)])
    return figure


def plot_affinity_colormap(clusters, affinities):
    """Corner-plot clusters colored on a diverging scale by their DAG affinity.

    Negative affinity (more strongly fit by DAG 1) is one end of the PiYG
    colormap, positive affinity (DAG 2) the other; affinity near zero is
    the neutral midpoint.
    """
    min_affinity = min(0, np.min(affinities))
    max_affinity = max(0, np.max(affinities))
    bound = max(abs(min_affinity), abs(max_affinity))
    norm = mpl.colors.Normalize(vmin=-bound, vmax=bound)

    mappable = cm.ScalarMappable(norm=norm, cmap=cm.PiYG)
    figure = corner.corner(clusters[0], color=mappable.to_rgba(affinities[0]))
    for i in range(1, len(clusters)):
        figure = corner.corner(clusters[i], fig=figure, color=mappable.to_rgba(affinities[i]))
    return figure


def draw_dag(directed_edges, labels=None):
    """Plot a directed graph from a list of (parent, child) index pairs."""
    graph = nx.DiGraph()
    graph.add_edges_from(directed_edges)
    pos = nx.spectral_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color="#9370db")
    nx.draw_networkx_labels(graph, pos, labels=labels)
    nx.draw_networkx_edges(graph, pos, arrows=True, arrowsize=20)
    plt.show()
