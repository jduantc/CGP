"""Averaged Residuals (AR): the loss function used to score how well a DAG fits a dataset.

AR(g, D) is the average, across nodes, of the per-node sum-of-squared-residuals
from fitting a linear SEM with structure g to dataset D. Lower AR means g fits D
more closely.

Computed via cgp.covstats's sufficient-statistics form (see docs/notes/covariance_ar.md),
which is exactly equivalent to fitting each node's OLS regression directly
(cgp.sem.estimate_parameters, kept as the reference implementation and used by
tests/test_covstats.py to pin the two together) but far cheaper when scoring
many DAGs or many point-set unions against the same data, as cgp.probing does.
"""

import numpy as np

from cgp.covstats import ar_from_stats, raw_stats


def edgelist_to_matrix(edges, num_vars):
    """Convert a list of (parent, child) edge tuples into an adjacency matrix.

    edges is typically the output of an FGES search, which returns each
    undirected pair only once; this fills in a directed adjacency matrix
    with entry [i, j] == 1 for edge i -> j.
    """
    edge_mat = np.zeros((num_vars, num_vars))
    for parent, child in edges:
        if edge_mat[child, parent] == 0:
            edge_mat[parent, child] = 1
    return edge_mat


def averaged_residuals(dag, data, var_names=None, plot=False):
    """Compute the Averaged Residuals (AR) score of a DAG against a dataset.

    :param dag: either an adjacency matrix (num_vars x num_vars) or an edge
        list of (parent, child) tuples.
    :param data: num_samples x num_vars array of observations.
    :param var_names: optional labels for plotting.
    :param plot: if True, show edge-weight and residual heatmaps.
    :return: the AR score (float).
    """
    num_vars = data.shape[1]
    if not isinstance(dag, np.ndarray):
        dag = edgelist_to_matrix(dag, num_vars)

    # raw_stats + centered_scatter (inside ar_from_stats) handles centering
    # correctly regardless of data's mean -- no need to pre-center here.
    ar_score = ar_from_stats(dag, raw_stats(data))

    if plot:
        from cgp.sem import estimate_parameters
        from cgp.viz import plot_ar_diagnostics
        edge_params, residuals = estimate_parameters(dag, data)
        plot_ar_diagnostics(edge_params, residuals, var_names or [f"X{i+1}" for i in range(num_vars)])

    return ar_score
