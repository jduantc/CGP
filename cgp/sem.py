"""Linear SEM parameter estimation.

Adapted from search_util.py in the fges-py GitHub repository by Iman Wahle,
7/9/2023 (see external/SETUP.md for provenance of the fges-py dependency).
"""

import numpy as np


def mean_shift_data(data):
    """Shift all variables in a dataset to have mean zero."""
    return data - np.mean(data, axis=0)


def estimate_parameters(dag, data):
    """Estimate the parameters of a DAG to fit the data via least squares.

    :param dag: num_nodes x num_nodes adjacency matrix, dag[i, j] == 1 means
        edge i -> j.
    :param data: num_samples x num_nodes array of observations.
    :return: (edge_parameters, residuals) where edge_parameters[i, j] is the
        weight of edge i -> j, and residuals is a diagonal matrix whose
        [j, j] entry is the sum of squared residuals for node j, divided by
        (num_samples - 1).
    """
    data = mean_shift_data(data)
    num_nodes = dag.shape[0]

    edge_parameters = np.zeros((num_nodes, num_nodes))
    residuals = np.zeros((num_nodes, num_nodes))

    for j in range(num_nodes):
        inbound_nodes = [i for i in range(num_nodes) if dag[i, j] == 1]

        if len(inbound_nodes) == 0:
            residuals[j, j] = np.var(data[:, j])
            continue

        a = data[:, inbound_nodes]
        b = data[:, j]

        params, _, _, _ = np.linalg.lstsq(a, b, rcond=None)

        # Computed directly (rather than from lstsq's `residuals` output)
        # since that output is an empty array whenever the regression is
        # rank-deficient, which np.linalg.lstsq does not guard against here.
        sum_sq_residuals = np.sum((b - a @ params) ** 2)
        residuals[j, j] = sum_sq_residuals / (data.shape[0] - 1)

        for i, node in enumerate(inbound_nodes):
            edge_parameters[node, j] = params[i]

    return edge_parameters, residuals


def get_covariance_matrix(params, resids):
    """Get the covariance matrix implied by a DAG's edge parameters and residuals.

    For the equation, see "Causal Mapping of Emotion Networks in the Human
    Brain" (p. 15). params[i, j] is the weight for edge i -> j.
    """
    identity = np.identity(params.shape[0])
    a = np.linalg.inv(identity - params.transpose())
    return np.matmul(np.matmul(a, resids), np.transpose(a))


def get_correlation_matrix(params, resids):
    """Get the correlation matrix implied by a DAG's edge parameters and residuals."""
    cov = get_covariance_matrix(params, resids)
    stdev_products = np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
    return cov / stdev_products
