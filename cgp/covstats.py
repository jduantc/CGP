"""Sufficient-statistics form of Averaged Residuals.

For mean-centered data, the OLS residual variance of a node given its
parents is the Schur complement of the sample covariance matrix:

    resid_var(X_j | P) = S_jj - S_jP @ inv(S_PP) @ S_Pj

where S is the centered scatter matrix (n times the covariance). This means
AR(g, D) depends on D only through its sufficient statistics (n, sum, X^T X), which are additive across disjoint
point sets. 


This allows us to:
  1. Score many candidate DAGs against the same point set using one
     scatter-matrix computation instead of re-fitting repeatedly
  2. Score the affinity/swing step in
     cgp/probing.py efficiently by adding each side's sufficient stats,
     instead of re-centering and re-fitting on raw data
"""

import numpy as np


def raw_stats(data):
    """Sufficient statistics (n, sum of rows, X^T X) of a point set."""
    return data.shape[0], data.sum(axis=0), data.T @ data


def combine_stats(*groups):
    """Stats of the union of disjoint point sets: sufficient stats are additive."""
    n = sum(g[0] for g in groups)
    s = sum(g[1] for g in groups)
    Q = sum(g[2] for g in groups)
    return n, s, Q


def centered_scatter(n, s, Q):
    """Centered scatter S = Q - s s^T / n (== n * the ML covariance estimate).

    Broadcasts over a leading batch dimension: n can be scalar or (B,),
    s can be (d,) or (B, d), Q can be (d, d) or (B, d, d).
    """
    return Q - np.einsum("...i,...j->...ij", s, s) / np.asarray(n)[..., None, None]


def _parents(dag, node):
    return [i for i in range(dag.shape[0]) if dag[i, node] == 1]


def edge_params_from_stats(dag, stats):
    """Edge-weight matrix (edge_params[i, j] = weight of i -> j) via one Schur solve per node."""
    n, s, Q = stats
    S = centered_scatter(n, s, Q)
    num_nodes = S.shape[0]
    edge_params = np.zeros((num_nodes, num_nodes))
    for j in range(num_nodes):
        parents = _parents(dag, j)
        if not parents:
            continue
        S_PP, S_Pj = S[np.ix_(parents, parents)], S[parents, j]
        beta = np.linalg.pinv(S_PP) @ S_Pj
        for i, p in enumerate(parents):
            edge_params[p, j] = beta[i]
    return edge_params


def residuals_from_stats(dag, stats):
    """Diagonal residual matrix, matching cgp.sem.estimate_parameters's convention:
    residuals[j, j] = SSR_j / (n - 1) if j has parents, else var(X_j) (SSR_j / n).
    """
    n, s, Q = stats
    S = centered_scatter(n, s, Q)
    num_nodes = S.shape[0]
    residuals = np.zeros((num_nodes, num_nodes))
    for j in range(num_nodes):
        parents = _parents(dag, j)
        if not parents:
            residuals[j, j] = S[j, j] / n
            continue
        S_PP, S_Pj = S[np.ix_(parents, parents)], S[parents, j]
        ssr = S[j, j] - S_Pj @ np.linalg.pinv(S_PP) @ S_Pj
        residuals[j, j] = ssr / (n - 1)
    return residuals


def ar_from_stats(dag, stats):
    """Averaged Residuals score for one DAG against one point set's sufficient stats.

    A point set with fewer than 2 points has no meaningfully defined
    residual variance (and would divide by zero downstream); scored as
    +inf so a degenerate near-empty partition never wins a search over
    candidate DAGs.
    """
    if stats[0] < 2:
        return np.inf
    residuals = residuals_from_stats(dag, stats)
    return np.trace(residuals) / dag.shape[0]


def ar_batch(dag, n, s, Q):
    """AR for one DAG scored against a batch of point sets' sufficient stats.

    n: scalar or (B,); s: (B, d); Q: (B, d, d). Returns (B,) AR values with no
    Python loop over the batch (only over the (small, fixed) number of nodes).
    Batch entries with n < 2 are scored as +inf (see ar_from_stats) -- their
    scatter matrices are substituted with a placeholder
    before the batched solve to prevent
    LinAlgError.
    """
    n_arr = np.broadcast_to(np.asarray(n, dtype=float), (Q.shape[0],)).astype(float)
    degenerate = n_arr < 2

    n_safe = np.where(degenerate, 2.0, n_arr)
    s_safe = np.where(degenerate[:, None], 0.0, s)
    Q_safe = np.where(degenerate[:, None, None], np.eye(Q.shape[-1]), Q)

    S = centered_scatter(n_safe, s_safe, Q_safe)
    batch_size, num_nodes, _ = S.shape
    total = np.zeros(batch_size)
    for j in range(num_nodes):
        parents = _parents(dag, j)
        if not parents:
            total += S[:, j, j] / n_safe
            continue
        S_PP = S[np.ix_(range(batch_size), parents, parents)]
        S_Pj = S[:, parents, j]
        sol = np.linalg.solve(S_PP, S_Pj[:, :, None])[:, :, 0]
        ssr = S[:, j, j] - np.einsum("bi,bi->b", S_Pj, sol)
        total += ssr / (n_safe - 1)

    return np.where(degenerate, np.inf, total / num_nodes)


def ar_batch_multi_dag(dags, n, s, Q):
    """AR for MULTIPLE DAGs scored against a BATCH of point sets' sufficient stats.

    Returns an array of shape (len(dags), B): row k is ar_batch(dags[k], n, s, Q).
    This computes cgp.probing score for one cluster's swing toward
    every candidate DAG in a single pass.
    """
    return np.stack([ar_batch(dag, n, s, Q) for dag in dags], axis=0)
