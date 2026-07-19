"""Nonlinear Averaged Residuals: score a DAG by how well each node is
explained by a POLYNOMIAL regression on its parents, instead of a linear
one.

The key point (and why this is cheap): polynomial features are
linear-in-parameters, so augmenting the data X -> [x_j, x_j^2, ..., x_j^deg
for each node j] keeps the ENTIRE covstats sufficient-statistics machinery
valid. A node's degree-`deg` regression on its parents is just a linear
regression of that node's degree-1 column on its parents' full polynomial
blocks -- a Schur complement in the augmented scatter matrix. So nonlinear
AR is still additive-across-point-sets and batchable, exactly like the
linear version (cgp/covstats.py), and slots into CGP's affinity/swing
machinery unchanged.

This is the axis where AR/CGP has a structural advantage over
likelihood-based mixture EM: AR needs only a goodness-of-fit RESIDUAL, so
swapping a linear fit for a polynomial one is trivial. A Gaussian-mixture
EM needs a valid normalized joint DENSITY per component, which the
closed-form linear-SEM covariance only provides in the linear-Gaussian
case (see docs/notes/polynomial_features_em.md for what breaks).
"""

import numpy as np

from cgp.covstats import centered_scatter


def augment(data, degree):
    """Polynomial-feature augmentation, blocked by node: column (j*degree + k)
    is x_j ** (k+1). So node j occupies columns [j*degree : (j+1)*degree] and
    its linear term is column j*degree."""
    cols = [data[:, j] ** k for j in range(data.shape[1]) for k in range(1, degree + 1)]
    return np.column_stack(cols)


def augmented_stats(data, degree):
    """Sufficient statistics (n, sum, X_aug^T X_aug) of the augmented data --
    additive across disjoint point sets, like cgp.covstats.raw_stats."""
    aug = augment(data, degree)
    return aug.shape[0], aug.sum(axis=0), aug.T @ aug


def nl_ar_batch(dag, degree, n, s, Q, ridge=1e-6):
    """Nonlinear (degree-`degree`) Averaged Residuals for one DAG against a
    BATCH of point sets given as AUGMENTED sufficient stats (from
    augmented_stats). n: scalar or (B,); s: (B, D); Q: (B, D, D), where
    D = num_vars * degree. Returns (B,) AR values. degree==1 reduces exactly
    to the linear cgp.covstats.ar_batch.

    For each original node j, the target is its linear column (j*degree) and
    the predictors are its parents' full polynomial blocks. A small ridge
    stabilizes the solve, since x, x^2, x^3 columns are correlated.
    """
    num_vars = dag.shape[0]
    n_arr = np.broadcast_to(np.asarray(n, dtype=float), (Q.shape[0],)).astype(float)
    degenerate = n_arr < 2
    n_safe = np.where(degenerate, 2.0, n_arr)
    s_safe = np.where(degenerate[:, None], 0.0, s)
    Q_safe = np.where(degenerate[:, None, None], np.eye(Q.shape[-1]), Q)

    S = centered_scatter(n_safe, s_safe, Q_safe)          # (B, D, D)
    batch_size = S.shape[0]
    total = np.zeros(batch_size)
    for j in range(num_vars):
        target = j * degree                                # linear column of node j
        parents = [i for i in range(num_vars) if dag[i, j] == 1]
        if not parents:
            total += S[:, target, target] / n_safe
            continue
        pred = [p * degree + k for p in parents for k in range(degree)]
        S_PP = S[np.ix_(range(batch_size), pred, pred)]
        S_PP = S_PP + ridge * np.eye(len(pred))
        S_Pt = S[:, pred, target]
        sol = np.linalg.solve(S_PP, S_Pt[:, :, None])[:, :, 0]
        ssr = S[:, target, target] - np.einsum("bi,bi->b", S_Pt, sol)
        total += ssr / (n_safe - 1)

    return np.where(degenerate, np.inf, total / num_vars)


def nl_ar(dag, data, degree):
    """Convenience: nonlinear AR of one DAG (adjacency matrix or edge list)
    against one dataset."""
    if not isinstance(dag, np.ndarray):
        from cgp.scoring import edgelist_to_matrix
        dag = edgelist_to_matrix(dag, data.shape[1])
    n, s, Q = augmented_stats(data, degree)
    return float(nl_ar_batch(dag, degree, np.array([n]), s[None], Q[None])[0])
