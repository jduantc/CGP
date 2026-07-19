"""Mixture-of-SEMs EM: a principled replacement for the affinity/threshold/
white-point heuristic in cgp.probing.

Recognizes the CGP problem as fitting a Gaussian mixture where each
component's covariance is constrained to factor according to a DAG (a
linear-Gaussian SEM), instead of a free covariance as in an ordinary GMM.
Fitting via EM gives:

  - Soft point-to-component responsibilities (replacing the affinity/
    threshold cutoff and the ad hoc random white-point splitting).
  - A real log-likelihood objective with a BIC-based complexity penalty
    (Averaged Residuals has none -- see docs/notes/mixture_em.md -- which
    lets a denser or merely different graph win by chance; BIC penalizes
    extra edge parameters).
  - A direct generalization to K > 2 candidate DAGs: nothing here assumes
    exactly two components.

Reuses cgp.covstats's sufficient-statistics machinery for the M-step: a
weighted M-step is just an ordinary M-step fed WEIGHTED sufficient
statistics, with no new math.
"""

from itertools import combinations

import numpy as np
from scipy.special import logsumexp

from cgp.covstats import edge_params_from_stats, residuals_from_stats
from cgp.scoring import edgelist_to_matrix
from cgp.sem import get_covariance_matrix


def weighted_stats(data, weights):
    """Sufficient statistics of `data` under per-row weights (a weighted
    generalization of cgp.covstats.raw_stats; weights all 1 reduces to it)."""
    n_eff = weights.sum()
    s = weights @ data
    Q = data.T @ (weights[:, None] * data)
    return n_eff, s, Q


def _mvn_logpdf_zero_mean(data, cov):
    """log N(x; 0, cov) for every row of data, via Cholesky (stable, and
    batches cheaply since num_vars is tiny).

    Retries with escalating diagonal jitter if the covariance is not
    positive-definite. A linear-Gaussian component covariance can go
    singular/non-PD when the true mechanism is strongly nonlinear (the
    linear-SEM fit is badly misspecified) -- standard mixture-model
    regularization keeps EM producing an answer instead of crashing.
    """
    d = cov.shape[0]
    scale = np.trace(cov) / d
    for jitter in (0.0, 1e-9, 1e-6, 1e-3):
        try:
            chol = np.linalg.cholesky(cov + jitter * scale * np.eye(d))
            break
        except np.linalg.LinAlgError:
            continue
    else:
        raise np.linalg.LinAlgError("covariance not PD even after jitter")
    z = np.linalg.solve(chol, data.T)
    quad = np.sum(z ** 2, axis=0)
    log_det = 2 * np.sum(np.log(np.diag(chol)))
    return -0.5 * (d * np.log(2 * np.pi) + log_det + quad)


def _to_matrices(dags, num_vars):
    return [d if isinstance(d, np.ndarray) else edgelist_to_matrix(d, num_vars) for d in dags]


def _m_step(data, dag_mats, responsibilities):
    mixing = responsibilities.mean(axis=0)
    params_list, resid_list, cov_list = [], [], []
    for k, dag in enumerate(dag_mats):
        stats = weighted_stats(data, responsibilities[:, k])
        params = edge_params_from_stats(dag, stats)
        resid = residuals_from_stats(dag, stats)
        params_list.append(params)
        resid_list.append(resid)
        cov_list.append(get_covariance_matrix(params, resid))
    return params_list, resid_list, cov_list, mixing


def _e_step(data, cov_list, mixing):
    log_liks = np.stack(
        [np.log(mixing[k] + 1e-300) + _mvn_logpdf_zero_mean(data, cov) for k, cov in enumerate(cov_list)],
        axis=1,
    )
    log_norm = logsumexp(log_liks, axis=1)
    responsibilities = np.exp(log_liks - log_norm[:, None])
    return responsibilities, log_norm.sum()


def _bic(loglik, dag_mats, num_vars, n):
    # free params: one coefficient per edge, plus one noise variance per node, per
    # component, plus (K-1) free mixing weights.
    n_params = sum(int(dag.sum()) + num_vars for dag in dag_mats) + (len(dag_mats) - 1)
    return -2 * loglik + n_params * np.log(n)


def em_fit(data, dags, n_iter=100, tol=1e-6, n_init=3, rng=None):
    """Fit a mixture of linear-Gaussian SEMs with fixed structures `dags` to `data`.

    :param dags: list of K candidate DAGs (edge lists or adjacency matrices);
        K=2 recovers the CGP use case, K>2 is a direct generalization.
    :param n_init: number of random initializations (EM only finds a local
        optimum); the highest-log-likelihood run is kept.
    :return: dict with params, residuals, cov, mixing, responsibilities,
        loglik, loglik_history, bic (all for the best of n_init runs).
    """
    rng = rng if rng is not None else np.random.default_rng()
    centered = data - data.mean(axis=0)
    n, num_vars = centered.shape
    dag_mats = _to_matrices(dags, num_vars)
    k = len(dag_mats)

    best = None
    for _ in range(n_init):
        responsibilities = rng.dirichlet(np.ones(k), size=n)
        history = []
        prev_loglik = -np.inf
        for _iteration in range(n_iter):
            params_list, resid_list, cov_list, mixing = _m_step(centered, dag_mats, responsibilities)
            responsibilities, loglik = _e_step(centered, cov_list, mixing)
            history.append(loglik)
            if abs(loglik - prev_loglik) < tol * max(abs(prev_loglik), 1.0):
                break
            prev_loglik = loglik

        result = {
            "params": params_list, "residuals": resid_list, "cov": cov_list,
            "mixing": mixing, "responsibilities": responsibilities,
            "loglik": loglik, "loglik_history": history,
            "bic": _bic(loglik, dag_mats, num_vars, n),
        }
        if best is None or result["loglik"] > best["loglik"]:
            best = result
    return best


def search_best_graph_set(data, candidate_graphs, set_size=2, n_iter=100, n_init=3,
                           max_sets=None, rng=None):
    """Try every `set_size`-combination of candidate_graphs, fit a mixture to
    each, and rank by BIC (lower is better).

    :param max_sets: if given, only evaluate the first `max_sets` combinations
        (in the itertools.combinations order) -- use to bound runtime for
        large candidate lists; the caller should log what was skipped.
    :return: list of (bic, graph_set, em_result), sorted ascending by BIC.
    """
    rng = rng if rng is not None else np.random.default_rng()
    sets = list(combinations(range(len(candidate_graphs)), set_size))
    if max_sets is not None:
        sets = sets[:max_sets]

    results = []
    for idxs in sets:
        graphs = [candidate_graphs[i] for i in idxs]
        fit = em_fit(data, graphs, n_iter=n_iter, n_init=n_init, rng=rng)
        results.append((fit["bic"], idxs, fit))
    results.sort(key=lambda r: r[0])
    return results
