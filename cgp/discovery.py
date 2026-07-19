"""Score-based greedy DAG discovery -- the scalable replacement for CGP's
hand-enumerated Markov-equivalence-class candidate list.

CGP (cgp/probing.py) searches a hardcoded list of the 11 three-node
equivalence classes. That list is ~11 for 3 nodes but hundreds of DAGs for
4 nodes and tens of thousands for 5 -- enumeration doesn't scale. This
module instead DISCOVERS a good DAG for a given dataset directly, via a
greedy hill-climb over add/remove/reverse edge moves, scored by BIC. It
works for any number of nodes, and (via `candidate_pool`) generates a
small set of candidate DAGs from data bootstraps to feed the rest of the
CGP pipeline in place of the enumerated list.

Scored by BIC, NOT raw Averaged Residuals: AR has no complexity penalty
(see docs/notes/), so a pure-AR greedy search marches straight to the
fully-connected DAG. BIC's per-parameter penalty is what makes the search
prefer the true sparse structure. For degree>1 the per-node fit uses
polynomial features (still recovers nonlinear additive-noise structure);
degree==1 uses the fast covstats Schur-complement closed form.
"""

import networkx as nx
import numpy as np

from cgp.covstats import centered_scatter, raw_stats


def _resid_var_linear(S, n, j, parents):
    """ML residual variance of node j given parents, from the centered scatter S."""
    if not parents:
        return S[j, j] / n
    S_PP = S[np.ix_(parents, parents)]
    S_Pj = S[parents, j]
    schur = S[j, j] - S_Pj @ np.linalg.solve(S_PP, S_Pj)
    return schur / n


def _resid_var_poly(centered, n, j, parents, degree):
    """ML residual variance of node j given parents, via polynomial-feature
    regression (degree>1 -- captures nonlinear additive-noise mechanisms)."""
    if not parents:
        return np.var(centered[:, j])
    cols = [centered[:, p] ** d for p in parents for d in range(1, degree + 1)]
    x = np.column_stack(cols)
    x = x - x.mean(axis=0)
    y = centered[:, j]
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return np.mean((y - x @ coef) ** 2)


def bic_score(dag, data, degree=1, penalty=1.0, _cache=None):
    """BIC of a DAG for `data` (lower is better). Gaussian-SEM BIC:
    n * sum_j log(resid_var_j) + penalty * (#params) * log(n), up to an
    additive constant that doesn't affect comparisons.

    Node count is arbitrary -- this is the whole point vs. enumeration.
    """
    n, num_vars = data.shape
    degree = int(degree)
    if degree == 1:
        S = centered_scatter(*raw_stats(data))
        resid = lambda j, P: _resid_var_linear(S, n, j, P)
    else:
        centered = data - data.mean(axis=0)
        resid = lambda j, P: _resid_var_poly(centered, n, j, P, degree)

    total, n_params = 0.0, 0
    for j in range(num_vars):
        parents = [i for i in range(num_vars) if dag[i, j] == 1]
        rv = max(resid(j, parents), 1e-12)
        total += n * np.log(rv)
        n_params += len(parents) * degree + 1  # coeffs per parent (x degree) + noise var
    return total + penalty * n_params * np.log(n)


def _adj_to_nx(dag):
    g = nx.DiGraph()
    g.add_nodes_from(range(dag.shape[0]))
    g.add_edges_from(zip(*np.where(dag == 1)))
    return g


def greedy_dag_search(data, degree=1, penalty=1.0, max_steps=200, init_dag=None):
    """Greedy hill-climb over add/remove/reverse edge moves, minimizing BIC,
    maintaining acyclicity throughout. Returns the best adjacency matrix found.

    :param init_dag: optional starting adjacency (defaults to the empty graph).
    """
    num_vars = data.shape[1]
    dag = np.zeros((num_vars, num_vars)) if init_dag is None else init_dag.copy()
    best_score = bic_score(dag, data, degree, penalty)

    for _step in range(max_steps):
        best_move, move_score = None, best_score
        for i in range(num_vars):
            for j in range(num_vars):
                if i == j:
                    continue
                if dag[i, j] == 1:
                    moves = [("remove", i, j), ("reverse", i, j)]
                else:
                    moves = [("add", i, j)]
                for kind, a, b in moves:
                    cand = dag.copy()
                    if kind == "add":
                        cand[a, b] = 1
                    elif kind == "remove":
                        cand[a, b] = 0
                    else:  # reverse
                        cand[a, b] = 0
                        cand[b, a] = 1
                    if not nx.is_directed_acyclic_graph(_adj_to_nx(cand)):
                        continue
                    score = bic_score(cand, data, degree, penalty)
                    if score < move_score - 1e-9:
                        best_move, move_score = cand, score
        if best_move is None:
            break
        dag, best_score = best_move, move_score

    return dag


def candidate_pool(data, n_bootstraps=20, bootstrap_frac=0.6, degree=1, penalty=1.0, rng=None):
    """Generate a small pool of distinct candidate DAGs by running greedy
    discovery on random bootstrap subsets -- the scalable stand-in for CGP's
    enumerated equivalence-class list.

    In a genuine mixture, different bootstraps emphasize different latent
    components, so the pool tends to contain each true graph's structure.

    :return: list of (edge_list, count) sorted by descending frequency --
        edge_list is a sorted list of (parent, child) tuples; count is how
        many bootstraps discovered it.
    """
    rng = rng if rng is not None else np.random.default_rng()
    n = data.shape[0]
    size = max(int(bootstrap_frac * n), 10)
    counts = {}
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=size, replace=True)
        dag = greedy_dag_search(data[idx], degree=degree, penalty=penalty)
        key = tuple(sorted((int(p), int(c)) for p, c in zip(*np.where(dag == 1))))
        counts[key] = counts.get(key, 0) + 1
    return sorted(([list(k), c] for k, c in counts.items()), key=lambda kc: -kc[1])
