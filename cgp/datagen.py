"""Synthetic SEM data generation, for building CGP test datasets without TETRAD/JVM.

Supports linear, quadratic, and cubic structural mechanisms over an
arbitrary DAG (as an edge list), so the benchmark suite
(scripts/run_benchmark_suite.py) can characterize how CGP's accuracy
degrades outside the purely linear setting.
"""

import networkx as nx
import numpy as np


def structural_distance(edges1, edges2):
    """Number of edges present in exactly one of the two graphs (a simple
    proxy for how distinguishable two DAGs are -- 0 means identical; the
    "hardest" bundled dataset uses a distance-1 pair)."""
    return len(set(edges1) ^ set(edges2))


def _topological_order(edges, num_vars):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_vars))
    graph.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError(f"edges {edges} do not form a DAG")
    return list(nx.topological_sort(graph))


MECHANISMS = ("linear", "quadratic", "cubic")


def sample_dag_data(edges, num_vars, n_samples, mechanism="linear",
                     coef_range=(0.5, 2.0), quad_coef_range=(0.1, 0.5),
                     cubic_coef_range=(0.02, 0.08), noise_std=1.0, rng=None):
    """Forward-sample data from a structural equation model with the given DAG.

    :param mechanism: 'linear' (child = sum coef*parent + noise), 'quadratic'
        (+ quad_coef*parent**2), or 'cubic' (+ cubic_coef*parent**3).
    :param coef_range, quad_coef_range, cubic_coef_range: magnitude range for
        randomly-signed edge coefficients. The cubic range defaults much
        smaller than the others -- a cubic term's scale grows fast enough
        across even 2-3 topological generations that a coefficient in the
        linear/quadratic range routinely overflows; this range was chosen to
        keep generated values numerically sane at this graph size.
    :return: (data, coefficients) where coefficients maps (parent, child) -> dict.

    Note: every call draws its own fresh random edge coefficients from `rng`.
    Calling this twice for "train" and "test" data -- even reusing the same
    rng stream -- gives two DIFFERENT underlying SEMs, not train/test splits
    of one. Generate one larger dataset and split the rows instead (a subtle
    bug this convention has caused in practice).
    """
    if mechanism not in MECHANISMS:
        raise ValueError(f"unknown mechanism '{mechanism}', must be one of {MECHANISMS}")

    rng = rng if rng is not None else np.random.default_rng()
    order = _topological_order(edges, num_vars)
    parents = {j: [] for j in range(num_vars)}
    for p, c in edges:
        parents[c].append(p)

    coefficients = {}
    for p, c in edges:
        coefficients[(p, c)] = {
            "linear": rng.choice([-1.0, 1.0]) * rng.uniform(*coef_range),
            "quadratic": (rng.choice([-1.0, 1.0]) * rng.uniform(*quad_coef_range)
                          if mechanism == "quadratic" else 0.0),
            "cubic": (rng.choice([-1.0, 1.0]) * rng.uniform(*cubic_coef_range)
                      if mechanism == "cubic" else 0.0),
        }

    data = rng.normal(0, noise_std, size=(n_samples, num_vars))
    for node in order:
        for p in parents[node]:
            coef = coefficients[(p, node)]
            data[:, node] += (coef["linear"] * data[:, p] + coef["quadratic"] * data[:, p] ** 2
                              + coef["cubic"] * data[:, p] ** 3)
    return data, coefficients


def sample_mixture(edges1, edges2, num_vars, n_each, mechanism="linear", rng=None, **kwargs):
    """Sample a dataset that is a 50/50 mixture of two DAGs' data, for CGP testing.

    :return: (merged, subset1, subset2) -- merged is the concatenation, in the
        same row order convention used by data/synthetic/*/merged.dat.
    """
    merged, subsets = sample_k_mixture([edges1, edges2], num_vars, n_each, mechanism, rng=rng, **kwargs)
    return merged, subsets[0], subsets[1]


def sample_k_mixture(edges_list, num_vars, n_each, mechanism="linear", rng=None, **kwargs):
    """Sample a dataset that is an equal-size mixture of K DAGs' data.

    Generalizes sample_mixture to any number of latent components -- for
    testing whether CGP/EM-style recovery extends past the 2-graph case.

    :param edges_list: list of K edge lists, one per latent component.
    :return: (merged, subsets) -- subsets is the list of K per-component
        arrays, in the same order as edges_list; merged is their concatenation.
    """
    rng = rng if rng is not None else np.random.default_rng()
    subsets = [sample_dag_data(edges, num_vars, n_each, mechanism, rng=rng, **kwargs)[0]
               for edges in edges_list]
    merged = np.vstack(subsets)
    return merged, subsets
