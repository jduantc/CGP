#!/usr/bin/env python
"""Benchmark the covariance-based AR backend against the reference lstsq implementation.

See docs/notes/covariance_ar.md for the derivation this benchmarks.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cgp.covstats import ar_batch, raw_stats
from cgp.probing import THREE_NODE_EQUIVALENCE_CLASSES, affinity_table
from cgp.scoring import averaged_residuals, edgelist_to_matrix
from cgp.sem import estimate_parameters


def ar_reference(dag, data):
    """The direct per-row lstsq implementation (cgp.sem.estimate_parameters), for comparison."""
    _, residuals = estimate_parameters(dag, data)
    return np.trace(residuals) / data.shape[1]


def check_equivalence(data):
    dags = {
        "empty": np.zeros((3, 3)),
        "chain": np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]),
        "collider": np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]]),
        "complete": np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]),
    }
    rng = np.random.default_rng(0)
    print("Numerical equivalence, covariance AR vs. reference lstsq AR:")
    worst = 0.0
    for name, dag in dags.items():
        for label, d in (("real", data), ("random", rng.standard_normal((500, 3)) * [3, 1, 7])):
            a, b = ar_reference(dag, d), averaged_residuals(dag, d)
            worst = max(worst, abs(a - b))
            print(f"  {name:9s} {label:6s}  reference={a:12.6f}  covariance={b:12.6f}  diff={abs(a - b):.2e}")
    print(f"  worst-case absolute difference: {worst:.2e}\n")


def benchmark_all_dags(data):
    """The realistic optimize_graphs workload: swing of one cluster toward all
    11 equivalence-class DAGs, compared against redoing it per-DAG the slow way."""
    dags = [edgelist_to_matrix(e, 3) for e in THREE_NODE_EQUIVALENCE_CLASSES]
    cluster, background = data[:75], data[75:]
    subset_size, nsubsets = 1000, 400
    rng = np.random.default_rng(7)
    idx = np.array([rng.choice(background.shape[0], size=subset_size, replace=False)
                     for _ in range(nsubsets)])

    t0 = time.perf_counter()
    reference = []
    for dag in dags:
        swings = [ar_reference(dag, np.vstack((background[row], cluster))) - ar_reference(dag, background[row])
                   for row in idx]
        reference.append(np.mean(swings))
    t_reference = time.perf_counter() - t0

    t0 = time.perf_counter()
    cn, cs, cQ = raw_stats(cluster)
    pts = background[idx]
    s, Q = pts.sum(axis=1), np.matmul(pts.transpose(0, 2, 1), pts)
    fast = [(ar_batch(dag, subset_size + cn, s + cs, Q + cQ) - ar_batch(dag, subset_size, s, Q)).mean()
            for dag in dags]
    t_fast = time.perf_counter() - t0

    print(f"Swing toward all 11 DAGs, one cluster, {nsubsets} subsets of size {subset_size}:")
    print(f"  reference : {t_reference:8.3f} s")
    print(f"  covariance: {t_fast:8.3f} s")
    print(f"  speedup   : {t_reference / t_fast:6.1f}x")
    print(f"  max swing disagreement: {max(abs(a - b) for a, b in zip(reference, fast)):.2e}\n")


def benchmark_optimize_graphs(data):
    """End-to-end optimize_graphs affinity_table call, at realistic scale."""
    nclusters, subset_size, nsubsets = 40, 1000, 1000
    rng = np.random.default_rng(0)
    from cgp.probing import generate_kmeans_clusters
    clusters = generate_kmeans_clusters(data, nclusters)

    t0 = time.perf_counter()
    affinity_table(clusters, THREE_NODE_EQUIVALENCE_CLASSES, subset_size, nsubsets, rng)
    t_fast = time.perf_counter() - t0
    print(f"affinity_table: {nclusters} clusters x 11 DAGs x {nsubsets} subsets of size {subset_size}: {t_fast:.2f} s")
    print("(the pre-optimization equivalent -- 55 pairs each redrawing/rescoring independently -- "
          "would be roughly 10x this for the affinity step alone, before even counting FAR trials)")


if __name__ == "__main__":
    data = np.loadtxt("data/synthetic/easy_pair/merged.dat", skiprows=1)
    check_equivalence(data)
    benchmark_all_dags(data)
    benchmark_optimize_graphs(data)
