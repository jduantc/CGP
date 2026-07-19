#!/usr/bin/env python
"""Code to test the ability of nonlinear-AR CGP to recover the true graph pair
(better than linear-AR CGP and EM)

All three use the same data. CGP-linear and CGP-nonlinear use the Same
clustering + affinity-swing + Final-AR partition machinery (validated
faithful in cgp_kway_experiment.py); the only difference is the regression
degree inside the AR (degree 1 = linear = cgp.covstats; degree>1 =
polynomial, cgp.nonlinear). So any gap is attributable purely to the
linear-vs-nonlinear fit, nothing else.

Uses well-separated true pairs (structural distance >= 3) and strong
nonlinear coeffs to test in cases where identifiability is not the confound.
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cgp.datagen import sample_k_mixture
from cgp.mixture import search_best_graph_set
from cgp.nonlinear import augment, nl_ar, nl_ar_batch
from cgp.probing import THREE_NODE_EQUIVALENCE_CLASSES as C
from cgp.probing import _draw_disjoint_subsets, generate_kmeans_clusters
from cgp.scoring import edgelist_to_matrix


def nl_affinity_table(clusters, dag_mats, degree, subset_size, nsubsets, rng):
    """Swing of every cluster toward every candidate DAG, using degree-`degree`
    AR -- the nonlinear analog of cgp.probing.affinity_table."""
    numvars = clusters[0].shape[1]
    table = np.empty((len(clusters), len(dag_mats)))
    for i in range(len(clusters)):
        background = np.vstack([clusters[j] for j in range(len(clusters)) if j != i])
        caug = augment(clusters[i], degree)
        cn, cs, cQ = caug.shape[0], caug.sum(0), caug.T @ caug
        baug = augment(background, degree)
        idx = _draw_disjoint_subsets(baug.shape[0], subset_size, nsubsets, rng)
        pts = baug[idx]
        s, Q = pts.sum(1), np.matmul(pts.transpose(0, 2, 1), pts)
        for k, dag in enumerate(dag_mats):
            with_c = nl_ar_batch(dag, degree, subset_size + cn, s + cs, Q + cQ)
            without = nl_ar_batch(dag, degree, subset_size, s, Q)
            table[i, k] = (with_c - without).mean()
    return table


def cgp_pair_ranking(merged, candidates, degree, nclusters, subset_size, nsubsets, rng):
    """Rank all candidate PAIRS by Final-AR of the partition their affinities
    induce, using degree-`degree` AR. Returns sorted [(FAR, (i,j)), ...]."""
    dag_mats = [edgelist_to_matrix(c, merged.shape[1]) for c in candidates]
    clusters = generate_kmeans_clusters(merged, nclusters)
    table = nl_affinity_table(clusters, dag_mats, degree, subset_size, nsubsets, rng)
    table_c = table - table.mean(axis=0, keepdims=True)   # per-candidate baseline removal

    results = []
    for i, j in combinations(range(len(candidates)), 2):
        assign = np.argmin(table_c[:, [i, j]], axis=1)
        far, valid = 0.0, True
        for k, cand in enumerate((i, j)):
            members = [clusters[c] for c in range(len(clusters)) if assign[c] == k]
            if not members:
                valid = False
                break
            far += nl_ar(dag_mats[cand], np.vstack(members), degree)
        if valid:
            results.append((far, (i, j)))
    results.sort()
    return results


def rank_of(results, true_pair):
    tgt = frozenset(true_pair)
    for r, (_, combo) in enumerate(results, 1):
        if frozenset(combo) == tgt:
            return r
    return None


def em_rank(merged, true_pair, rng):
    try:
        res = search_best_graph_set(merged, C, set_size=2, n_iter=60, n_init=2, rng=rng)
    except np.linalg.LinAlgError:
        return None  # EM's linear-Gaussian covariance stayed non-PD even after jitter
    tgt = frozenset(true_pair)
    for r, (_, idxs, _) in enumerate(res, 1):
        if frozenset(idxs) == tgt:
            return r
    return None


def main():
    true_pairs = [(1, 7), (0, 10), (6, 9)]  # all structural distance >= 3, well-separated
    configs = [("quadratic", 2, {"quad_coef_range": (2.0, 3.5)}),
               ("cubic", 3, {"cubic_coef_range": (0.5, 1.2)})]
    n_seeds = 3
    P = dict(nclusters=25, subset_size=300, nsubsets=200)

    for mechanism, degree, kw in configs:
        print(f"\n{'='*70}\nMechanism: {mechanism} (strong coefs) -- true-pair rank / 55 (lower better)")
        print(f"{'true pair':12s} {'CGP-linear':>12s} {'CGP-nonlinear':>14s} {'EM (linear-G)':>14s}")
        for tp in true_pairs:
            lin, nl, em = [], [], []
            for seed in range(n_seeds):
                merged, _ = sample_k_mixture([C[tp[0]], C[tp[1]]], num_vars=3, n_each=800,
                                             mechanism=mechanism, rng=np.random.default_rng(seed), **kw)
                lin.append(rank_of(cgp_pair_ranking(merged, C, 1, rng=np.random.default_rng(seed), **P), tp))
                nl.append(rank_of(cgp_pair_ranking(merged, C, degree, rng=np.random.default_rng(seed), **P), tp))
                em.append(em_rank(merged, tp, np.random.default_rng(seed)))
            fmt = lambda xs: "[" + ",".join(str(x) if x else "-" for x in xs) + "]"
            print(f"{str(tp):12s} {fmt(lin):>12s} {fmt(nl):>14s} {fmt(em):>14s}")


if __name__ == "__main__":
    main()
