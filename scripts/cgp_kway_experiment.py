#!/usr/bin/env python
"""Actually run CGP-proper (the AR + cluster-affinity method, NOT EM) on the
two problems I'd previously only tested with EM:
  (1) K>2 mixture of 3-node graphs
  (2) 2-component mixture of >3-node graphs

CGP-proper is pairwise-only in cgp/probing.py (optimize_graphs). This
script implements the K-way generalization I'd claimed was "mechanical" but
had never run, reusing CGP's actual machinery: k-means clusters, the
affinity/swing table (cgp.probing.affinity_table), and Averaged Residuals
of the partition. Assignment is K-way argmin over per-graph-CENTERED swings
(centering removes each candidate graph's baseline AR level, generalizing
the relative swing1-vs-swing2 comparison the pairwise method uses; without
it, denser candidate graphs with systematically lower swings would win
every cluster). FAR of a K-tuple = sum over components of AR(graph_k,
points of clusters assigned to k). Lower is better; a tuple leaving any
component empty is skipped.
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cgp.datagen import sample_k_mixture
from cgp.probing import THREE_NODE_EQUIVALENCE_CLASSES as C
from cgp.probing import affinity_table, generate_kmeans_clusters, optimize_graphs
from cgp.scoring import averaged_residuals


def cgp_kway(data, candidate_graphs, set_size, nclusters, subset_size, nsubsets, rng, center=True):
    clusters = generate_kmeans_clusters(data, nclusters)
    table = affinity_table(clusters, candidate_graphs, subset_size, nsubsets, rng)  # (nclusters, ncand)
    table_use = table - table.mean(axis=0, keepdims=True) if center else table

    results = []
    for combo in combinations(range(len(candidate_graphs)), set_size):
        assign = np.argmin(table_use[:, list(combo)], axis=1)
        far, valid = 0.0, True
        for k in range(set_size):
            members = [clusters[c] for c in range(len(clusters)) if assign[c] == k]
            if not members:
                valid = False
                break
            far += averaged_residuals(candidate_graphs[combo[k]], np.vstack(members))
        if valid:
            results.append((far, combo))
    results.sort()
    return results


def rank_of(results, true_combo):
    target = frozenset(true_combo)
    for r, (_, combo) in enumerate(results, 1):
        if frozenset(combo) == target:
            return r
    return None


def experiment_1_kway_3node():
    print("=" * 64)
    print("EXPERIMENT 1: CGP-proper on a K=3 mixture of 3-node graphs")
    print("(the exact setup EM found at rank 1/165 -- can CGP-proper too?)")
    true_idx = (0, 4, 10)  # empty, [(0,1),(0,2)], complete -- well-separated
    n_combos = len(list(combinations(range(len(C)), 3)))
    for run in range(3):
        rng = np.random.default_rng(run)
        merged, _ = sample_k_mixture([C[i] for i in true_idx], num_vars=3, n_each=500,
                                      mechanism="linear", rng=rng)
        results = cgp_kway(merged, C, set_size=3, nclusters=30, subset_size=300, nsubsets=300,
                           rng=np.random.default_rng(run))
        r = rank_of(results, true_idx)
        r_str = f"{r}/{n_combos}" if r else "NOT FOUND (a true component got no clusters)"
        print(f"  run {run}: true triple rank {r_str:42s} top={results[0][1]}   (EM: 1/165)")


def experiment_2_2graph_4node():
    print("=" * 64)
    print("EXPERIMENT 2: CGP-proper on a K=2 mixture of 4-NODE graphs")
    print("(oracle candidate set: the 2 true graphs + distractors -- isolates")
    print(" whether CGP's affinity+FAR mechanism works at 4 nodes, separate")
    print(" from the candidate-generation problem)")
    g_true1 = [(0, 1), (0, 2), (1, 3), (2, 3)]   # diamond
    g_true2 = [(3, 2), (3, 1), (2, 0), (1, 0)]   # reversed-ish, different skeleton dir
    distractors = [
        [(0, 1), (1, 2), (2, 3)],                # chain
        [(0, 1), (0, 2), (0, 3)],                # star out of 0
        [(3, 0), (3, 1), (3, 2)],                # star into 3
        [(0, 2), (1, 2), (2, 3)],                # collider-ish
    ]
    candidates = [g_true1, g_true2] + distractors
    true_idx = (0, 1)
    n_combos = len(list(combinations(range(len(candidates)), 2)))
    for run in range(3):
        rng = np.random.default_rng(run)
        merged, _ = sample_k_mixture([g_true1, g_true2], num_vars=4, n_each=800,
                                      mechanism="linear", rng=rng)
        results = cgp_kway(merged, candidates, set_size=2, nclusters=30, subset_size=300, nsubsets=300,
                           rng=np.random.default_rng(run))
        r = rank_of(results, true_idx)
        r_str = f"{r}/{n_combos}" if r else "NOT FOUND"
        print(f"  run {run}: true pair rank {r_str:10s}  top={results[0][1]}")


def _real_cgp_rank(fars, true_idx):
    n = fars.shape[0]
    flat = sorted((fars[a, b], a, b) for a in range(n) for b in range(a))
    tgt = frozenset(true_idx)
    for r, (_, a, b) in enumerate(flat, 1):
        if frozenset((a, b)) == tgt:
            return r
    return None


def control_k2_3node():
    """Control: my k-way machinery (set_size=2) vs. the REAL optimize_graphs on
    the SAME data. If my k-way rank differs a lot from the real method's, the
    problem is my unfaithful reimplementation, not CGP's ability -- so K>2
    results from it would be uninterpretable. Tested on an EASY pair (where
    real CGP is known to do well) and a HARD distance-1 pair (where the
    benchmark showed real CGP is weak anyway)."""
    print("=" * 64)
    print("CONTROL: my k-way (set_size=2) vs. REAL optimize_graphs, same data")
    pairs = {"easy (1 vs 7)": (1, 7), "hard/dist-1 (7 vs 10)": (7, 10)}
    for label, true_idx in pairs.items():
        rng = np.random.default_rng(0)
        merged, _ = sample_k_mixture([C[true_idx[0]], C[true_idx[1]]], num_vars=3,
                                      n_each=500, mechanism="linear", rng=rng)
        fars, _ = optimize_graphs(merged, nclusters=30, subset_size=300, nsubsets=300, ntrials=15,
                                  rng=np.random.default_rng(0))
        real_r = _real_cgp_rank(fars, true_idx)
        mine = cgp_kway(merged, C, set_size=2, nclusters=30, subset_size=300, nsubsets=300,
                        rng=np.random.default_rng(0))
        my_r = rank_of(mine, true_idx)
        print(f"  {label:24s}: REAL optimize_graphs rank {real_r}/55   my k-way rank {my_r}/55")


def diagnose_assignment(true_idx=(0, 4, 10)):
    """Why did the K=3 true triple come back as rank None? Check whether the
    argmin assignment ever leaves one of the three true components empty."""
    print("=" * 64)
    print(f"DIAGNOSTIC: cluster assignment for the true triple {true_idx}")
    rng = np.random.default_rng(0)
    merged, _ = sample_k_mixture([C[i] for i in true_idx], num_vars=3, n_each=500,
                                  mechanism="linear", rng=rng)
    clusters = generate_kmeans_clusters(merged, 30)
    table = affinity_table(clusters, C, subset_size=300, nsubsets=300, rng=np.random.default_rng(0))
    for center in (True, False):
        t = table - table.mean(axis=0, keepdims=True) if center else table
        assign = np.argmin(t[:, list(true_idx)], axis=1)
        counts = [int((assign == k).sum()) for k in range(len(true_idx))]
        print(f"  center={center}: clusters assigned to each true component = {counts}")


if __name__ == "__main__":
    control_k2_3node()
    print()
    diagnose_assignment()
    print()
    experiment_1_kway_3node()
    print()
    experiment_2_2graph_4node()
