#!/usr/bin/env python
"""Feasibility test for generalizing CGP past hand-enumeration: can greedy
BIC-scored DAG discovery (cgp/discovery.py) recover known 4- and 5-node
structures, linear and nonlinear? This is the load-bearing claim behind
"replace the enumerated equivalence-class list with discovery."

Primary metric is SKELETON recovery (undirected edges): BIC is
score-equivalent, so it identifies a DAG only up to its Markov equivalence
class -- edge *orientations* within a class aren't distinguishable from
observational data alone, but the skeleton is. Exact-orientation recovery
is reported as a secondary, harder check.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cgp.datagen import sample_dag_data, sample_k_mixture
from cgp.discovery import candidate_pool, greedy_dag_search


def skeleton(edges, d):
    return {frozenset((p, c)) for p, c in edges}


def skeleton_from_adj(dag):
    return {frozenset((i, j)) for i, j in zip(*np.where(dag == 1))}


def f1(true_skel, pred_skel):
    if not pred_skel and not true_skel:
        return 1.0
    tp = len(true_skel & pred_skel)
    prec = tp / len(pred_skel) if pred_skel else 0.0
    rec = tp / len(true_skel) if true_skel else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


CASES = {
    "4-node diamond": (4, [(0, 1), (0, 2), (1, 3), (2, 3)]),
    "4-node chain+fork": (4, [(0, 1), (1, 2), (1, 3)]),
    "5-node": (5, [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]),
}


def run(mechanism, degree):
    print(f"\n{'='*60}\nMechanism: {mechanism} (search degree={degree})")
    for name, (d, edges) in CASES.items():
        true_skel = skeleton(edges, d)
        kw = {"quad_coef_range": (0.5, 2.0)} if mechanism == "quadratic" else {}
        skel_f1s, exact_hits = [], 0
        for seed in range(5):
            rng = np.random.default_rng(seed)
            data, _ = sample_dag_data(edges, d, 3000, mechanism=mechanism, rng=rng, **kw)
            found = greedy_dag_search(data, degree=degree)
            skel_f1s.append(f1(true_skel, skeleton_from_adj(found)))
            exact_hits += set(zip(*np.where(found == 1))) == set(edges)
        print(f"  {name:18s}: skeleton F1 = {np.mean(skel_f1s):.2f}  "
              f"exact-orientation recovery = {exact_hits}/5")


def main():
    run("linear", degree=1)
    run("quadratic", degree=2)

    print(f"\n{'='*60}\nCandidate-pool demo 1: SINGLE 4-node graph (should converge to one structure)")
    rng = np.random.default_rng(0)
    data, _ = sample_dag_data([(0, 1), (0, 2), (1, 3), (2, 3)], 4, 3000, mechanism="linear", rng=rng)
    for edges, count in candidate_pool(data, n_bootstraps=15, degree=1, rng=rng):
        print(f"  {count:2d}/15 bootstraps: {edges}")

    print(f"\nCandidate-pool demo 2: 4-node MIXTURE of two graphs")
    print("does whole-dataset bootstrapping surface BOTH true structures, or just a blend?")
    g1 = [(0, 1), (1, 2), (2, 3)]        # chain
    g2 = [(3, 2), (2, 1), (1, 0)]        # reversed chain -- opposite orientation
    print(f"  true g1={g1}")
    print(f"  true g2={g2}")
    rng = np.random.default_rng(1)
    merged, _ = sample_k_mixture([g1, g2], num_vars=4, n_each=1500, mechanism="linear", rng=rng)
    for edges, count in candidate_pool(merged, n_bootstraps=15, degree=1, rng=rng):
        print(f"  {count:2d}/15 bootstraps: {edges}")


if __name__ == "__main__":
    main()
