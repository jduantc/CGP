#!/usr/bin/env python
"""Enlarged benchmark: heuristic CGP, feature-space CGP, and mixture EM,
across ALL 55 candidate graph pairs and three mechanisms (linear, quadratic,
cubic), 5 seeds each -- 825 datasets total.

This supersedes the pilot in scripts/run_benchmark_suite.py (16 sampled
pairs x 2 mechanisms x 2 seeds = 64 datasets), which was thin enough that
per-cell accuracy estimates had wide uncertainty (some cells had as few as
n=2). Unlike that pilot, this script does NOT save each generated dataset's
raw points to disk -- 825 datasets at ~200KB each would add ~170MB of
regenerable data to the repository, which isn't worthwhile when the
pilot's 64 datasets already serve as the fixed, inspectable sample and
generation here is fully deterministic (same seeding scheme). Writes
progressively to results/benchmark_suite_full.csv so partial progress
survives an interruption.
"""

import csv
import sys
import time
from functools import partial
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cgp.datagen import sample_mixture, structural_distance
from cgp.feature_clustering import generate_feature_space_clusters
from cgp.mixture import search_best_graph_set
from cgp.probing import THREE_NODE_EQUIVALENCE_CLASSES as CLASSES
from cgp.probing import optimize_graphs

TRUE_PAIRS = list(combinations(range(len(CLASSES)), 2))  # all 55
MECHANISMS = ["linear", "quadratic", "cubic"]
SEEDS = [0, 1, 2, 3, 4]
N_EACH = 600

CGP_PARAMS = dict(nclusters=20, subset_size=300, nsubsets=300, ntrials=15)
EM_PARAMS = dict(n_iter=60, n_init=2)
FC_K_NEIGHBORS = 60

OUT_PATH = Path(__file__).resolve().parent.parent / "results" / "benchmark_suite_full.csv"
FIELDNAMES = ["true_i", "true_j", "distance", "mechanism", "seed",
              "cgp_rank", "cgp_top1_correct", "fc_rank", "fc_top1_correct",
              "em_rank", "em_top1_correct"]


def run_one(true_i, true_j, mechanism, seed):
    rng = np.random.default_rng(seed * 1000 + true_i * 11 + true_j)
    merged, _, _ = sample_mixture(CLASSES[true_i], CLASSES[true_j], num_vars=3,
                                   n_each=N_EACH, mechanism=mechanism, rng=rng)

    fars, _ = optimize_graphs(merged, candidate_graphs=CLASSES, rng=np.random.default_rng(seed), **CGP_PARAMS)
    flat = sorted((fars[a, b], a, b) for a in range(len(CLASSES)) for b in range(a))
    cgp_rank = next(k for k, (_, a, b) in enumerate(flat, 1) if {a, b} == {true_i, true_j})
    cgp_top1 = set(flat[0][1:]) == {true_i, true_j}

    fc_fn = partial(generate_feature_space_clusters, k_neighbors=FC_K_NEIGHBORS)
    fars_fc, _ = optimize_graphs(merged, candidate_graphs=CLASSES, rng=np.random.default_rng(seed),
                                  cluster_fn=fc_fn, **CGP_PARAMS)
    flat_fc = sorted((fars_fc[a, b], a, b) for a in range(len(CLASSES)) for b in range(a))
    fc_rank = next(k for k, (_, a, b) in enumerate(flat_fc, 1) if {a, b} == {true_i, true_j})
    fc_top1 = set(flat_fc[0][1:]) == {true_i, true_j}

    em_results = search_best_graph_set(merged, CLASSES, set_size=2, rng=np.random.default_rng(seed), **EM_PARAMS)
    em_rank = next(k for k, (_, idxs, _) in enumerate(em_results, 1) if set(idxs) == {true_i, true_j})
    em_top1 = set(em_results[0][1]) == {true_i, true_j}

    return {
        "true_i": true_i, "true_j": true_j,
        "distance": structural_distance(CLASSES[true_i], CLASSES[true_j]),
        "mechanism": mechanism, "seed": seed,
        "cgp_rank": cgp_rank, "cgp_top1_correct": int(cgp_top1),
        "fc_rank": fc_rank, "fc_top1_correct": int(fc_top1),
        "em_rank": em_rank, "em_top1_correct": int(em_top1),
    }


def main():
    total = len(TRUE_PAIRS) * len(MECHANISMS) * len(SEEDS)
    print(f"Full benchmark suite: {len(TRUE_PAIRS)} true pairs (all C(11,2)) x "
          f"{len(MECHANISMS)} mechanisms x {len(SEEDS)} seeds = {total} datasets")

    t_start = time.perf_counter()
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        n_done = 0
        for true_i, true_j in TRUE_PAIRS:
            for mechanism in MECHANISMS:
                for seed in SEEDS:
                    row = run_one(true_i, true_j, mechanism, seed)
                    writer.writerow(row)
                    f.flush()
                    n_done += 1
                    elapsed = time.perf_counter() - t_start
                    eta = elapsed / n_done * (total - n_done)
                    print(f"  [{n_done}/{total}] dist={row['distance']} {mechanism:9s} seed={seed}  "
                          f"cgp={row['cgp_rank']:2d} fc={row['fc_rank']:2d} em={row['em_rank']:2d}  "
                          f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

    print(f"\ntotal time: {time.perf_counter() - t_start:.1f}s, wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
