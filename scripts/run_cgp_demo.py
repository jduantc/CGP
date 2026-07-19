#!/usr/bin/env python
"""Run Causal Graph Probing on a dataset and report the best-fitting 2-graph partition.

Probe a mixture dataset for the pair of DAGs that best partitions it, and
(optionally) score that partition against a known ground-truth split.

Example (the "hard" pair, data/synthetic/hard_pair/ -- two DAGs differing by a
single edge):

    python scripts/run_cgp_demo.py data/synthetic/hard_pair/merged.dat \\
        --true-subset1 data/synthetic/hard_pair/true_subset1.dat \\
        --true-subset2 data/synthetic/hard_pair/true_subset2.dat
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cgp.probing import THREE_NODE_EQUIVALENCE_CLASSES, evaluate_accuracy, optimize_graphs


def load_file(path):
    return np.loadtxt(path, skiprows=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="Path to the (possibly mixed) dataset to probe.")
    parser.add_argument("--true-subset1", help="Optional: known ground-truth subset for DAG 1, to score accuracy.")
    parser.add_argument("--true-subset2", help="Optional: known ground-truth subset for DAG 2, to score accuracy.")
    parser.add_argument("--nclusters", type=int, default=40)
    parser.add_argument("--subset-size", type=int, default=1000)
    parser.add_argument("--nsubsets", type=int, default=2000)
    parser.add_argument("--ntrials", type=int, default=40)
    args = parser.parse_args()

    data = load_file(args.data)

    fars, (best_dag1, best_dag2) = optimize_graphs(
        data, args.nclusters, args.subset_size, args.nsubsets, args.ntrials,
        candidate_graphs=THREE_NODE_EQUIVALENCE_CLASSES if data.shape[1] == 3 else None,
    )
    print(f"Best DAG pair (lowest FAR = {fars.min():.2f}):")
    print(f"  DAG 1: {best_dag1}")
    print(f"  DAG 2: {best_dag2}")

    if args.true_subset1 and args.true_subset2:
        true1 = load_file(args.true_subset1)
        true2 = load_file(args.true_subset2)
        dag1_allocation, dag2_allocation = evaluate_accuracy(
            best_dag1, best_dag2, data, true1, true2,
            args.nclusters, args.subset_size, args.nsubsets,
        )
        print(f"True DAG 1 allocation (correct, neutral, incorrect): {dag1_allocation}")
        print(f"True DAG 2 allocation (correct, neutral, incorrect): {dag2_allocation}")


if __name__ == "__main__":
    main()
