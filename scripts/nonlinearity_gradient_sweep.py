#!/usr/bin/env python
"""Wider, mild->strong nonlinearity sweep: CGP-linear vs CGP-nonlinear vs
linear-Gaussian EM, across a gradient of quadratic-coefficient strength.

Goal: map out exactly where EM starts to break down relative to CGP, and
quantify the CGP-linear-vs-EM and nonlinear-vs-linear-AR gaps as a function
of nonlinearity strength, rather than a single strong/weak snapshot.

Reuses cgp_pair_ranking / rank_of / em_rank from nonlinear_cgp_experiment.py
unchanged -- same pipeline, same fairness guarantees (EM given jitter
regularization; CGP variants share the same clustering/affinity/Final-AR
machinery, differing only in AR polynomial degree).

Nonlinearity strength is reported two ways: (1) the quad_coef_range used to
generate the data, and (2) an intrinsic, verifiable measure -- mean absolute
skewness of the marginals under that coefficient range -- so "how nonlinear"
is not just a label but an empirically grounded quantity (see the Shapiro/
skew/kurtosis check that motivated this: linear mechanisms give Gaussian
marginals, nonlinear ones don't).
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import skew

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cgp.datagen import sample_k_mixture
from cgp.probing import THREE_NODE_EQUIVALENCE_CLASSES as C
from scripts.nonlinear_cgp_experiment import cgp_pair_ranking, em_rank, rank_of


LEVELS = [
    ("linear (control)", (0.0, 0.0)),
    ("mild", (0.3, 0.6)),
    ("moderate", (0.8, 1.2)),
    ("strong-ish", (1.5, 2.0)),
    ("strong", (2.0, 3.5)),
]
TRUE_PAIRS = [(1, 7), (0, 10), (6, 9)]
N_SEEDS = 10
P = dict(nclusters=25, subset_size=300, nsubsets=200)
DEGREE = 2  # quadratic nonlinear-AR


def mean_abs_skew(data):
    return float(np.mean(np.abs(skew(data, axis=0))))


def main():
    rows = []
    t_start = time.time()
    n_total = len(LEVELS) * len(TRUE_PAIRS) * N_SEEDS
    n_done = 0

    for level_name, coef_range in LEVELS:
        for tp in TRUE_PAIRS:
            for seed in range(N_SEEDS):
                rng_data = np.random.default_rng(1000 * seed + tp[0] * 10 + tp[1])
                merged, _ = sample_k_mixture(
                    [C[tp[0]], C[tp[1]]], num_vars=3, n_each=800,
                    mechanism="quadratic", rng=rng_data, quad_coef_range=coef_range,
                )
                strength = mean_abs_skew(merged)

                lin_rank = rank_of(
                    cgp_pair_ranking(merged, C, 1, rng=np.random.default_rng(seed), **P), tp)
                nl_rank = rank_of(
                    cgp_pair_ranking(merged, C, DEGREE, rng=np.random.default_rng(seed), **P), tp)
                em_r = em_rank(merged, tp, np.random.default_rng(seed))

                rows.append(dict(
                    level=level_name, coef_lo=coef_range[0], coef_hi=coef_range[1],
                    true_pair=f"{tp[0]}-{tp[1]}", seed=seed, mean_abs_skew=strength,
                    cgp_linear_rank=lin_rank, cgp_nonlinear_rank=nl_rank, em_rank=em_r,
                ))
                n_done += 1
                elapsed = time.time() - t_start
                eta = elapsed / n_done * (n_total - n_done)
                print(f"[{n_done}/{n_total}] level={level_name:18s} pair={tp} seed={seed} "
                      f"skew={strength:.2f} lin={lin_rank} nl={nl_rank} em={em_r} "
                      f"(elapsed {elapsed:.0f}s, eta {eta:.0f}s)", flush=True)

    out_path = Path(__file__).resolve().parent.parent / "results" / "nonlinearity_gradient_sweep.csv"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
