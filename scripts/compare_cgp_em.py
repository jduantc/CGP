#!/usr/bin/env python
"""Head-to-head comparison of the two mixture-of-DAGs recovery methods in this
repo, on linear and nonlinear-mechanism mixtures:

  1. CGP + (nonlinear) Averaged Residuals -- the paper's method: cluster,
     estimate each cluster's affinity toward each candidate DAG, partition, and
     rank candidate DAG PAIRS by combined AR. Searches all 55 pairs of the 11
     three-node Markov-equivalence classes. See cgp/probing.py, cgp/nonlinear.py.
  2. Mixture-of-SEMs EM -- the established (Thiesson et al. 1998) baseline: fit
     a Gaussian mixture whose per-component covariance is DAG-constrained, rank
     pairs by BIC. See cgp/mixture.py.

Two metrics, both reported:

  - GRAPH RECOVERY: the RANK of the true pair out of 55 (did the search find
    it?), and the structural distance (SHD; edge-set symmetric difference,
    minimized over component permutation) of the TOP-1 pair to the true pair
    (0 = exact recovery).
  - PARTITION ACCURACY: fraction of points assigned to the correct component
    (best of the two label permutations).

Both methods search all 55 candidate pairs. Torch-free (numpy/scipy/sklearn).
"""
from itertools import combinations, permutations
from pathlib import Path
import sys

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cgp.datagen import sample_k_mixture
from cgp.mixture import search_best_graph_set
from cgp.nonlinear import augment, nl_ar, nl_ar_batch
from cgp.probing import THREE_NODE_EQUIVALENCE_CLASSES as CANDIDATES
from cgp.scoring import edgelist_to_matrix

DEFAULT_PARAMS = dict(nclusters=25, subset_size=300, nsubsets=200)


def _draw_disjoint_subsets(nrows, subset_size, nsubsets, rng):
    return np.argsort(rng.random((nsubsets, nrows)), axis=1)[:, :subset_size]


def _edge_set(edges):
    return frozenset((int(a), int(b)) for a, b in edges)


def structural_distance(pred_edges_list, true_edges_list):
    """Min over component permutation of summed edge-set symmetric differences
    (a simple SHD that counts a reversed edge as distance 2). 0 = exact."""
    tv = [_edge_set(e) for e in true_edges_list]
    pv = [_edge_set(e) for e in pred_edges_list]
    return min(sum(len(pv[perm[k]] ^ tv[k]) for k in range(len(tv)))
               for perm in permutations(range(len(tv))))


def _assignment_accuracy(labels, true_labels):
    return max(np.mean(labels == true_labels), np.mean(labels == (1 - true_labels)))


def cgp_recover(merged, true_pair, true_labels, degree=2, candidates=CANDIDATES,
                rng=None, params=DEFAULT_PARAMS):
    """CGP nonlinear-AR search over all candidate pairs. Returns dict with the
    true pair's rank, the top-1 pair's structural distance to truth, and the
    top-1 pair's induced partition accuracy."""
    rng = rng if rng is not None else np.random.default_rng()
    num_vars = merged.shape[1]
    km = KMeans(n_clusters=params["nclusters"], n_init=10).fit(merged)
    clab = km.labels_
    clusters = [merged[clab == i] for i in range(params["nclusters"])]
    mats = [edgelist_to_matrix(c, num_vars) for c in candidates]

    # swing of every cluster toward every candidate DAG (shared subset draws)
    table = np.empty((len(clusters), len(candidates)))
    for i in range(len(clusters)):
        background = np.vstack([clusters[j] for j in range(len(clusters)) if j != i])
        caug = augment(clusters[i], degree)
        cn, cs, cQ = caug.shape[0], caug.sum(0), caug.T @ caug
        baug = augment(background, degree)
        idx = _draw_disjoint_subsets(baug.shape[0], params["subset_size"], params["nsubsets"], rng)
        pts = baug[idx]
        s, Q = pts.sum(1), np.matmul(pts.transpose(0, 2, 1), pts)
        for k, dag in enumerate(mats):
            with_c = nl_ar_batch(dag, degree, params["subset_size"] + cn, s + cs, Q + cQ)
            without = nl_ar_batch(dag, degree, params["subset_size"], s, Q)
            table[i, k] = (with_c - without).mean()
    table_c = table - table.mean(axis=0, keepdims=True)  # per-candidate baseline

    ranking = []
    for i, j in combinations(range(len(candidates)), 2):
        assign = np.argmin(table_c[:, [i, j]], axis=1)
        far, valid = 0.0, True
        for k, cand in enumerate((i, j)):
            members = [clusters[c] for c in range(len(clusters)) if assign[c] == k]
            if not members:
                valid = False
                break
            far += nl_ar(mats[cand], np.vstack(members), degree)
        if valid:
            ranking.append((far, (i, j)))
    ranking.sort()

    tgt = frozenset(true_pair)
    rank = next((r for r, (_, ij) in enumerate(ranking, 1) if frozenset(ij) == tgt), None)
    top1 = ranking[0][1]
    shd = structural_distance([candidates[top1[0]], candidates[top1[1]]],
                              [candidates[true_pair[0]], candidates[true_pair[1]]])
    assign = np.argmin(table_c[:, list(top1)], axis=1)
    point_labels = assign[clab]
    pacc = _assignment_accuracy(point_labels, true_labels)
    return {"rank": rank, "shd": shd, "partition_acc": pacc, "top1_pair": top1,
            "labels": point_labels}


def em_recover(merged, true_pair, true_labels, candidates=CANDIDATES, rng=None):
    """Mixture-EM (BIC) search over all candidate pairs."""
    rng = rng if rng is not None else np.random.default_rng()
    try:
        res = search_best_graph_set(merged, candidates, set_size=2, n_iter=60, n_init=2, rng=rng)
    except np.linalg.LinAlgError:
        return {"rank": None, "shd": None, "partition_acc": np.nan, "top1_pair": None,
                "labels": None}
    tgt = frozenset(true_pair)
    rank = next((r for r, (_, ij, _) in enumerate(res, 1) if frozenset(ij) == tgt), None)
    top1_idx, top1_fit = res[0][1], res[0][2]
    shd = structural_distance([candidates[top1_idx[0]], candidates[top1_idx[1]]],
                              [candidates[true_pair[0]], candidates[true_pair[1]]])
    point_labels = np.argmax(top1_fit["responsibilities"], axis=1)
    pacc = _assignment_accuracy(point_labels, true_labels)
    return {"rank": rank, "shd": shd, "partition_acc": pacc, "top1_pair": top1_idx,
            "labels": point_labels}


# The nonlinearity gradient (quadratic-coefficient strength) and the pairs used
# in results/cgp_vs_em_comparison.csv.
NONLINEARITY_LEVELS = {"linear": (0.0, 0.0), "moderate": (1.0, 1.5), "strong": (2.0, 3.5)}
PAIRS = {"(1,7) separated": (1, 7), "(6,9) separated": (6, 9),
         "(0,10) empty-vs-full": (0, 10), "(7,10) nested dist-1": (7, 10)}


def run_comparison(pairs=PAIRS, levels=NONLINEARITY_LEVELS, seeds=range(5),
                   n_each=800, degree=2, verbose=True):
    """Run the full sweep; return a list of per-cell result dicts."""
    rows = []
    for pname, pair in pairs.items():
        for lname, qr in levels.items():
            for seed in seeds:
                merged, _ = sample_k_mixture([CANDIDATES[pair[0]], CANDIDATES[pair[1]]],
                                             num_vars=3, n_each=n_each, mechanism="quadratic",
                                             rng=np.random.default_rng(seed), quad_coef_range=qr)
                tl = np.array([0] * n_each + [1] * n_each)
                cgp = cgp_recover(merged, pair, tl, degree, rng=np.random.default_rng(seed))
                em = em_recover(merged, pair, tl, rng=np.random.default_rng(seed))
                rows.append(dict(pair=pname, level=lname, seed=seed,
                                 cgp_rank=cgp["rank"], cgp_shd=cgp["shd"], cgp_pacc=cgp["partition_acc"],
                                 em_rank=em["rank"], em_shd=em["shd"], em_pacc=em["partition_acc"]))
                if verbose:
                    print(f"DONE {pname:22s} {lname:9s} s{seed}  "
                          f"CGP(rank={cgp['rank']},shd={cgp['shd']}) "
                          f"EM(rank={em['rank']},shd={em['shd']})", flush=True)
    return rows


def main():
    import csv
    rows = run_comparison()
    out = Path(__file__).resolve().parent.parent / "results" / "cgp_vs_em_comparison.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
