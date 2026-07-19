"""Thin wrapper around fges-py: run FGES search and cross-validate AR/BIC.

Requires fges-py to be installed / on PYTHONPATH (see external/SETUP.md) —
it is a dependency of this project, not vendored here.
"""

import pickle

import numpy as np

from cgp.scoring import averaged_residuals, edgelist_to_matrix
from cgp.viz import draw_dag, generic_var_labels

try:
    from fges import FGES
    from SEMScore import SEMBicScore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "fges-py is required but not importable. See external/SETUP.md "
        "for installation instructions."
    ) from exc


def load_file(data_file):
    return np.loadtxt(data_file, skiprows=1)


def run_fges(dataset, sparsity, save_path=None, checkpoint_frequency=0, plot=False):
    """Run FGES on a dataset and return (edges, BIC score).

    :param save_path: if given, pickle the raw FGES result to this path
        (e.g. "results/run1.pkl").
    """
    score = SEMBicScore(sparsity, dataset=dataset)
    variables = list(range(dataset.shape[1]))
    fges = FGES(variables, score, filename=dataset, checkpoint_frequency=checkpoint_frequency,
                save_name=str(save_path) if save_path else None)
    result = fges.search()
    edges = result["graph"].edges()
    final_score = result["score"]

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

    if plot:
        var_names = generic_var_labels(dataset.shape[1])
        labels = dict(enumerate(var_names))
        draw_dag(edges, labels=labels)

    return edges, final_score


def cross_validate_metrics(data, sparsity, subset_size, nsubsets, var_names=None):
    """Bootstrap-subsample `data`, run FGES on each subset, and average AR and BIC."""
    var_names = var_names or generic_var_labels(data.shape[1])
    nrows = data.shape[0]
    ars, bics = [], []
    for _ in range(nsubsets):
        rand_row_indices = np.random.choice(range(nrows), size=subset_size, replace=False)
        subset = data[rand_row_indices, :]
        edges, bic = run_fges(subset, sparsity)
        bics.append(bic)
        ars.append(averaged_residuals(edgelist_to_matrix(edges, subset.shape[1]), subset, var_names))
    return np.mean(ars), np.mean(bics)
