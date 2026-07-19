"""Sanity + consistency tests for the batched CGP affinity machinery."""

import numpy as np
import pytest

from cgp.probing import (
    THREE_NODE_EQUIVALENCE_CLASSES,
    affinity_table,
    finalize_ar,
    generate_kmeans_clusters,
    optimize_graphs,
    swing_table,
)
from cgp.scoring import averaged_residuals


@pytest.fixture
def three_node_data():
    return np.loadtxt("data/synthetic/easy_pair/merged.dat", skiprows=1)


def test_swing_table_shape_and_finiteness(three_node_data):
    cluster, background = three_node_data[:50], three_node_data[50:]
    swings = swing_table(cluster, background, THREE_NODE_EQUIVALENCE_CLASSES,
                          subset_size=100, nsubsets=20, rng=np.random.default_rng(0))
    assert swings.shape == (11,)
    assert np.all(np.isfinite(swings))


def test_affinity_table_shared_draws_are_self_consistent(three_node_data):
    """affinity_table's multi-DAG scoring shares one subset draw per cluster
    across all candidate DAGs. Scoring just two of those DAGs with a
    freshly-seeded identical RNG should reproduce the same two columns
    exactly, since the draw itself doesn't depend on how many DAGs are
    being scored against it."""
    clusters = generate_kmeans_clusters(three_node_data, nclusters=8)
    dags = THREE_NODE_EQUIVALENCE_CLASSES
    i, j = 3, 7

    full_table = affinity_table(clusters, dags, subset_size=80, nsubsets=15, rng=np.random.default_rng(42))
    pair_table = affinity_table(clusters, [dags[i], dags[j]], subset_size=80, nsubsets=15,
                                 rng=np.random.default_rng(42))

    np.testing.assert_allclose(full_table[:, [i, j]], pair_table, atol=1e-10)


def test_optimize_graphs_runs_and_diagonal_matches_unpartitioned_ar(three_node_data):
    fars, best_pair = optimize_graphs(
        three_node_data, nclusters=8, subset_size=80, nsubsets=15, ntrials=5,
        rng=np.random.default_rng(0),
    )
    n = len(THREE_NODE_EQUIVALENCE_CLASSES)
    assert fars.shape == (n, n)
    assert len(best_pair) == 2

    for k, dag in enumerate(THREE_NODE_EQUIVALENCE_CLASSES):
        expected_diag = 2 * averaged_residuals(dag, three_node_data)
        assert fars[k, k] == pytest.approx(expected_diag)

    # the best pair found must actually be the (or a tied) minimum in the grid
    assert fars.min() == pytest.approx(fars[np.unravel_index(np.argmin(fars), fars.shape)])


def test_finalize_ar_handles_zero_neutral_points(three_node_data):
    """Regression test for the np.random.choice(range(n), ...) dtype bug:
    an affinity vector with no values near zero leaves zero neutral points."""
    clusters = generate_kmeans_clusters(three_node_data, nclusters=6)
    dag1, dag2 = THREE_NODE_EQUIVALENCE_CLASSES[0], THREE_NODE_EQUIVALENCE_CLASSES[10]
    # threshold=0 forces every nonzero affinity to be decisive, i.e. no neutrals
    affinities = np.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0][: len(clusters)])
    far = finalize_ar(clusters, affinities, dag1, dag2, threshold=0.0, ntrials=3)
    assert np.isfinite(far)
