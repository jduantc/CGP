import numpy as np
import pytest

from cgp.datagen import sample_mixture
from cgp.mixture import em_fit, search_best_graph_set, weighted_stats
from cgp.covstats import raw_stats
from cgp.probing import THREE_NODE_EQUIVALENCE_CLASSES as CLASSES


def test_weighted_stats_all_ones_matches_raw_stats():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((100, 3))
    n_raw, s_raw, Q_raw = raw_stats(data)
    n_w, s_w, Q_w = weighted_stats(data, np.ones(100))
    assert n_w == pytest.approx(n_raw)
    np.testing.assert_allclose(s_w, s_raw)
    np.testing.assert_allclose(Q_w, Q_raw)


def test_em_fit_recovers_true_pair_on_known_ground_truth():
    """The 'hardest' bundled case: two DAGs differing by one edge."""
    rng = np.random.default_rng(0)
    true_dags = [CLASSES[10], CLASSES[7]]  # complete graph vs. single-edge-removed
    merged, _, _ = sample_mixture(*true_dags, num_vars=3, n_each=750, mechanism="linear", rng=rng)

    results = search_best_graph_set(merged, CLASSES, set_size=2, n_iter=60, n_init=2,
                                     rng=np.random.default_rng(1))
    best_bic, best_idxs, _ = results[0]
    assert set(best_idxs) == {7, 10}


def test_em_fit_mixing_weights_sum_to_one():
    rng = np.random.default_rng(2)
    merged, _, _ = sample_mixture(CLASSES[0], CLASSES[10], num_vars=3, n_each=200, rng=rng)
    fit = em_fit(merged, [CLASSES[0], CLASSES[10]], n_iter=30, n_init=1, rng=np.random.default_rng(3))
    assert fit["mixing"].sum() == pytest.approx(1.0)
    assert np.all(fit["responsibilities"].sum(axis=1) == pytest.approx(1.0))


def test_em_fit_loglik_nondecreasing_within_a_run():
    rng = np.random.default_rng(4)
    merged, _, _ = sample_mixture(CLASSES[1], CLASSES[8], num_vars=3, n_each=200, rng=rng)
    fit = em_fit(merged, [CLASSES[1], CLASSES[8]], n_iter=30, n_init=1, rng=np.random.default_rng(5))
    history = np.array(fit["loglik_history"])
    # EM increases the log-likelihood monotonically (up to float error) each iteration
    assert np.all(np.diff(history) >= -1e-6)
