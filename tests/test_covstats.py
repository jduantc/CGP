"""Pin cgp.covstats's sufficient-statistics math against the direct per-row
lstsq reference (cgp.sem.estimate_parameters), which the covariance form
is derived to be exactly equivalent to (see docs/notes/covariance_ar.md)."""

import numpy as np
import pytest

from cgp.covstats import ar_batch, ar_from_stats, combine_stats, edge_params_from_stats, raw_stats
from cgp.sem import estimate_parameters

DAGS = {
    "empty": np.zeros((3, 3)),
    "chain": np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]),
    "collider": np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]]),
    "complete": np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]),
    "single_parent": np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
}


def _random_data(n, d, seed, offset=False):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, d)) * rng.uniform(0.5, 5, size=d)
    if offset:
        data = data + rng.uniform(-10, 10, size=d)  # nonzero mean, to check centering is handled
    return data


@pytest.mark.parametrize("dag_name", DAGS.keys())
@pytest.mark.parametrize("n,offset", [(50, False), (500, False), (500, True)])
def test_ar_matches_reference(dag_name, n, offset):
    dag = DAGS[dag_name]
    data = _random_data(n, 3, seed=hash((dag_name, n, offset)) % (2**32), offset=offset)

    _, residuals_ref = estimate_parameters(dag, data)
    ar_ref = np.trace(residuals_ref) / data.shape[1]

    ar_cov = ar_from_stats(dag, raw_stats(data))

    assert ar_cov == pytest.approx(ar_ref, abs=1e-10)


@pytest.mark.parametrize("dag_name", DAGS.keys())
def test_edge_params_match_reference(dag_name):
    dag = DAGS[dag_name]
    data = _random_data(300, 3, seed=1)

    params_ref, _ = estimate_parameters(dag, data)
    params_cov = edge_params_from_stats(dag, raw_stats(data))

    np.testing.assert_allclose(params_cov, params_ref, atol=1e-8)


def test_combine_stats_matches_concatenation():
    a = _random_data(120, 3, seed=2)
    b = _random_data(80, 3, seed=3)
    combined_direct = raw_stats(np.vstack((a, b)))
    combined_via_add = combine_stats(raw_stats(a), raw_stats(b))
    for direct, added in zip(combined_direct, combined_via_add):
        np.testing.assert_allclose(added, direct, atol=1e-10)


def test_ar_batch_matches_ar_from_stats_elementwise():
    dag = DAGS["single_parent"]
    batch = [_random_data(100, 3, seed=s) for s in range(5)]
    n = np.array([100] * 5)
    s = np.stack([b.sum(axis=0) for b in batch])
    Q = np.stack([b.T @ b for b in batch])

    batched = ar_batch(dag, n, s, Q)
    individual = [ar_from_stats(dag, raw_stats(b)) for b in batch]

    np.testing.assert_allclose(batched, individual, atol=1e-10)


def test_zero_size_neutral_edge_case_does_not_crash():
    """Regression test: np.random.choice(range(0), ...) used to return a
    float array, breaking indexing downstream (see cgp/probing.py finalize_ar)."""
    idx = np.random.choice(0, size=0, replace=False)
    assert idx.dtype.kind in "iu"  # integer, not float


def test_ar_from_stats_empty_point_set_is_inf_not_a_crash():
    """Regression test: finalize_ar can end up scoring an empty point set
    (e.g. a candidate DAG with zero assigned + zero neutral points), which
    used to divide by n=0 and crash SVD/solve deep in the batched path."""
    dag = DAGS["chain"]
    empty_stats = (0, np.zeros(3), np.zeros((3, 3)))
    assert ar_from_stats(dag, empty_stats) == np.inf


def test_ar_batch_degenerate_entry_does_not_break_the_whole_batch():
    dag = DAGS["single_parent"]
    good = _random_data(50, 3, seed=9)
    n = np.array([50, 0, 1])
    s = np.stack([good.sum(axis=0), np.zeros(3), good[0]])
    Q = np.stack([good.T @ good, np.zeros((3, 3)), np.outer(good[0], good[0])])

    result = ar_batch(dag, n, s, Q)
    assert np.isfinite(result[0])
    assert result[1] == np.inf
    assert result[2] == np.inf
