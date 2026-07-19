import numpy as np
import pytest

from cgp.datagen import sample_dag_data, sample_k_mixture, sample_mixture, structural_distance
from cgp.probing import THREE_NODE_EQUIVALENCE_CLASSES as CLASSES
from cgp.scoring import averaged_residuals


@pytest.mark.parametrize("mechanism", ["linear", "quadratic", "cubic"])
def test_sample_dag_data_is_finite_and_fits_true_dag_better_than_empty(mechanism):
    rng = np.random.default_rng(0)
    data, _ = sample_dag_data(CLASSES[10], 3, 1000, mechanism=mechanism, rng=rng)
    assert np.all(np.isfinite(data))
    assert averaged_residuals(CLASSES[10], data) < averaged_residuals(CLASSES[0], data)


def test_unknown_mechanism_raises():
    with pytest.raises(ValueError):
        sample_dag_data(CLASSES[1], 3, 100, mechanism="quartic")


def test_sample_mixture_matches_sample_k_mixture_for_two_components():
    rng1, rng2 = np.random.default_rng(0), np.random.default_rng(0)
    merged_a, sub1_a, sub2_a = sample_mixture(CLASSES[0], CLASSES[10], num_vars=3, n_each=100, rng=rng1)
    merged_b, subsets_b = sample_k_mixture([CLASSES[0], CLASSES[10]], num_vars=3, n_each=100, rng=rng2)

    np.testing.assert_allclose(merged_a, merged_b)
    np.testing.assert_allclose(sub1_a, subsets_b[0])
    np.testing.assert_allclose(sub2_a, subsets_b[1])


def test_sample_k_mixture_supports_more_than_two_components():
    rng = np.random.default_rng(1)
    merged, subsets = sample_k_mixture([CLASSES[0], CLASSES[1], CLASSES[10]], num_vars=3, n_each=150, rng=rng)
    assert len(subsets) == 3
    assert merged.shape == (450, 3)
    assert all(s.shape == (150, 3) for s in subsets)


def test_structural_distance_basic():
    assert structural_distance(CLASSES[0], CLASSES[0]) == 0
    assert structural_distance(CLASSES[7], CLASSES[10]) == 1


def test_separate_calls_draw_different_coefficients():
    """Documents real behavior that has caused a subtle train/test bug in
    practice: two calls to sample_dag_data, even sharing one rng stream, use
    DIFFERENT random edge coefficients -- so they are not train/test splits of
    one SEM. Generate one dataset and split rows instead of calling this twice."""
    rng = np.random.default_rng(0)
    _, coefs_a = sample_dag_data(CLASSES[1], 3, 50, mechanism="quadratic", rng=rng)
    _, coefs_b = sample_dag_data(CLASSES[1], 3, 50, mechanism="quadratic", rng=rng)
    assert coefs_a[(0, 1)]["linear"] != coefs_b[(0, 1)]["linear"]
