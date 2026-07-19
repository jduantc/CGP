import numpy as np
import pytest

from cgp.scoring import averaged_residuals, edgelist_to_matrix


def test_edgelist_to_matrix_basic():
    mat = edgelist_to_matrix([(0, 1), (1, 2)], 3)
    expected = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    np.testing.assert_array_equal(mat, expected)


def test_averaged_residuals_accepts_edgelist_or_matrix():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((200, 3))
    edges = [(0, 1), (1, 2)]
    mat = edgelist_to_matrix(edges, 3)

    ar_edges = averaged_residuals(edges, data)
    ar_matrix = averaged_residuals(mat, data)

    assert ar_edges == pytest.approx(ar_matrix)


def test_denser_graph_never_worse_fit_than_sparser_subgraph():
    """A superset of edges can only reduce (or match) each node's residual
    variance relative to a subset of those same edges -- OLS with an extra
    regressor never increases in-sample SSR."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal((300, 3))
    sparse = [(0, 2)]
    dense = [(0, 2), (1, 2)]

    assert averaged_residuals(dense, data) <= averaged_residuals(sparse, data)
