"""Unit tests for CorrelationMatrix."""

import numpy as np
import pandas as pd
import pytest

from src.inputs.correlation_matrix import CorrelationMatrix


def make_valid_df():
    return pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]],
        columns=["region_a", "region_b"],
        index=["region_a", "region_b"],
    )


class TestConstructionFromDataFrame:
    def test_factor_names_default_to_columns(self):
        cm = CorrelationMatrix(matrix=make_valid_df())
        assert cm.factor_names == ["region_a", "region_b"]

    def test_matching_factor_names_accepted(self):
        cm = CorrelationMatrix(matrix=make_valid_df(), factor_names=["region_a", "region_b"])
        assert cm.factor_names == ["region_a", "region_b"]

    def test_mismatched_factor_names_raises(self):
        with pytest.raises(ValueError, match="does not match matrix columns"):
            CorrelationMatrix(matrix=make_valid_df(), factor_names=["x", "y"])


class TestConstructionFromNdarray:
    def test_requires_factor_names(self):
        matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        with pytest.raises(ValueError, match="factor_names must be provided"):
            CorrelationMatrix(matrix=matrix)

    def test_valid_ndarray_accepted(self):
        matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        cm = CorrelationMatrix(matrix=matrix, factor_names=["a", "b"])
        assert cm.factor_names == ["a", "b"]
        assert cm.n_factors == 2


class TestValidation:
    def test_non_square_matrix_raises(self):
        matrix = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3]])
        with pytest.raises(ValueError, match="must be square"):
            CorrelationMatrix(matrix=matrix, factor_names=["a", "b"])

    def test_shape_mismatch_with_factor_names_raises(self):
        matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        with pytest.raises(ValueError, match="does not match number of"):
            CorrelationMatrix(matrix=matrix, factor_names=["a", "b", "c"])

    def test_duplicate_factor_names_raises(self):
        matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        with pytest.raises(ValueError, match="must be unique"):
            CorrelationMatrix(matrix=matrix, factor_names=["a", "a"])

    def test_asymmetric_matrix_raises(self):
        matrix = np.array([[1.0, 0.9], [0.1, 1.0]])
        with pytest.raises(ValueError, match="must be symmetric"):
            CorrelationMatrix(matrix=matrix, factor_names=["a", "b"])


class TestDerivedAttributes:
    def test_factor_index_maps_names_to_positions(self):
        cm = CorrelationMatrix(matrix=make_valid_df())
        assert cm.factor_index == {"region_a": 0, "region_b": 1}

    def test_n_factors(self):
        cm = CorrelationMatrix(matrix=make_valid_df())
        assert cm.n_factors == 2

    def test_cholesky_reconstructs_covariance(self):
        cm = CorrelationMatrix(matrix=make_valid_df())
        reconstructed = cm.cholesky @ cm.cholesky.T
        np.testing.assert_allclose(reconstructed, cm.cov, atol=1e-8)

    def test_already_positive_definite_matrix_is_unchanged(self):
        matrix = np.array([[2.0, 0.0], [0.0, 2.0]])
        cm = CorrelationMatrix(matrix=matrix, factor_names=["a", "b"], eps=1e-8)
        np.testing.assert_allclose(cm.cov, matrix, atol=1e-6)

    def test_non_positive_semi_definite_matrix_is_repaired(self):
        matrix = np.array(
            [
                [1.0, 0.9, -0.9],
                [0.9, 1.0, 0.9],
                [-0.9, 0.9, 1.0],
            ]
        )
        cm = CorrelationMatrix(matrix=matrix, factor_names=["a", "b", "c"])
        eigvals = np.linalg.eigvalsh(cm.cov)
        assert eigvals.min() > 0
        # Cholesky factor must exist for the repaired matrix.
        assert cm.cholesky.shape == (3, 3)
