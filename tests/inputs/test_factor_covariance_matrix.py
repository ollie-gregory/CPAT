"""Unit tests for FactorCovarianceMatrix."""

import numpy as np
import pandas as pd
import pytest

from src.inputs.factor_covariance_matrix import FactorCovarianceMatrix


def make_valid_df():
    return pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]],
        columns=["region_a", "region_b"],
        index=["region_a", "region_b"],
    )


class TestConstructionFromDataFrame:
    def test_factor_names_default_to_columns(self):
        fcov = FactorCovarianceMatrix(matrix=make_valid_df())
        assert fcov.factor_names == ["region_a", "region_b"]

    def test_matching_factor_names_accepted(self):
        fcov = FactorCovarianceMatrix(matrix=make_valid_df(), factor_names=["region_a", "region_b"])
        assert fcov.factor_names == ["region_a", "region_b"]

    def test_mismatched_factor_names_raises(self):
        with pytest.raises(ValueError, match="does not match matrix columns"):
            FactorCovarianceMatrix(matrix=make_valid_df(), factor_names=["x", "y"])


class TestConstructionFromNdarray:
    def test_requires_factor_names(self):
        matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        with pytest.raises(ValueError, match="factor_names must be provided"):
            FactorCovarianceMatrix(matrix=matrix)

    def test_valid_ndarray_accepted(self):
        matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        fcov = FactorCovarianceMatrix(matrix=matrix, factor_names=["a", "b"])
        assert fcov.factor_names == ["a", "b"]
        assert fcov.n_factors == 2


class TestValidation:
    def test_non_square_matrix_raises(self):
        matrix = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3]])
        with pytest.raises(ValueError, match="must be square"):
            FactorCovarianceMatrix(matrix=matrix, factor_names=["a", "b"])

    def test_shape_mismatch_with_factor_names_raises(self):
        matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        with pytest.raises(ValueError, match="does not match number of"):
            FactorCovarianceMatrix(matrix=matrix, factor_names=["a", "b", "c"])

    def test_duplicate_factor_names_raises(self):
        matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        with pytest.raises(ValueError, match="must be unique"):
            FactorCovarianceMatrix(matrix=matrix, factor_names=["a", "a"])

    def test_asymmetric_matrix_raises(self):
        matrix = np.array([[1.0, 0.9], [0.1, 1.0]])
        with pytest.raises(ValueError, match="must be symmetric"):
            FactorCovarianceMatrix(matrix=matrix, factor_names=["a", "b"])


class TestDerivedAttributes:
    def test_factor_index_maps_names_to_positions(self):
        fcov = FactorCovarianceMatrix(matrix=make_valid_df())
        assert fcov.factor_index == {"region_a": 0, "region_b": 1}

    def test_n_factors(self):
        fcov = FactorCovarianceMatrix(matrix=make_valid_df())
        assert fcov.n_factors == 2

    def test_cholesky_reconstructs_covariance(self):
        fcov = FactorCovarianceMatrix(matrix=make_valid_df())
        reconstructed = fcov.cholesky @ fcov.cholesky.T
        np.testing.assert_allclose(reconstructed, fcov.cov, atol=1e-8)

    def test_already_positive_definite_matrix_is_unchanged(self):
        matrix = np.array([[2.0, 0.0], [0.0, 2.0]])
        fcov = FactorCovarianceMatrix(matrix=matrix, factor_names=["a", "b"], eps=1e-8)
        np.testing.assert_allclose(fcov.cov, matrix, atol=1e-6)

    def test_non_positive_semi_definite_matrix_is_repaired(self):
        matrix = np.array(
            [
                [1.0, 0.9, -0.9],
                [0.9, 1.0, 0.9],
                [-0.9, 0.9, 1.0],
            ]
        )
        fcov = FactorCovarianceMatrix(matrix=matrix, factor_names=["a", "b", "c"])
        eigvals = np.linalg.eigvalsh(fcov.cov)
        assert eigvals.min() > 0
        # Cholesky factor must exist for the repaired matrix.
        assert fcov.cholesky.shape == (3, 3)


class TestFromCorrelation:
    def test_rescales_correlation_by_volatilities(self):
        fcov = FactorCovarianceMatrix.from_correlation(make_valid_df(), [0.2, 0.1])
        expected = np.array([[0.04, 0.01], [0.01, 0.01]])
        np.testing.assert_allclose(fcov.cov, expected, atol=1e-12)

    def test_factor_names_default_to_columns(self):
        fcov = FactorCovarianceMatrix.from_correlation(make_valid_df(), [0.2, 0.1])
        assert fcov.factor_names == ["region_a", "region_b"]

    def test_unit_volatilities_leave_correlation_unchanged(self):
        fcov = FactorCovarianceMatrix.from_correlation(make_valid_df(), [1.0, 1.0])
        np.testing.assert_allclose(fcov.cov, make_valid_df().to_numpy(), atol=1e-12)

    def test_accepts_ndarray_with_factor_names(self):
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        fcov = FactorCovarianceMatrix.from_correlation(corr, [0.2, 0.1], factor_names=["a", "b"])
        assert fcov.factor_names == ["a", "b"]

    def test_ndarray_without_factor_names_raises(self):
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        with pytest.raises(ValueError, match="factor_names must be provided"):
            FactorCovarianceMatrix.from_correlation(corr, [0.2, 0.1])

    def test_series_volatilities_align_by_name(self):
        vols = pd.Series({"region_b": 0.1, "region_a": 0.2})
        fcov = FactorCovarianceMatrix.from_correlation(make_valid_df(), vols)
        expected = np.array([[0.04, 0.01], [0.01, 0.01]])
        np.testing.assert_allclose(fcov.cov, expected, atol=1e-12)

    def test_series_missing_a_factor_raises(self):
        vols = pd.Series({"region_a": 0.2})
        with pytest.raises(ValueError, match="missing factors"):
            FactorCovarianceMatrix.from_correlation(make_valid_df(), vols)

    def test_non_square_correlation_raises(self):
        corr = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3]])
        with pytest.raises(ValueError, match="correlation must be square"):
            FactorCovarianceMatrix.from_correlation(corr, [0.2, 0.1], factor_names=["a", "b"])

    def test_wrong_length_volatilities_raises(self):
        with pytest.raises(ValueError, match="volatilities has length"):
            FactorCovarianceMatrix.from_correlation(make_valid_df(), [0.2, 0.1, 0.3])

    def test_negative_volatilities_raise(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            FactorCovarianceMatrix.from_correlation(make_valid_df(), [0.2, -0.1])

    def test_non_unit_diagonal_raises(self):
        cov = pd.DataFrame(
            [[0.04, 0.006], [0.006, 0.01]],
            columns=["region_a", "region_b"],
            index=["region_a", "region_b"],
        )
        with pytest.raises(ValueError, match="unit diagonal"):
            FactorCovarianceMatrix.from_correlation(cov, [1.0, 1.0])
