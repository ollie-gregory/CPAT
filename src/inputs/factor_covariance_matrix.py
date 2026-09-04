"""FactorCovarianceMatrix input type: a validated systematic-factor covariance matrix."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FactorCovarianceMatrix:
    """Validated wrapper around a systematic-factor covariance matrix.

    Accepts either a pandas DataFrame (factor names taken from its
    columns unless `factor_names` is also given, in which case they
    must match) or a raw ndarray (`factor_names` is then required).
    Owns matrix validity - shape, symmetry, positive-semi-definite
    repair - and derives the Cholesky factor used to correlate
    simulated systematic shocks.

    This is a covariance matrix, not a correlation matrix: the factor
    variances are part of the model, not a nuisance scale. A loan
    loading on several factors gets its relative weight on each from
    their variances, so passing a correlation matrix in place of the
    covariance it came from is a different model, not a rescaling of
    the same one. Use `from_correlation` to combine a correlation
    matrix with per-factor volatilities. A correlation matrix is
    still a valid input in its own right - it just states that every
    factor has unit variance.

    Parameters
    ----------
    matrix : numpy.ndarray or pandas.DataFrame
        Square covariance matrix over the systematic factors.
    factor_names : list of str, optional
        Names of the factors, in the same order as `matrix`'s
        rows/columns. Required when `matrix` is a raw ndarray;
        optional (and must match) when `matrix` is a DataFrame, since
        its columns are used as a default.
    eps : float, default 1e-8
        Minimum eigenvalue floor used when repairing `matrix` to be
        positive semi-definite before taking its Cholesky factor.

    Attributes
    ----------
    factor_names : list of str
        Names of the factors, in matrix order.
    factor_index : dict of str to int
        Maps each factor name to its position in `matrix`.
    matrix : numpy.ndarray
        The input matrix as supplied, coerced to an ndarray, before
        positive-semi-definite repair. Use `cov` for the matrix the
        model actually simulates from.
    cov : numpy.ndarray
        The positive-semi-definite-repaired covariance matrix.
    cholesky : numpy.ndarray
        Lower-triangular Cholesky factor of `cov`.

    Raises
    ------
    ValueError
        If `matrix` is not square, its shape doesn't match
        `factor_names`, `factor_names` are not unique, `factor_names`
        conflicts with a DataFrame's columns, or `matrix` is not
        symmetric.
    """

    matrix: np.ndarray | pd.DataFrame
    factor_names: list[str] | None = None
    eps: float = 1e-8

    factor_index: dict = field(init=False, repr=False)
    cov: np.ndarray = field(init=False, repr=False)
    cholesky: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if isinstance(self.matrix, pd.DataFrame):
            columns = list(self.matrix.columns)
            if self.factor_names is None:
                self.factor_names = columns
            elif list(self.factor_names) != columns:
                raise ValueError(
                    f"factor_names {list(self.factor_names)} does not match "
                    f"matrix columns {columns}"
                )
            matrix = self.matrix.to_numpy(dtype=float)
        else:
            if self.factor_names is None:
                raise ValueError("factor_names must be provided when matrix is a raw ndarray")
            matrix = np.asarray(self.matrix, dtype=float)

        self.factor_names = list(self.factor_names)

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"FactorCovarianceMatrix must be square, got shape {matrix.shape}")

        if matrix.shape[0] != len(self.factor_names):
            raise ValueError(
                f"matrix shape {matrix.shape} does not match number of "
                f"factor_names ({len(self.factor_names)})"
            )

        if len(set(self.factor_names)) != len(self.factor_names):
            raise ValueError(f"factor_names must be unique, got {self.factor_names}")

        if not np.allclose(matrix, matrix.T, atol=1e-6):
            raise ValueError("FactorCovarianceMatrix must be symmetric")

        self.matrix = matrix
        self.factor_index = {name: i for i, name in enumerate(self.factor_names)}
        self.cov = self._make_pd(matrix, eps=self.eps)
        self.cholesky = np.linalg.cholesky(self.cov)

    @classmethod
    def from_correlation(
        cls,
        correlation: np.ndarray | pd.DataFrame,
        volatilities,
        factor_names: list[str] | None = None,
        eps: float = 1e-8,
    ) -> "FactorCovarianceMatrix":
        """Build a covariance matrix from a correlation matrix and volatilities.

        Rescales `correlation` into a covariance matrix via
        ``diag(volatilities) @ correlation @ diag(volatilities)``, then
        validates and repairs it as usual. Use this when the
        correlation structure and the factor volatilities are
        estimated or sourced separately.

        Parameters
        ----------
        correlation : numpy.ndarray or pandas.DataFrame
            Square correlation matrix over the systematic factors.
            Must have a unit diagonal.
        volatilities : array-like or pandas.Series
            Per-factor standard deviations, in the same order as
            `correlation`'s rows/columns. A Series is aligned by
            factor name instead of by position.
        factor_names : list of str, optional
            As in the constructor: required when `correlation` is a
            raw ndarray, optional (and must match) when it is a
            DataFrame.
        eps : float, default 1e-8
            Minimum eigenvalue floor, as in the constructor.

        Returns
        -------
        FactorCovarianceMatrix
            The rescaled covariance matrix.

        Raises
        ------
        ValueError
            If `correlation` is not square, does not have a unit
            diagonal, `volatilities` has the wrong length, contains a
            negative value, or (as a Series) is missing a factor.
        """
        is_frame = isinstance(correlation, pd.DataFrame)
        corr = correlation.to_numpy(dtype=float) if is_frame else np.asarray(correlation, dtype=float)

        if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
            raise ValueError(f"correlation must be square, got shape {corr.shape}")

        if isinstance(volatilities, pd.Series):
            names = list(correlation.columns) if is_frame else factor_names
            if names is None:
                raise ValueError(
                    "factor_names must be provided when correlation is a raw ndarray"
                )
            missing = [name for name in names if name not in volatilities.index]
            if missing:
                raise ValueError(f"volatilities is missing factors {missing}")
            vols = volatilities.reindex(names).to_numpy(dtype=float)
        else:
            vols = np.asarray(volatilities, dtype=float)

        if vols.shape != (corr.shape[0],):
            raise ValueError(
                f"volatilities has length {vols.shape} but correlation has "
                f"{corr.shape[0]} factors"
            )

        if np.any(vols < 0):
            raise ValueError("volatilities must be non-negative")

        if not np.allclose(np.diag(corr), 1.0, atol=1e-6):
            raise ValueError(
                "correlation must have a unit diagonal; pass a covariance matrix "
                "to the constructor directly instead"
            )

        cov = corr * np.outer(vols, vols)
        if is_frame:
            cov = pd.DataFrame(cov, index=correlation.index, columns=correlation.columns)
        return cls(matrix=cov, factor_names=factor_names, eps=eps)

    @staticmethod
    def _make_pd(cov: np.ndarray, eps: float) -> np.ndarray:
        cov = (cov + cov.T) / 2
        lam_min = np.linalg.eigvalsh(cov).min()
        delta = max(0.0, -lam_min + eps)
        return cov + delta * np.eye(cov.shape[0])

    @property
    def n_factors(self) -> int:
        """int: Number of systematic factors."""
        return len(self.factor_names)
