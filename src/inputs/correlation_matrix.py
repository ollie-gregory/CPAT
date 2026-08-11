"""CorrelationMatrix input type: a validated systematic-factor covariance matrix."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CorrelationMatrix:
    """Validated wrapper around a systematic-factor covariance/correlation matrix.

    Accepts either a pandas DataFrame (factor names taken from its
    columns unless `factor_names` is also given, in which case they
    must match) or a raw ndarray (`factor_names` is then required).
    Owns matrix validity - shape, symmetry, positive-semi-definite
    repair - and derives the Cholesky factor used to correlate
    simulated systematic shocks.

    Parameters
    ----------
    matrix : numpy.ndarray or pandas.DataFrame
        Square covariance (or correlation) matrix over the systematic
        factors.
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
            raise ValueError(f"CorrelationMatrix must be square, got shape {matrix.shape}")

        if matrix.shape[0] != len(self.factor_names):
            raise ValueError(
                f"matrix shape {matrix.shape} does not match number of "
                f"factor_names ({len(self.factor_names)})"
            )

        if len(set(self.factor_names)) != len(self.factor_names):
            raise ValueError(f"factor_names must be unique, got {self.factor_names}")

        if not np.allclose(matrix, matrix.T, atol=1e-6):
            raise ValueError("CorrelationMatrix must be symmetric")

        self.matrix = matrix
        self.factor_index = {name: i for i, name in enumerate(self.factor_names)}
        self.cov = self._make_pd(matrix, eps=self.eps)
        self.cholesky = np.linalg.cholesky(self.cov)

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
