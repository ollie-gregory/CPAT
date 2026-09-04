"""Validated input types (LoanBook, FactorCovarianceMatrix, ...) shared across models."""

from .loan_book import LoanBook
from .factor_covariance_matrix import FactorCovarianceMatrix

__all__ = ["LoanBook", "FactorCovarianceMatrix"]
