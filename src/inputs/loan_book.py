"""LoanBook input type: a validated, column-generic loan-level portfolio."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REQUIRED_NUMERIC_COLUMNS = ("ead", "lgd", "pd", "rsq")
DEFAULT_FACTOR_COLUMNS = ("region", "sector")


@dataclass
class LoanBook:
    """Validated wrapper around a loan-level portfolio DataFrame.

    Wraps a raw pandas DataFrame of loan-level exposures and validates
    that it carries the columns a credit portfolio model needs, without
    hardcoding which categorical columns define systematic-factor
    membership.

    Parameters
    ----------
    data : pandas.DataFrame
        Loan-level portfolio data. Must contain the numeric columns
        ``ead`` (exposure at default, >= 0), ``lgd`` (loss given
        default, in [0, 1]), ``pd`` (probability of default, in
        [0, 1]) and ``rsq`` (systematic R-squared, in [0, 1]), plus
        every column named in `factor_columns`.
    factor_columns : list of str, default ['region', 'sector']
        Names of the categorical columns in `data` that define each
        loan's systematic-factor membership (e.g. region, sector,
        industry). Any number of columns is supported.

    Attributes
    ----------
    data : pandas.DataFrame
        The validated portfolio data, with its index reset.
    factor_columns : list of str
        The configured factor-membership columns.

    Raises
    ------
    TypeError
        If `data` is not a DataFrame, or a required numeric column is
        not numeric dtype.
    ValueError
        If a required column is missing, contains nulls, falls outside
        its valid range, or `factor_columns` is empty or references
        columns not present in `data`.
    """

    data: pd.DataFrame
    factor_columns: list[str] = field(default_factory=lambda: list(DEFAULT_FACTOR_COLUMNS))

    def __post_init__(self):
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError(f"LoanBook.data must be a pandas DataFrame, got {type(self.data)}")

        self.data = self.data.reset_index(drop=True)

        missing = [c for c in REQUIRED_NUMERIC_COLUMNS if c not in self.data.columns]
        if missing:
            raise ValueError(f"LoanBook is missing required column(s): {missing}")

        non_numeric = [c for c in REQUIRED_NUMERIC_COLUMNS if not pd.api.types.is_numeric_dtype(self.data[c])]
        if non_numeric:
            raise TypeError(f"LoanBook column(s) must be numeric: {non_numeric}")

        with_nulls = [c for c in REQUIRED_NUMERIC_COLUMNS if self.data[c].isna().any()]
        if with_nulls:
            raise ValueError(f"LoanBook column(s) contain missing values: {with_nulls}")

        if (self.data["ead"] < 0).any():
            raise ValueError("LoanBook column 'ead' must be non-negative")

        for col in ("lgd", "pd", "rsq"):
            if not self.data[col].between(0.0, 1.0).all():
                raise ValueError(f"LoanBook column '{col}' must be within [0, 1]")

        if not self.factor_columns:
            raise ValueError("LoanBook.factor_columns must be non-empty")

        missing_factor_cols = [c for c in self.factor_columns if c not in self.data.columns]
        if missing_factor_cols:
            raise ValueError(f"LoanBook is missing factor column(s): {missing_factor_cols}")

    @property
    def n_loans(self) -> int:
        """int: Number of loans in the portfolio."""
        return len(self.data)

    @property
    def ead(self) -> np.ndarray:
        """numpy.ndarray: Exposure at default for each loan."""
        return self.data["ead"].to_numpy(dtype=float)

    @property
    def lgd(self) -> np.ndarray:
        """numpy.ndarray: Loss given default for each loan, in [0, 1]."""
        return self.data["lgd"].to_numpy(dtype=float)

    @property
    def pd(self) -> np.ndarray:
        """numpy.ndarray: Probability of default for each loan, in [0, 1]."""
        return self.data["pd"].to_numpy(dtype=float)

    @property
    def rsq(self) -> np.ndarray:
        """numpy.ndarray: Systematic R-squared for each loan, in [0, 1]."""
        return self.data["rsq"].to_numpy(dtype=float)

    def factor_values(self, column: str) -> np.ndarray:
        """Return each loan's category label for one factor column.

        Parameters
        ----------
        column : str
            Name of a column in `factor_columns`.

        Returns
        -------
        numpy.ndarray
            The category label of every loan for `column`.

        Raises
        ------
        ValueError
            If `column` is not one of `factor_columns`.
        """
        if column not in self.factor_columns:
            raise ValueError(f"'{column}' is not a configured factor column ({self.factor_columns})")
        return self.data[column].to_numpy()
