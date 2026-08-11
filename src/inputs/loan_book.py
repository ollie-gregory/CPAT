from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REQUIRED_NUMERIC_COLUMNS = ("ead", "lgd", "pd", "rsq")
DEFAULT_FACTOR_COLUMNS = ("region", "sector")


@dataclass
class LoanBook:
    """Validated wrapper around a loan-level portfolio DataFrame.

    Required columns: ead, lgd, pd, rsq (all numeric).
    factor_columns names the categorical columns that define each loan's
    systematic-factor membership (e.g. region, sector) - any number of
    columns is supported.
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
        return len(self.data)

    @property
    def ead(self) -> np.ndarray:
        return self.data["ead"].to_numpy(dtype=float)

    @property
    def lgd(self) -> np.ndarray:
        return self.data["lgd"].to_numpy(dtype=float)

    @property
    def pd(self) -> np.ndarray:
        return self.data["pd"].to_numpy(dtype=float)

    @property
    def rsq(self) -> np.ndarray:
        return self.data["rsq"].to_numpy(dtype=float)

    def factor_values(self, column: str) -> np.ndarray:
        if column not in self.factor_columns:
            raise ValueError(f"'{column}' is not a configured factor column ({self.factor_columns})")
        return self.data[column].to_numpy()
