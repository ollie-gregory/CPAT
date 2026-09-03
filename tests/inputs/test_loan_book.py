"""Unit tests for LoanBook."""

import numpy as np
import pandas as pd
import pytest

from src.inputs.loan_book import LoanBook


def make_valid_data():
    return pd.DataFrame(
        {
            "ead": [100.0, 200.0, 50.0],
            "lgd": [0.4, 0.6, 0.5],
            "pd": [0.01, 0.02, 0.03],
            "rsq": [0.2, 0.3, 0.1],
            "region": ["a", "b", "a"],
            "sector": ["x", "y", "x"],
        }
    )


class TestConstruction:
    def test_valid_data_accepted(self):
        lb = LoanBook(data=make_valid_data())
        assert lb.n_loans == 3

    def test_index_is_reset(self):
        data = make_valid_data().set_index(pd.Index([5, 6, 7]))
        lb = LoanBook(data=data)
        assert list(lb.data.index) == [0, 1, 2]

    def test_non_dataframe_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            LoanBook(data=[1, 2, 3])

    def test_custom_factor_columns(self):
        data = make_valid_data().rename(columns={"region": "country"})
        lb = LoanBook(data=data, factor_columns=["country", "sector"])
        assert lb.factor_columns == ["country", "sector"]


class TestRequiredColumnValidation:
    @pytest.mark.parametrize("column", ["ead", "lgd", "pd", "rsq"])
    def test_missing_required_column_raises(self, column):
        data = make_valid_data().drop(columns=[column])
        with pytest.raises(ValueError, match="missing required column"):
            LoanBook(data=data)

    @pytest.mark.parametrize("column", ["ead", "lgd", "pd", "rsq"])
    def test_non_numeric_column_raises(self, column):
        data = make_valid_data()
        data[column] = data[column].astype(str)
        with pytest.raises(TypeError, match="must be numeric"):
            LoanBook(data=data)

    @pytest.mark.parametrize("column", ["ead", "lgd", "pd", "rsq"])
    def test_null_values_raise(self, column):
        data = make_valid_data()
        data.loc[0, column] = np.nan
        with pytest.raises(ValueError, match="missing values"):
            LoanBook(data=data)

    def test_negative_ead_raises(self):
        data = make_valid_data()
        data.loc[0, "ead"] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            LoanBook(data=data)

    @pytest.mark.parametrize("column", ["lgd", "pd", "rsq"])
    def test_out_of_range_values_raise(self, column):
        data = make_valid_data()
        data.loc[0, column] = 1.5
        with pytest.raises(ValueError, match=r"must be within \[0, 1\]"):
            LoanBook(data=data)


class TestFactorColumnValidation:
    def test_empty_factor_columns_raises(self):
        with pytest.raises(ValueError, match="must be non-empty"):
            LoanBook(data=make_valid_data(), factor_columns=[])

    def test_missing_factor_column_raises(self):
        with pytest.raises(ValueError, match="missing factor column"):
            LoanBook(data=make_valid_data(), factor_columns=["region", "missing_col"])


class TestAccessors:
    def test_ead_lgd_pd_rsq_arrays(self):
        lb = LoanBook(data=make_valid_data())
        np.testing.assert_array_equal(lb.ead, [100.0, 200.0, 50.0])
        np.testing.assert_array_equal(lb.lgd, [0.4, 0.6, 0.5])
        np.testing.assert_array_equal(lb.pd, [0.01, 0.02, 0.03])
        np.testing.assert_array_equal(lb.rsq, [0.2, 0.3, 0.1])

    def test_factor_values_returns_column(self):
        lb = LoanBook(data=make_valid_data())
        np.testing.assert_array_equal(lb.factor_values("region"), ["a", "b", "a"])

    def test_factor_values_unknown_column_raises(self):
        lb = LoanBook(data=make_valid_data())
        with pytest.raises(ValueError, match="not a configured factor column"):
            lb.factor_values("not_a_column")
