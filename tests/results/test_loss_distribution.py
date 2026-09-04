"""Unit tests for the shared LossDistribution result type."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.results.loss_distribution import LossDistribution


def make_distribution(n=500, notional=1_000.0, seed=0, analytic_el=None):
    rng = np.random.default_rng(seed)
    return LossDistribution(
        losses=rng.gamma(2.0, 10.0, n),
        notional=notional,
        analytic_el=analytic_el,
    )


class TestValidation:
    def test_coerces_losses_to_float_array(self):
        dist = LossDistribution(losses=[1, 2, 3], notional=10.0)
        assert isinstance(dist.losses, np.ndarray)
        assert dist.losses.dtype == float

    def test_accepts_a_series(self):
        dist = LossDistribution(losses=pd.Series([1.0, 2.0]), notional=10.0)
        np.testing.assert_allclose(dist.losses, [1.0, 2.0])

    def test_two_dimensional_losses_raise(self):
        with pytest.raises(ValueError, match="1-D"):
            LossDistribution(losses=np.zeros((2, 2)), notional=10.0)

    def test_empty_losses_raise(self):
        with pytest.raises(ValueError, match="at least one scenario"):
            LossDistribution(losses=np.array([]), notional=10.0)

    def test_non_finite_losses_raise(self):
        with pytest.raises(ValueError, match="finite"):
            LossDistribution(losses=np.array([1.0, np.nan]), notional=10.0)

    @pytest.mark.parametrize("notional", [0.0, -5.0, np.inf, np.nan])
    def test_bad_notional_raises(self, notional):
        with pytest.raises(ValueError, match="finite and positive"):
            LossDistribution(losses=np.array([1.0]), notional=notional)


class TestStatistics:
    def test_len_and_n_scenarios_agree(self):
        dist = make_distribution(n=250)
        assert len(dist) == 250
        assert dist.n_scenarios == 250

    def test_expected_loss_is_the_mean(self):
        dist = make_distribution()
        assert dist.expected_loss == pytest.approx(dist.losses.mean())

    def test_rates_divide_by_notional(self):
        dist = LossDistribution(losses=np.array([10.0, 30.0]), notional=100.0)
        np.testing.assert_allclose(dist.loss_rates, [0.1, 0.3])
        assert dist.expected_loss_rate == pytest.approx(0.2)

    def test_unexpected_loss_is_the_standard_deviation(self):
        dist = make_distribution()
        assert dist.unexpected_loss == pytest.approx(dist.losses.std())

    def test_var_matches_numpy_quantile(self):
        dist = make_distribution()
        expected = float(np.quantile(dist.losses, 0.95))
        assert dist.var(0.95) == pytest.approx(expected)

    def test_expected_shortfall_greater_or_equal_to_var(self):
        dist = make_distribution()
        assert dist.expected_shortfall(0.95) >= dist.var(0.95)

    def test_expected_shortfall_handles_degenerate_tail(self):
        dist = make_distribution(n=50)
        # alpha=1.0 -> tail should be the max value (or fall back to the quantile).
        assert dist.expected_shortfall(1.0) == pytest.approx(dist.losses.max())

    def test_to_series_carries_the_name(self):
        dist = make_distribution()
        series = dist.to_series()
        assert isinstance(series, pd.Series)
        assert series.name == "Portfolio"
        np.testing.assert_allclose(series.to_numpy(), dist.losses)


class TestSummary:
    def test_reports_requested_alphas_only(self):
        summary = make_distribution().summary(alphas=(0.99,))
        assert "VaR 99%" in summary.index
        assert "VaR 95%" not in summary.index

    def test_rates_are_consistent_with_amounts(self):
        dist = make_distribution(notional=2_000.0)
        summary = dist.summary()
        assert summary["VaR 99% Rate"] == pytest.approx(summary["VaR 99%"] / dist.notional)
        assert summary["EL Rate"] == pytest.approx(summary["EL"] / dist.notional)

    def test_analytic_el_included_only_when_set(self):
        assert "Analytic EL" not in make_distribution().summary().index
        summary = make_distribution(analytic_el=25.0).summary()
        assert summary["Analytic EL"] == pytest.approx(25.0)

    def test_named_after_the_distribution(self):
        dist = LossDistribution(losses=np.array([1.0, 2.0]), notional=10.0, name="Equity")
        assert dist.summary().name == "Equity"


class TestPlot:
    def test_returns_a_figure(self):
        fig = make_distribution().plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_draws_on_a_supplied_axes(self):
        fig, ax = plt.subplots()
        returned = make_distribution().plot(ax=ax)
        assert returned is fig
        plt.close(fig)

    def test_analytic_el_line_only_when_set(self):
        without = make_distribution().plot()
        with_el = make_distribution(analytic_el=25.0).plot()
        labels = lambda fig: [line.get_label() for line in fig.axes[0].lines]
        assert "Analytic EL" not in labels(without)
        assert "Analytic EL" in labels(with_el)
        plt.close(without)
        plt.close(with_el)

    def test_one_var_line_per_alpha(self):
        fig = make_distribution().plot(alphas=(0.9, 0.95, 0.99))
        var_lines = [ln for ln in fig.axes[0].lines if ln.get_label().startswith("VaR")]
        assert len(var_lines) == 3
        plt.close(fig)

    def test_title_defaults_to_the_name(self):
        dist = LossDistribution(losses=np.array([1.0, 2.0]), notional=10.0, name="Mezz")
        fig = dist.plot()
        assert fig.axes[0].get_title() == "Mezz Loss Distribution"
        plt.close(fig)
