"""Unit tests for the CapitalStructure tranche stack."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.models.tranching.capital_structure import CapitalStructure
from src.results.loss_distribution import LossDistribution


def make_portfolio(n=2_000, notional=100.0, seed=0):
    rng = np.random.default_rng(seed)
    return LossDistribution(losses=rng.gamma(2.0, 3.0, n), notional=notional)


@pytest.fixture
def stack():
    return CapitalStructure(make_portfolio(), [0.03, 0.07], names=["Equity", "Mezz", "Senior"])


class TestValidation:
    def test_empty_cut_points_raise(self):
        with pytest.raises(ValueError, match="at least one point"):
            CapitalStructure(make_portfolio(), [])

    @pytest.mark.parametrize("cut_points", [[0.0, 0.5], [0.5, 1.0], [-0.1], [1.5]])
    def test_cut_points_outside_the_unit_interval_raise(self, cut_points):
        with pytest.raises(ValueError, match=r"strictly within \(0, 1\)"):
            CapitalStructure(make_portfolio(), cut_points)

    @pytest.mark.parametrize("cut_points", [[0.07, 0.03], [0.03, 0.03]])
    def test_unordered_cut_points_raise(self, cut_points):
        with pytest.raises(ValueError, match="strictly increasing"):
            CapitalStructure(make_portfolio(), cut_points)

    def test_wrong_number_of_names_raises(self):
        with pytest.raises(ValueError, match="must have 3 entries"):
            CapitalStructure(make_portfolio(), [0.03, 0.07], names=["Equity", "Senior"])


class TestConstruction:
    def test_builds_one_more_tranche_than_cut_points(self, stack):
        assert len(stack) == 3
        assert [t.name for t in stack] == ["Equity", "Mezz", "Senior"]

    def test_tranches_tile_the_pool(self, stack):
        assert stack[0].attachment == pytest.approx(0.0)
        assert stack[-1].detachment == pytest.approx(1.0)
        for junior, senior in zip(stack.tranches, stack.tranches[1:]):
            assert junior.detachment == pytest.approx(senior.attachment)

    def test_names_default_to_subordination_points(self):
        unnamed = CapitalStructure(make_portfolio(), [0.03, 0.07])
        assert [t.name for t in unnamed] == ["0%-3%", "3%-7%", "7%-100%"]

    def test_indexable_by_position_and_by_name(self, stack):
        assert stack["Mezz"] is stack[1]

    def test_unknown_name_raises(self, stack):
        with pytest.raises(KeyError, match="no tranche named"):
            stack["Junior"]


class TestTilingInvariant:
    def test_tranche_losses_sum_to_the_portfolio_loss(self, stack):
        total = sum(tranche.distribution.losses for tranche in stack)
        np.testing.assert_allclose(total, stack.portfolio.losses)

    def test_tranche_expected_losses_sum_to_the_portfolio_expected_loss(self, stack):
        total = sum(tranche.distribution.expected_loss for tranche in stack)
        assert total == pytest.approx(stack.portfolio.expected_loss)

    def test_tranche_sizes_sum_to_the_pool_notional(self, stack):
        assert sum(t.size for t in stack) == pytest.approx(stack.portfolio.notional)


class TestSummary:
    def test_one_row_per_tranche_named_and_ordered(self, stack):
        summary = stack.summary()
        assert list(summary.index) == ["Equity", "Mezz", "Senior"]
        assert {"Attachment", "Detachment", "EL", "EL Rate", "P(Wipeout)"} <= set(summary.columns)

    def test_expected_loss_rate_falls_with_seniority(self, stack):
        rates = stack.summary()["EL Rate"].to_numpy()
        assert (np.diff(rates) < 0).all()

    def test_reports_requested_alphas_only(self, stack):
        summary = stack.summary(alphas=(0.99,))
        assert "VaR 99%" in summary.columns
        assert "VaR 95%" not in summary.columns


class TestPlots:
    def test_expected_loss_plot_returns_a_figure(self, stack):
        fig = stack.plot_expected_loss()
        assert isinstance(fig, plt.Figure)
        assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["Equity", "Mezz", "Senior"]
        plt.close(fig)

    def test_expected_loss_plot_draws_on_a_supplied_axes(self, stack):
        fig, ax = plt.subplots()
        assert stack.plot_expected_loss(ax=ax) is fig
        plt.close(fig)

    def test_distributions_plot_has_one_panel_per_tranche(self, stack):
        fig = stack.plot_distributions()
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == len(stack)
        assert fig.axes[0].get_title() == "Equity Loss Distribution"
        plt.close(fig)
