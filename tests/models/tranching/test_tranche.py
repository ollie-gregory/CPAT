"""Unit tests for the Tranche loss-distribution transformation."""

import numpy as np
import pytest

from src.models.tranching.tranche import Tranche
from src.results.loss_distribution import LossDistribution


def make_portfolio(losses=(0.0, 10.0, 50.0, 90.0, 100.0), notional=100.0):
    return LossDistribution(losses=np.array(losses, dtype=float), notional=notional)


class TestValidation:
    @pytest.mark.parametrize(
        "attachment,detachment",
        [(0.5, 0.5), (0.7, 0.3), (-0.1, 0.5), (0.5, 1.2)],
    )
    def test_bad_points_raise(self, attachment, detachment):
        with pytest.raises(ValueError, match="attachment < detachment"):
            Tranche(make_portfolio(), attachment, detachment)

    def test_non_finite_points_raise(self):
        with pytest.raises(ValueError, match="finite"):
            Tranche(make_portfolio(), 0.0, np.nan)

    def test_name_defaults_to_subordination_points(self):
        assert Tranche(make_portfolio(), 0.03, 0.07).name == "3%-7%"

    def test_explicit_name_is_kept(self):
        assert Tranche(make_portfolio(), 0.03, 0.07, "Mezz").name == "Mezz"


class TestGeometry:
    def test_amounts_scale_by_pool_notional(self):
        tranche = Tranche(make_portfolio(notional=1_000.0), 0.03, 0.07)
        assert tranche.attachment_amount == pytest.approx(30.0)
        assert tranche.detachment_amount == pytest.approx(70.0)
        assert tranche.thickness == pytest.approx(0.04)
        assert tranche.size == pytest.approx(40.0)


class TestDistribution:
    def test_clips_at_both_boundaries(self):
        # Pool notional 100, so the tranche spans losses of 20 to 60.
        tranche = Tranche(make_portfolio(), 0.2, 0.6)
        np.testing.assert_allclose(tranche.distribution.losses, [0.0, 0.0, 30.0, 40.0, 40.0])

    def test_notional_is_the_tranche_size(self):
        tranche = Tranche(make_portfolio(), 0.2, 0.6)
        assert tranche.distribution.notional == pytest.approx(tranche.size)

    def test_full_stack_tranche_reproduces_the_portfolio(self):
        portfolio = make_portfolio()
        whole = Tranche(portfolio, 0.0, 1.0)
        np.testing.assert_allclose(whole.distribution.losses, portfolio.losses)
        assert whole.distribution.notional == pytest.approx(portfolio.notional)

    def test_carries_no_analytic_el(self):
        portfolio = LossDistribution(losses=np.array([1.0, 2.0]), notional=10.0, analytic_el=1.5)
        assert Tranche(portfolio, 0.0, 0.5).distribution.analytic_el is None

    def test_is_cached(self):
        tranche = Tranche(make_portfolio(), 0.2, 0.6)
        assert tranche.distribution is tranche.distribution

    def test_statistics_come_from_the_shared_type(self):
        tranche = Tranche(make_portfolio(), 0.2, 0.6)
        dist = tranche.distribution
        assert dist.expected_loss == pytest.approx(dist.losses.mean())
        assert dist.expected_loss_rate == pytest.approx(dist.losses.mean() / tranche.size)


class TestProbabilities:
    def test_prob_attachment_counts_scenarios_above_the_attachment(self):
        # Losses 0, 10, 50, 90, 100 against a notional of 100.
        assert Tranche(make_portfolio(), 0.2, 0.6).prob_attachment == pytest.approx(0.6)

    def test_prob_wipeout_counts_scenarios_at_or_above_the_detachment(self):
        assert Tranche(make_portfolio(), 0.2, 0.6).prob_wipeout == pytest.approx(0.4)

    def test_wipeout_never_exceeds_attachment(self):
        rng = np.random.default_rng(3)
        portfolio = LossDistribution(losses=rng.gamma(2.0, 5.0, 1_000), notional=100.0)
        tranche = Tranche(portfolio, 0.05, 0.2)
        assert tranche.prob_wipeout <= tranche.prob_attachment


class TestSubordination:
    def test_expected_loss_rate_falls_with_seniority(self):
        rng = np.random.default_rng(1)
        portfolio = LossDistribution(losses=rng.gamma(2.0, 3.0, 5_000), notional=100.0)
        rates = [
            Tranche(portfolio, a, d).distribution.expected_loss_rate
            for a, d in [(0.0, 0.03), (0.03, 0.07), (0.07, 1.0)]
        ]
        assert rates[0] > rates[1] > rates[2]


class TestSummary:
    def test_carries_points_statistics_and_probabilities(self):
        summary = Tranche(make_portfolio(), 0.2, 0.6, "Mezz").summary()
        assert summary.name == "Mezz"
        assert summary["Attachment"] == pytest.approx(0.2)
        assert summary["Detachment"] == pytest.approx(0.6)
        assert summary["Size"] == pytest.approx(40.0)
        assert summary["P(Attachment)"] == pytest.approx(0.6)
        assert summary["P(Wipeout)"] == pytest.approx(0.4)
        assert "VaR 99%" in summary.index
