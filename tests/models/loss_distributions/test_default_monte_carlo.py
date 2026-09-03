"""Unit tests for the default-only Monte Carlo DefaultSimulator."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.inputs.correlation_matrix import CorrelationMatrix
from src.inputs.loan_book import LoanBook
from src.models.loss_distributions.default_monte_carlo import DefaultSimulator


def make_loan_book(n=20, seed=0):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {
            "ead": rng.uniform(50, 200, n),
            "lgd": rng.uniform(0.2, 0.6, n),
            "pd": rng.uniform(0.01, 0.05, n),
            "rsq": rng.uniform(0.1, 0.4, n),
            "region": rng.choice(["a", "b"], n),
            "sector": rng.choice(["x", "y"], n),
        }
    )
    return LoanBook(data=data, factor_columns=["region", "sector"])


def make_correlation_matrix():
    factor_names = ["a", "b", "x", "y"]
    matrix = pd.DataFrame(np.eye(4) * 0.5 + 0.5, columns=factor_names, index=factor_names)
    return CorrelationMatrix(matrix=matrix, factor_names=factor_names)


@pytest.fixture
def simulator():
    return DefaultSimulator()


class TestUnsimulatedState:
    def test_initial_attributes_are_none(self, simulator):
        assert simulator.loan_book is None
        assert simulator.correlation_matrix is None
        assert simulator.losses is None

    def test_var_quantile_before_simulate_raises(self, simulator):
        with pytest.raises(RuntimeError, match="simulate_losses"):
            simulator.var_quantile()

    def test_expected_shortfall_before_simulate_raises(self, simulator):
        with pytest.raises(RuntimeError, match="simulate_losses"):
            simulator.expected_shortfall()

    def test_plot_before_simulate_raises(self, simulator):
        with pytest.raises(RuntimeError, match="simulate_losses"):
            simulator.plot_loss_distribution()


class TestComputeThresholds:
    def test_matches_norm_ppf(self, simulator):
        pd_arr = np.array([0.01, 0.05, 0.5])
        thresholds = simulator._compute_thresholds(pd_arr)
        np.testing.assert_allclose(thresholds, norm.ppf(pd_arr))

    def test_clips_extreme_probabilities(self, simulator):
        thresholds = simulator._compute_thresholds(np.array([0.0, 1.0]))
        assert np.isfinite(thresholds).all()


class TestBuildFactorLoadings:
    def test_row_norms_are_one(self, simulator):
        lb = make_loan_book()
        cm = make_correlation_matrix()
        fc_hat = simulator._build_factor_loadings(lb, cm)
        norms = np.linalg.norm(fc_hat, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-8)

    def test_shape_is_loans_by_factors(self, simulator):
        lb = make_loan_book(n=10)
        cm = make_correlation_matrix()
        fc_hat = simulator._build_factor_loadings(lb, cm)
        assert fc_hat.shape == (10, cm.n_factors)

    def test_unknown_factor_value_raises(self, simulator):
        data = pd.DataFrame(
            {
                "ead": [100.0],
                "lgd": [0.4],
                "pd": [0.02],
                "rsq": [0.2],
                "region": ["unknown_region"],
                "sector": ["x"],
            }
        )
        lb = LoanBook(data=data, factor_columns=["region", "sector"])
        cm = make_correlation_matrix()
        with pytest.raises(ValueError, match="not a known factor"):
            simulator._build_factor_loadings(lb, cm)


class TestSimulateLosses:
    def test_returns_series_of_correct_length(self, simulator):
        lb = make_loan_book()
        cm = make_correlation_matrix()
        losses = simulator.simulate_losses(lb, cm, n_scenarios=200, chunk=50, seed=1)
        assert isinstance(losses, pd.Series)
        assert len(losses) == 200

    def test_losses_are_non_negative_and_bounded_by_total_exposure(self, simulator):
        lb = make_loan_book()
        cm = make_correlation_matrix()
        losses = simulator.simulate_losses(lb, cm, n_scenarios=200, chunk=50, seed=1)
        assert (losses >= 0).all()
        assert (losses <= lb.ead.sum()).all()

    def test_sets_loan_book_and_correlation_matrix(self, simulator):
        lb = make_loan_book()
        cm = make_correlation_matrix()
        simulator.simulate_losses(lb, cm, n_scenarios=100, chunk=50, seed=1)
        assert simulator.loan_book is lb
        assert simulator.correlation_matrix is cm

    def test_deterministic_with_same_seed(self):
        lb = make_loan_book()
        cm = make_correlation_matrix()
        losses_1 = DefaultSimulator().simulate_losses(lb, cm, n_scenarios=100, chunk=30, seed=7)
        losses_2 = DefaultSimulator().simulate_losses(lb, cm, n_scenarios=100, chunk=30, seed=7)
        pd.testing.assert_series_equal(losses_1, losses_2)

    def test_uneven_chunk_size_still_fills_all_scenarios(self, simulator):
        lb = make_loan_book()
        cm = make_correlation_matrix()
        losses = simulator.simulate_losses(lb, cm, n_scenarios=100, chunk=7, seed=3)
        assert len(losses) == 100
        assert losses.notna().all()


class TestStochasticLGD:
    def test_qsq_none_leaves_lgd_deterministic(self, simulator):
        lb = make_loan_book()
        cm = make_correlation_matrix()
        simulator.simulate_losses(lb, cm, n_scenarios=100, chunk=50, seed=1)
        assert simulator.qsq is None
        assert simulator.lgd_k is None

    def test_records_qsq_and_lgd_k(self, simulator):
        lb = make_loan_book()
        cm = make_correlation_matrix()
        simulator.simulate_losses(lb, cm, n_scenarios=100, chunk=50, seed=1, QSQ=0.3, lgd_k=0.5)
        np.testing.assert_allclose(simulator.qsq, np.full(lb.n_loans, 0.3))
        assert simulator.lgd_k == 0.5

    def test_zero_lgd_k_reproduces_deterministic_lgd(self):
        lb = make_loan_book()
        cm = make_correlation_matrix()
        base = DefaultSimulator().simulate_losses(lb, cm, n_scenarios=300, chunk=100, seed=4)
        degenerate = DefaultSimulator().simulate_losses(
            lb, cm, n_scenarios=300, chunk=100, seed=4, QSQ=0.5, lgd_k=0.0
        )
        np.testing.assert_allclose(degenerate.to_numpy(), base.to_numpy())

    def test_losses_stay_bounded_by_total_exposure(self, simulator):
        lb = make_loan_book()
        cm = make_correlation_matrix()
        losses = simulator.simulate_losses(
            lb, cm, n_scenarios=300, chunk=100, seed=1, QSQ=0.5
        )
        assert (losses >= 0).all()
        assert (losses <= lb.ead.sum()).all()

    def test_deterministic_with_same_seed(self):
        lb = make_loan_book()
        cm = make_correlation_matrix()
        kwargs = dict(n_scenarios=200, chunk=60, seed=7, QSQ=0.4)
        losses_1 = DefaultSimulator().simulate_losses(lb, cm, **kwargs)
        losses_2 = DefaultSimulator().simulate_losses(lb, cm, **kwargs)
        pd.testing.assert_series_equal(losses_1, losses_2)

    def test_correlated_lgd_raises_expected_loss(self):
        lb = make_loan_book()
        cm = make_correlation_matrix()
        uncorrelated = DefaultSimulator()
        uncorrelated.simulate_losses(lb, cm, n_scenarios=100, chunk=100, seed=1, QSQ=0.0)
        correlated = DefaultSimulator()
        correlated.simulate_losses(lb, cm, n_scenarios=100, chunk=100, seed=1, QSQ=0.6)
        naive = float(np.sum(lb.pd * lb.lgd * lb.ead))
        assert uncorrelated.analytic_EL() == pytest.approx(naive)
        assert correlated.analytic_EL() > naive

    def test_analytic_el_matches_simulated_el(self):
        lb = make_loan_book(n=50)
        cm = make_correlation_matrix()
        sim = DefaultSimulator()
        sim.simulate_losses(lb, cm, n_scenarios=40_000, chunk=2_000, seed=11, QSQ=0.5)
        assert sim.losses.mean() == pytest.approx(sim.analytic_EL(), rel=0.05)

    def test_realised_lgd_keeps_its_input_mean(self, simulator):
        # Every loan defaults, so portfolio loss is purely realised LGD x EAD.
        data = make_loan_book(n=30).data.assign(pd=1.0)
        lb = LoanBook(data=data, factor_columns=["region", "sector"])
        cm = make_correlation_matrix()
        losses = simulator.simulate_losses(
            lb, cm, n_scenarios=20_000, chunk=2_000, seed=2, QSQ=0.0
        )
        assert losses.mean() == pytest.approx(float(np.sum(lb.lgd * lb.ead)), rel=0.01)

    def test_per_loan_qsq_is_accepted(self, simulator):
        lb = make_loan_book()
        cm = make_correlation_matrix()
        qsq = np.linspace(0.0, 1.0, lb.n_loans)
        losses = simulator.simulate_losses(lb, cm, n_scenarios=100, chunk=50, seed=1, QSQ=qsq)
        assert losses.notna().all()

    def test_wrong_length_qsq_raises(self, simulator):
        lb = make_loan_book()
        cm = make_correlation_matrix()
        with pytest.raises(ValueError, match="1-D array of length"):
            simulator.simulate_losses(lb, cm, n_scenarios=50, chunk=50, QSQ=np.zeros(3))

    def test_out_of_range_qsq_raises(self, simulator):
        lb = make_loan_book()
        cm = make_correlation_matrix()
        with pytest.raises(ValueError, match=r"within \[0, 1\]"):
            simulator.simulate_losses(lb, cm, n_scenarios=50, chunk=50, QSQ=1.5)

    def test_negative_lgd_k_raises(self, simulator):
        lb = make_loan_book()
        cm = make_correlation_matrix()
        with pytest.raises(ValueError, match="lgd_k"):
            simulator.simulate_losses(lb, cm, n_scenarios=50, chunk=50, QSQ=0.3, lgd_k=-0.1)


class TestSummaryStatistics:
    def test_analytic_el_matches_manual_formula(self, simulator):
        lb = make_loan_book()
        cm = make_correlation_matrix()
        simulator.simulate_losses(lb, cm, n_scenarios=50, chunk=50, seed=1)
        expected = float(np.sum(lb.pd * lb.lgd * lb.ead))
        assert simulator.analytic_EL() == pytest.approx(expected)

    def test_var_quantile_matches_numpy_quantile(self, simulator):
        lb = make_loan_book()
        cm = make_correlation_matrix()
        losses = simulator.simulate_losses(lb, cm, n_scenarios=500, chunk=100, seed=1)
        expected = float(np.quantile(losses.to_numpy(), 0.95))
        assert simulator.var_quantile(0.95) == pytest.approx(expected)

    def test_expected_shortfall_greater_or_equal_to_var(self, simulator):
        lb = make_loan_book()
        cm = make_correlation_matrix()
        simulator.simulate_losses(lb, cm, n_scenarios=500, chunk=100, seed=1)
        es = simulator.expected_shortfall(0.95)
        var = simulator.var_quantile(0.95)
        assert es >= var

    def test_expected_shortfall_handles_degenerate_tail(self, simulator):
        lb = make_loan_book()
        cm = make_correlation_matrix()
        simulator.simulate_losses(lb, cm, n_scenarios=50, chunk=50, seed=1)
        # alpha=1.0 -> tail should be the max value (or fall back to the quantile).
        es = simulator.expected_shortfall(1.0)
        assert es == pytest.approx(simulator.losses.max())


class TestPlotLossDistribution:
    def test_returns_figure_after_simulate(self, simulator):
        lb = make_loan_book()
        cm = make_correlation_matrix()
        simulator.simulate_losses(lb, cm, n_scenarios=100, chunk=50, seed=1)
        fig = simulator.plot_loss_distribution()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
