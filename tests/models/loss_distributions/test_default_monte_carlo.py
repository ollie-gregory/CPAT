"""Unit tests for the default-only Monte Carlo DefaultSimulator."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, t as student_t

from src.inputs.factor_covariance_matrix import FactorCovarianceMatrix
from src.inputs.loan_book import LoanBook
from src.models.loss_distributions.default_monte_carlo import DefaultSimulator
from src.results.loss_distribution import LossDistribution


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


def make_factor_covariance():
    factor_names = ["a", "b", "x", "y"]
    matrix = pd.DataFrame(np.eye(4) * 0.5 + 0.5, columns=factor_names, index=factor_names)
    return FactorCovarianceMatrix(matrix=matrix, factor_names=factor_names)


@pytest.fixture
def simulator():
    return DefaultSimulator()


class TestUnsimulatedState:
    def test_initial_attributes_are_none(self, simulator):
        assert simulator.loan_book is None
        assert simulator.factor_covariance is None
        assert simulator.copula is None
        assert simulator.copula_df is None
        assert simulator.distribution is None

    def test_analytic_el_before_simulate_raises(self, simulator):
        with pytest.raises(RuntimeError, match="simulate_losses"):
            simulator.analytic_EL()


class TestComputeThresholds:
    def test_matches_norm_ppf(self, simulator):
        pd_arr = np.array([0.01, 0.05, 0.5])
        thresholds = simulator._compute_thresholds(pd_arr)
        np.testing.assert_allclose(thresholds, norm.ppf(pd_arr))

    def test_clips_extreme_probabilities(self, simulator):
        thresholds = simulator._compute_thresholds(np.array([0.0, 1.0]))
        assert np.isfinite(thresholds).all()


class TestDefaultThresholds:
    def test_gaussian_copula_uses_probit(self, simulator):
        simulator._prep_copula("gaussian", 5.0)
        pd_arr = np.array([0.01, 0.05, 0.5])
        np.testing.assert_allclose(simulator._default_thresholds(pd_arr), norm.ppf(pd_arr))

    def test_t_copula_uses_t_quantile(self, simulator):
        simulator._prep_copula("t", 4.0)
        pd_arr = np.array([0.01, 0.05, 0.5])
        np.testing.assert_allclose(
            simulator._default_thresholds(pd_arr), student_t.ppf(pd_arr, 4.0)
        )

    def test_t_copula_clips_extreme_probabilities(self, simulator):
        simulator._prep_copula("t", 4.0)
        assert np.isfinite(simulator._default_thresholds(np.array([0.0, 1.0]))).all()


class TestMixingNodes:
    def test_weights_sum_to_one_and_mean_is_one(self, simulator):
        simulator._prep_copula("t", 6.0)
        nodes, weights = simulator._mixing_nodes()
        assert weights.sum() == pytest.approx(1.0)
        assert (nodes > 0).all()
        # W ~ chi2(df) / df has unit mean for any degrees of freedom. The
        # tolerance is loose because W itself is unbounded in the upper tail,
        # where the nodes are sparse; the bounded conditional probabilities
        # these nodes actually integrate converge far faster.
        assert weights @ nodes == pytest.approx(1.0, rel=1e-3)

    def test_stable_at_high_degrees_of_freedom(self, simulator):
        simulator._prep_copula("t", 1_000.0)
        nodes, weights = simulator._mixing_nodes()
        assert np.isfinite(nodes).all() and np.isfinite(weights).all()
        assert weights @ nodes == pytest.approx(1.0, rel=1e-3)


class TestBuildFactorLoadings:
    def test_row_norms_are_one(self, simulator):
        lb = make_loan_book()
        fcov = make_factor_covariance()
        fc_hat = simulator._build_factor_loadings(lb, fcov)
        norms = np.linalg.norm(fc_hat, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-8)

    def test_shape_is_loans_by_factors(self, simulator):
        lb = make_loan_book(n=10)
        fcov = make_factor_covariance()
        fc_hat = simulator._build_factor_loadings(lb, fcov)
        assert fc_hat.shape == (10, fcov.n_factors)

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
        fcov = make_factor_covariance()
        with pytest.raises(ValueError, match="not a known factor"):
            simulator._build_factor_loadings(lb, fcov)


class TestSimulateLosses:
    def test_returns_loss_distribution_of_correct_length(self, simulator):
        lb = make_loan_book()
        fcov = make_factor_covariance()
        dist = simulator.simulate_losses(lb, fcov, n_scenarios=200, chunk=50, seed=1)
        assert isinstance(dist, LossDistribution)
        assert len(dist) == 200
        assert dist is simulator.distribution

    def test_notional_is_total_exposure(self, simulator):
        lb = make_loan_book()
        fcov = make_factor_covariance()
        dist = simulator.simulate_losses(lb, fcov, n_scenarios=100, chunk=50, seed=1)
        assert dist.notional == pytest.approx(lb.ead.sum())

    def test_carries_analytic_el(self, simulator):
        lb = make_loan_book()
        fcov = make_factor_covariance()
        dist = simulator.simulate_losses(lb, fcov, n_scenarios=100, chunk=50, seed=1)
        assert dist.analytic_el == pytest.approx(simulator.analytic_EL())

    def test_losses_are_non_negative_and_bounded_by_total_exposure(self, simulator):
        lb = make_loan_book()
        fcov = make_factor_covariance()
        dist = simulator.simulate_losses(lb, fcov, n_scenarios=200, chunk=50, seed=1)
        assert (dist.losses >= 0).all()
        assert (dist.losses <= lb.ead.sum()).all()

    def test_sets_loan_book_and_factor_covariance(self, simulator):
        lb = make_loan_book()
        fcov = make_factor_covariance()
        simulator.simulate_losses(lb, fcov, n_scenarios=100, chunk=50, seed=1)
        assert simulator.loan_book is lb
        assert simulator.factor_covariance is fcov

    def test_deterministic_with_same_seed(self):
        lb = make_loan_book()
        fcov = make_factor_covariance()
        dist_1 = DefaultSimulator().simulate_losses(lb, fcov, n_scenarios=100, chunk=30, seed=7)
        dist_2 = DefaultSimulator().simulate_losses(lb, fcov, n_scenarios=100, chunk=30, seed=7)
        np.testing.assert_array_equal(dist_1.losses, dist_2.losses)

    def test_uneven_chunk_size_still_fills_all_scenarios(self, simulator):
        lb = make_loan_book()
        fcov = make_factor_covariance()
        dist = simulator.simulate_losses(lb, fcov, n_scenarios=100, chunk=7, seed=3)
        assert len(dist) == 100
        assert np.isfinite(dist.losses).all()


class TestStudentTCopula:
    def test_records_copula_and_degrees_of_freedom(self, simulator):
        lb = make_loan_book()
        fcov = make_factor_covariance()
        simulator.simulate_losses(lb, fcov, n_scenarios=100, chunk=50, seed=1, copula="t", copula_df=4)
        assert simulator.copula == "t"
        assert simulator.copula_df == 4.0

    def test_gaussian_copula_discards_degrees_of_freedom(self, simulator):
        lb = make_loan_book()
        fcov = make_factor_covariance()
        simulator.simulate_losses(lb, fcov, n_scenarios=100, chunk=50, seed=1, copula_df=4)
        assert simulator.copula == "gaussian"
        assert simulator.copula_df is None

    def test_unknown_copula_raises(self, simulator):
        lb = make_loan_book()
        fcov = make_factor_covariance()
        with pytest.raises(ValueError, match="copula must be one of"):
            simulator.simulate_losses(lb, fcov, n_scenarios=100, copula="clayton")

    @pytest.mark.parametrize("bad_df", [0.0, -1.0, np.nan, np.inf])
    def test_non_positive_degrees_of_freedom_raises(self, simulator, bad_df):
        lb = make_loan_book()
        fcov = make_factor_covariance()
        with pytest.raises(ValueError, match="copula_df must be finite and positive"):
            simulator.simulate_losses(lb, fcov, n_scenarios=100, copula="t", copula_df=bad_df)

    def test_deterministic_with_same_seed(self):
        lb = make_loan_book()
        fcov = make_factor_covariance()
        kwargs = dict(n_scenarios=200, chunk=60, seed=7, copula="t", copula_df=4)
        dist_1 = DefaultSimulator().simulate_losses(lb, fcov, **kwargs)
        dist_2 = DefaultSimulator().simulate_losses(lb, fcov, **kwargs)
        np.testing.assert_array_equal(dist_1.losses, dist_2.losses)

    def test_losses_stay_bounded_by_total_exposure(self, simulator):
        lb = make_loan_book()
        fcov = make_factor_covariance()
        dist = simulator.simulate_losses(
            lb, fcov, n_scenarios=300, chunk=100, seed=1, copula="t", copula_df=4
        )
        assert (dist.losses >= 0).all()
        assert (dist.losses <= lb.ead.sum()).all()

    def test_marginal_default_rate_still_matches_pd(self):
        """The t copula redistributes defaults into the tail, it does not add any."""
        lb = make_loan_book(n=40)
        fcov = make_factor_covariance()
        sim = DefaultSimulator()
        dist = sim.simulate_losses(
            lb, fcov, n_scenarios=60_000, chunk=2_000, seed=11, copula="t", copula_df=4
        )
        assert dist.expected_loss == pytest.approx(sim.analytic_EL(), rel=0.05)

    def test_analytic_el_is_unchanged_by_the_copula(self):
        lb = make_loan_book()
        fcov = make_factor_covariance()
        gaussian = DefaultSimulator()
        gaussian.simulate_losses(lb, fcov, n_scenarios=100, chunk=100, seed=1)
        heavy_tailed = DefaultSimulator()
        heavy_tailed.simulate_losses(
            lb, fcov, n_scenarios=100, chunk=100, seed=1, copula="t", copula_df=3
        )
        assert heavy_tailed.analytic_EL() == pytest.approx(gaussian.analytic_EL())

    def test_analytic_el_matches_simulated_el_with_correlated_lgd(self):
        lb = make_loan_book(n=50)
        fcov = make_factor_covariance()
        sim = DefaultSimulator()
        dist = sim.simulate_losses(
            lb, fcov, n_scenarios=60_000, chunk=2_000, seed=11,
            copula="t", copula_df=4, QSQ=0.5,
        )
        assert dist.expected_loss == pytest.approx(sim.analytic_EL(), rel=0.05)

    def test_fattens_the_tail_at_equal_expected_loss(self):
        lb = make_loan_book(n=200, seed=3)
        fcov = make_factor_covariance()
        kwargs = dict(n_scenarios=40_000, chunk=2_000, seed=5)
        gaussian = DefaultSimulator().simulate_losses(lb, fcov, **kwargs)
        heavy_tailed = DefaultSimulator().simulate_losses(
            lb, fcov, copula="t", copula_df=3, **kwargs
        )
        assert heavy_tailed.var(0.999) > gaussian.var(0.999)

    def test_high_degrees_of_freedom_converges_on_the_gaussian_copula(self):
        lb = make_loan_book(n=200, seed=3)
        fcov = make_factor_covariance()
        kwargs = dict(n_scenarios=40_000, chunk=2_000, seed=5)
        gaussian = DefaultSimulator().simulate_losses(lb, fcov, **kwargs)
        nearly_gaussian = DefaultSimulator().simulate_losses(
            lb, fcov, copula="t", copula_df=5_000, **kwargs
        )
        assert nearly_gaussian.var(0.99) == pytest.approx(gaussian.var(0.99), rel=0.05)


class TestStochasticLGD:
    def test_qsq_none_leaves_lgd_deterministic(self, simulator):
        lb = make_loan_book()
        fcov = make_factor_covariance()
        simulator.simulate_losses(lb, fcov, n_scenarios=100, chunk=50, seed=1)
        assert simulator.qsq is None
        assert simulator.lgd_k is None

    def test_records_qsq_and_lgd_k(self, simulator):
        lb = make_loan_book()
        fcov = make_factor_covariance()
        simulator.simulate_losses(lb, fcov, n_scenarios=100, chunk=50, seed=1, QSQ=0.3, lgd_k=0.5)
        np.testing.assert_allclose(simulator.qsq, np.full(lb.n_loans, 0.3))
        assert simulator.lgd_k == 0.5

    def test_zero_lgd_k_reproduces_deterministic_lgd(self):
        lb = make_loan_book()
        fcov = make_factor_covariance()
        base = DefaultSimulator().simulate_losses(lb, fcov, n_scenarios=300, chunk=100, seed=4)
        degenerate = DefaultSimulator().simulate_losses(
            lb, fcov, n_scenarios=300, chunk=100, seed=4, QSQ=0.5, lgd_k=0.0
        )
        np.testing.assert_allclose(degenerate.losses, base.losses)

    def test_losses_stay_bounded_by_total_exposure(self, simulator):
        lb = make_loan_book()
        fcov = make_factor_covariance()
        dist = simulator.simulate_losses(
            lb, fcov, n_scenarios=300, chunk=100, seed=1, QSQ=0.5
        )
        assert (dist.losses >= 0).all()
        assert (dist.losses <= lb.ead.sum()).all()

    def test_deterministic_with_same_seed(self):
        lb = make_loan_book()
        fcov = make_factor_covariance()
        kwargs = dict(n_scenarios=200, chunk=60, seed=7, QSQ=0.4)
        dist_1 = DefaultSimulator().simulate_losses(lb, fcov, **kwargs)
        dist_2 = DefaultSimulator().simulate_losses(lb, fcov, **kwargs)
        np.testing.assert_array_equal(dist_1.losses, dist_2.losses)

    def test_correlated_lgd_raises_expected_loss(self):
        lb = make_loan_book()
        fcov = make_factor_covariance()
        uncorrelated = DefaultSimulator()
        uncorrelated.simulate_losses(lb, fcov, n_scenarios=100, chunk=100, seed=1, QSQ=0.0)
        correlated = DefaultSimulator()
        correlated.simulate_losses(lb, fcov, n_scenarios=100, chunk=100, seed=1, QSQ=0.6)
        naive = float(np.sum(lb.pd * lb.lgd * lb.ead))
        assert uncorrelated.analytic_EL() == pytest.approx(naive)
        assert correlated.analytic_EL() > naive

    def test_shocks_are_unchanged_by_the_copula_option(self):
        """The LGD stream is spawned, so it does not shift when the copula does."""
        lb = make_loan_book()
        fcov = make_factor_covariance()
        kwargs = dict(n_scenarios=200, chunk=60, seed=7, QSQ=0.4)
        without = DefaultSimulator().simulate_losses(lb, fcov, **kwargs)
        with_df = DefaultSimulator().simulate_losses(lb, fcov, copula_df=3.0, **kwargs)
        np.testing.assert_array_equal(without.losses, with_df.losses)

    def test_analytic_el_matches_simulated_el(self):
        lb = make_loan_book(n=50)
        fcov = make_factor_covariance()
        sim = DefaultSimulator()
        dist = sim.simulate_losses(lb, fcov, n_scenarios=40_000, chunk=2_000, seed=11, QSQ=0.5)
        assert dist.expected_loss == pytest.approx(sim.analytic_EL(), rel=0.05)

    def test_realised_lgd_keeps_its_input_mean(self, simulator):
        # Every loan defaults, so portfolio loss is purely realised LGD x EAD.
        data = make_loan_book(n=30).data.assign(pd=1.0)
        lb = LoanBook(data=data, factor_columns=["region", "sector"])
        fcov = make_factor_covariance()
        dist = simulator.simulate_losses(
            lb, fcov, n_scenarios=20_000, chunk=2_000, seed=2, QSQ=0.0
        )
        assert dist.expected_loss == pytest.approx(float(np.sum(lb.lgd * lb.ead)), rel=0.01)

    def test_per_loan_qsq_is_accepted(self, simulator):
        lb = make_loan_book()
        fcov = make_factor_covariance()
        qsq = np.linspace(0.0, 1.0, lb.n_loans)
        dist = simulator.simulate_losses(lb, fcov, n_scenarios=100, chunk=50, seed=1, QSQ=qsq)
        assert np.isfinite(dist.losses).all()

    def test_wrong_length_qsq_raises(self, simulator):
        lb = make_loan_book()
        fcov = make_factor_covariance()
        with pytest.raises(ValueError, match="1-D array of length"):
            simulator.simulate_losses(lb, fcov, n_scenarios=50, chunk=50, QSQ=np.zeros(3))

    def test_out_of_range_qsq_raises(self, simulator):
        lb = make_loan_book()
        fcov = make_factor_covariance()
        with pytest.raises(ValueError, match=r"within \[0, 1\]"):
            simulator.simulate_losses(lb, fcov, n_scenarios=50, chunk=50, QSQ=1.5)

    def test_negative_lgd_k_raises(self, simulator):
        lb = make_loan_book()
        fcov = make_factor_covariance()
        with pytest.raises(ValueError, match="lgd_k"):
            simulator.simulate_losses(lb, fcov, n_scenarios=50, chunk=50, QSQ=0.3, lgd_k=-0.1)


class TestAnalyticEL:
    def test_analytic_el_matches_manual_formula(self, simulator):
        lb = make_loan_book()
        fcov = make_factor_covariance()
        simulator.simulate_losses(lb, fcov, n_scenarios=50, chunk=50, seed=1)
        expected = float(np.sum(lb.pd * lb.lgd * lb.ead))
        assert simulator.analytic_EL() == pytest.approx(expected)
