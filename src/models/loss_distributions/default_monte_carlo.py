"""Default-only single-factor copula Monte Carlo engine for credit portfolio loss."""

import numpy as np
from scipy.special import ndtr
from scipy.stats import chi2, norm, t as student_t
from tqdm import tqdm

from src.inputs.loan_book import LoanBook
from src.inputs.factor_covariance_matrix import FactorCovarianceMatrix
from src.results.loss_distribution import LossDistribution

MIN_VARIANCE = 1e-12
QUADRATURE_NODES = 64
COPULAS = ('gaussian', 't')


class DefaultSimulator:
    """Monte Carlo engine for a single-factor copula credit loss model.

    Simulates portfolio credit losses under a Vasicek/ASRF-style
    single-factor model: each loan defaults when a latent,
    factor-correlated asset value falls below a threshold implied by
    its probability of default, and systematic factors are correlated
    according to a `FactorCovarianceMatrix`. The latent variables are
    joined by a Gaussian copula by default, or by a Student-t copula
    with heavier joint tails (see the `copula` argument of
    `simulate_losses`). Optionally, LGD can be made stochastic and
    correlated with default through a second latent driver sharing the
    same systematic shock (see the `QSQ` argument of
    `simulate_losses`). Call `simulate_losses` first; it returns a
    `LossDistribution`, which owns the summary statistics and plots of
    the resulting losses.

    Attributes
    ----------
    loan_book : LoanBook or None
        The loan book used in the most recent call to
        `simulate_losses`.
    factor_covariance : FactorCovarianceMatrix or None
        The factor covariance matrix used in the most recent call to
        `simulate_losses`.
    copula : str or None
        The copula used in the most recent call to `simulate_losses`,
        either ``'gaussian'`` or ``'t'``.
    copula_df : float or None
        Degrees of freedom of the Student-t copula used in the most
        recent call to `simulate_losses`; None under the Gaussian
        copula.
    qsq : numpy.ndarray or None
        Per-loan LGD systematic R-squared used in the most recent call
        to `simulate_losses`; None when LGD was deterministic.
    lgd_k : float or None
        Latent LGD scale used in the most recent call to
        `simulate_losses`; None when LGD was deterministic.
    distribution : LossDistribution or None
        The simulated portfolio loss distribution, set by
        `simulate_losses`.
    """

    def __init__(self):
        self.loan_book = None
        self.factor_covariance = None
        self.copula = None
        self.copula_df = None
        self.qsq = None
        self.lgd_k = None
        self.distribution = None

    def _require_simulated(self):
        if self.distribution is None:
            raise RuntimeError("simulate_losses() must be called before accessing results")

    def _compute_thresholds(self, PD, eps=1e-12):
        PD_clipped = np.clip(PD, eps, 1.0 - eps)
        return norm.ppf(PD_clipped)

    def _prep_copula(self, copula, copula_df):
        """Validate and record the copula choice.

        Sets `self.copula_df` to the Student-t degrees of freedom, or
        to None under the Gaussian copula, which is the flag the rest
        of the engine branches on.
        """
        if copula not in COPULAS:
            raise ValueError(f"copula must be one of {COPULAS}, got {copula!r}")

        self.copula = copula

        if copula == 'gaussian':
            self.copula_df = None
            return

        if not np.isfinite(copula_df) or copula_df <= 0.0:
            raise ValueError(
                f"copula_df must be finite and positive, got {copula_df}"
            )
        self.copula_df = float(copula_df)

    def _default_thresholds(self, PD, eps=1e-12):
        """Default triggers on the copula's own marginal scale.

        Gaussian latent variables are compared against the probit
        threshold directly; under the Student-t copula they are first
        divided by the square root of the shared mixing variable, so
        the trigger is the t quantile of `PD` instead.
        """
        if self.copula_df is None:
            return self._compute_thresholds(PD)
        return student_t.ppf(np.clip(PD, eps, 1.0 - eps), self.copula_df)

    def _mixing_nodes(self, n_nodes=QUADRATURE_NODES):
        """Quadrature nodes and weights for the t copula's mixing variable.

        The mixing variable ``W ~ chi2(copula_df) / copula_df`` is
        integrated on the probability scale — Gauss-Legendre nodes
        pushed through the chi-square quantile function — which stays
        numerically well behaved for any degrees of freedom, unlike
        the generalised Gauss-Laguerre weights this would otherwise
        call for.
        """
        u, weights = np.polynomial.legendre.leggauss(n_nodes)
        W = chi2.ppf(0.5 * (u + 1.0), self.copula_df) / self.copula_df
        return W, 0.5 * weights

    def _prep_sim_vectors(self, PD, RSQ, LGD, EAD):
        thr = self._default_thresholds(PD)[:, None]
        sqrt_rsq = np.sqrt(np.clip(RSQ, 0.0, 1.0))[:, None]
        weight = (LGD * EAD)[:, None]
        return thr, sqrt_rsq, weight

    def _prep_lgd_driver(self, QSQ, lgd_k, LGD, n_loans):
        """Validate the LGD-driver inputs and return its simulation vectors.

        Returns None when `QSQ` is None, meaning LGD stays
        deterministic. Otherwise returns ``(sqrt_qsq, sqrt_1m_qsq,
        a)``, each a column vector of length `n_loans`, where `a` is
        the probit intercept chosen so that the realised LGD
        ``Phi(a - lgd_k * V)`` has mean equal to the loan's input LGD.
        """
        if QSQ is None:
            return None

        qsq = np.asarray(QSQ, dtype=float)
        if qsq.ndim == 0:
            qsq = np.full(n_loans, float(qsq))
        elif qsq.ndim != 1 or qsq.shape[0] != n_loans:
            raise ValueError(
                f"QSQ must be a scalar or a 1-D array of length {n_loans}, "
                f"got shape {qsq.shape}"
            )

        if not np.isfinite(qsq).all() or ((qsq < 0.0) | (qsq > 1.0)).any():
            raise ValueError("QSQ must be finite and within [0, 1]")

        if not np.isfinite(lgd_k) or lgd_k < 0.0:
            raise ValueError(f"lgd_k must be finite and non-negative, got {lgd_k}")

        self.qsq = qsq
        self.lgd_k = float(lgd_k)

        sqrt_qsq = np.sqrt(qsq)[:, None]
        sqrt_1m_qsq = np.sqrt(1.0 - qsq)[:, None]
        a = np.sqrt(1.0 + lgd_k ** 2) * self._compute_thresholds(LGD)[:, None]
        return sqrt_qsq, sqrt_1m_qsq, a

    def _expected_default_loss(self, thr, n_nodes=QUADRATURE_NODES):
        """Per-loan expected default loss under a correlated stochastic LGD.

        Evaluates ``EAD * E[1{X <= thr} * LGD_realised]`` by
        Gauss-Hermite quadrature over the shared systematic shock, on
        which the default event and the realised LGD are conditionally
        independent. Under the Student-t copula the conditional
        default probability is additionally averaged over the mixing
        variable, which the recovery driver does not share. Requires
        `self.qsq` to be set.
        """
        lb = self.loan_book
        k = self.lgd_k

        rsq = np.clip(lb.rsq, 0.0, 1.0)
        sd_eps = np.sqrt(np.clip(1.0 - rsq, MIN_VARIANCE, None))[:, None]
        sqrt_rsq = np.sqrt(rsq)[:, None]

        qsq = np.clip(self.qsq, 0.0, 1.0)
        a = np.sqrt(1.0 + k ** 2) * self._compute_thresholds(lb.lgd)[:, None]
        sd_eta = np.sqrt(1.0 + k ** 2 * (1.0 - qsq))[:, None]
        sqrt_qsq = np.sqrt(qsq)[:, None]

        nodes, weights = np.polynomial.hermite_e.hermegauss(n_nodes)
        weights = weights / np.sqrt(2.0 * np.pi)

        pd_given_phi = self._pd_given_phi(thr, sqrt_rsq, sd_eps, nodes)
        lgd_given_phi = norm.cdf((a - k * sqrt_qsq * nodes[None, :]) / sd_eta)

        return lb.ead * ((pd_given_phi * lgd_given_phi) @ weights)

    def _pd_given_phi(self, thr, sqrt_rsq, sd_eps, nodes):
        """Default probability at each systematic-shock quadrature node.

        Under the t copula the trigger scales with the square root of
        the mixing variable, so the conditional default probability is
        integrated over it. The mixing nodes are accumulated one at a
        time rather than broadcast into an ``(M, phi, W)`` array,
        which for a large loan book would not fit in memory.
        """
        shocks = sqrt_rsq * nodes[None, :]

        if self.copula_df is None:
            return norm.cdf((thr[:, None] - shocks) / sd_eps)

        W, weights = self._mixing_nodes()
        scaled = np.empty_like(shocks)
        pd_given_phi = np.zeros_like(shocks)
        for W_node, weight in zip(W, weights):
            np.multiply(thr[:, None], np.sqrt(W_node), out=scaled)
            scaled -= shocks
            scaled /= sd_eps
            pd_given_phi += weight * norm.cdf(scaled)
        return pd_given_phi

    def _build_factor_loadings(self, loan_book, factor_covariance):
        M = loan_book.n_loans
        N = factor_covariance.n_factors

        F = np.zeros((M, N), dtype=float)

        for column in loan_book.factor_columns:
            values = loan_book.factor_values(column)
            for i, value in enumerate(values):
                idx = factor_covariance.factor_index.get(value)
                if idx is None:
                    raise ValueError(
                        f"LoanBook factor column '{column}' contains value {value!r} "
                        f"at row {i} that is not a known factor in the FactorCovarianceMatrix "
                        f"(known factors: {sorted(factor_covariance.factor_index)})"
                    )
                F[i, idx] = 1.0

        FC = F @ factor_covariance.cholesky
        row_norms = np.linalg.norm(FC, axis=1, keepdims=True)
        if np.any(row_norms == 0):
            raise ValueError(
                "One or more loans have zero total factor loading; check that "
                "factor_columns values match the FactorCovarianceMatrix's factors"
            )
        FC_hat = FC / row_norms
        return FC_hat

    def simulate_losses(
        self,
        loan_book: LoanBook,
        factor_covariance: FactorCovarianceMatrix,
        n_scenarios=100_000,
        chunk=1_000,
        seed=42,
        copula='gaussian',
        copula_df=5.0,
        QSQ=None,
        lgd_k=1.0,
    ):
        """Simulate the portfolio loss distribution via Monte Carlo.

        Draws correlated systematic factor shocks (via
        `factor_covariance`) and idiosyncratic shocks for every loan
        in `loan_book`, flags a loan as defaulted when its simulated
        asset value falls below its default threshold, and sums
        LGD-weighted exposure across defaults to get one portfolio
        loss per scenario. Each loan's asset value is

        ``X_i = phi_i * sqrt(RSQ_i) + eps_i * sqrt(1 - RSQ_i)``

        where ``phi_i`` is the loan's correlated systematic shock and
        ``eps_i`` its idiosyncratic shock.

        Setting ``copula='t'`` swaps the Gaussian copula joining these
        latent variables for a Student-t copula. A single chi-square
        mixing variable ``W ~ chi2(copula_df) / copula_df`` is drawn
        per scenario and shared by every loan, so the latent vector
        becomes

        ``T_i = X_i / sqrt(W)``

        which is multivariate t, and a loan defaults when ``T_i`` falls
        below the t quantile of its PD. Marginal default probabilities
        are unchanged, but the shared ``W`` gives the copula non-zero
        lower tail dependence: the probability that one loan defaults
        given another has, taken ever further into the tail, tends to
        a positive limit rather than to zero as it does under the
        Gaussian copula. Defaults therefore cluster far harder in
        extreme scenarios at the same `rsq`, which lifts VaR and
        expected shortfall while leaving expected loss alone. Lower
        `copula_df` means heavier joint tails; as it grows the model
        converges back on the Gaussian copula.

        Passing `QSQ` additionally makes LGD stochastic and correlated
        with default. A second latent variable, the recovery driver,
        is built from the same systematic shock with its own
        systematic weight and its own idiosyncratic noise
        ``eta_i``, independent of ``eps_i``:

        ``V_i = phi_i * sqrt(QSQ_i) + eta_i * sqrt(1 - QSQ_i)``

        and the realised LGD of a defaulting loan is the probit
        transform ``LGD_i = Phi(a_i - lgd_k * V_i)``, with ``a_i``
        chosen so that ``E[LGD_i]`` equals the loan book's `lgd`. A
        bad systematic outcome (low ``phi``) therefore drives defaults
        and recoveries together, fattening the tail without shifting
        the marginal LGD. The recovery driver stays Gaussian and does
        not share the t copula's mixing variable, so the wrong-way
        risk between defaults and recoveries always runs through the
        systematic shock, and ``QSQ = 0`` leaves LGD independent of
        default under either copula.

        Parameters
        ----------
        loan_book : LoanBook
            The loan-level portfolio to simulate.
        factor_covariance : FactorCovarianceMatrix
            Covariance structure over the systematic factors. Must
            cover every category value referenced by
            `loan_book.factor_columns`. Factor variances set the
            relative weight of each factor a loan loads on, so this
            is not interchangeable with the corresponding correlation
            matrix.
        n_scenarios : int, default 100_000
            Number of Monte Carlo scenarios to simulate.
        chunk : int, default 1_000
            Number of scenarios simulated per batch, to bound peak
            memory.
        seed : int, default 42
            Seed for the random number generator.
        copula : {'gaussian', 't'}, default 'gaussian'
            Copula joining the latent asset values. ``'t'`` gives a
            Student-t copula with `copula_df` degrees of freedom,
            which adds tail dependence without changing any loan's
            marginal PD.
        copula_df : float, default 5.0
            Degrees of freedom of the Student-t copula; must be
            positive. Values in the 3-10 range are typical for credit
            portfolios, with lower values giving heavier joint tails.
            Ignored when `copula` is ``'gaussian'``.
        QSQ : float or array_like of float, optional
            The LGD equivalent of `rsq`: the share of the recovery
            driver's variance explained by the systematic shock, in
            [0, 1], as a scalar applied to every loan or one value per
            loan. When omitted (the default) LGD stays deterministic
            at the loan book's `lgd` and the simulation is identical
            to the model without a PD-LGD component. ``QSQ = 0`` keeps
            LGD stochastic but uncorrelated with default.
        lgd_k : float, default 1.0
            Latent scale of the probit LGD transform, controlling how
            dispersed realised LGD is around its mean. ``lgd_k = 0``
            collapses it to the deterministic `lgd`; larger values
            push realised LGD towards 0 and 1. Ignored when `QSQ` is
            None.

        Returns
        -------
        LossDistribution
            Simulated portfolio loss for each of `n_scenarios`
            scenarios, carrying the loan book's total EAD as its
            notional and the analytic expected loss as its reference
            EL. Also stored on `self.distribution`; see
            `LossDistribution` for the statistics and plots available
            on it.

        Raises
        ------
        ValueError
            If a loan's factor category isn't a known factor in
            `factor_covariance`, a loan ends up with zero total
            factor loading, `copula` is not a recognised name,
            `copula_df` is not positive, `QSQ` has the wrong length or
            falls outside [0, 1], or `lgd_k` is negative.
        """
        self.loan_book = loan_book
        self.factor_covariance = factor_covariance
        self.copula = None
        self.copula_df = None
        self.qsq = None
        self.lgd_k = None
        self.distribution = None

        M = loan_book.n_loans
        N = factor_covariance.n_factors

        self._prep_copula(copula, copula_df)
        FC_hat = self._build_factor_loadings(loan_book, factor_covariance)
        thr, sqrt_rsq, weight = self._prep_sim_vectors(
            loan_book.pd, loan_book.rsq, loan_book.lgd, loan_book.ead
        )
        lgd_driver = self._prep_lgd_driver(QSQ, lgd_k, loan_book.lgd, M)
        ead = loan_book.ead[:, None]

        rng = np.random.default_rng(seed)
        # Spawned rather than drawn from `rng`, so the systematic and
        # default shocks are identical with and without a PD-LGD component
        # or a t copula, and the runs stay directly comparable.
        lgd_rng, mixing_rng = rng.spawn(2)

        losses_out = np.empty(n_scenarios)

        filled = 0

        with tqdm(total=n_scenarios, desc='Running Simulations') as pbar:
            while filled < n_scenarios:
                s = min(chunk, n_scenarios - filled)
                sl = slice(filled, filled + s)

                Z = rng.standard_normal(size=(N, s))

                Y = FC_hat @ Z
                Xi = rng.standard_normal(size=(M, s))
                X = sqrt_rsq * Y + np.sqrt(1.0 - sqrt_rsq ** 2) * Xi

                if self.copula_df is not None:
                    # One chi-square draw per scenario, shared by every loan,
                    # is what turns the latent vector into a multivariate t:
                    # a small draw scales the whole portfolio towards its
                    # thresholds at once. Applied in place, since a full
                    # (M, s) copy of the scaled triggers is the largest
                    # avoidable allocation in the loop.
                    X *= np.sqrt(
                        self.copula_df
                        / mixing_rng.chisquare(self.copula_df, size=(1, s))
                    )

                defaults = X <= thr

                if lgd_driver is None:
                    losses_out[sl] = (defaults * weight).sum(axis=0)
                else:
                    sqrt_qsq, sqrt_1m_qsq, a = lgd_driver
                    eta = lgd_rng.standard_normal(size=(M, s))
                    V = sqrt_qsq * Y + sqrt_1m_qsq * eta
                    # The probit transform is only evaluated where it is
                    # actually needed: recoveries are realised on defaults.
                    losses = np.zeros_like(V)
                    losses[defaults] = ndtr(
                        np.broadcast_to(a, V.shape)[defaults] - lgd_k * V[defaults]
                    )
                    losses_out[sl] = (losses * ead).sum(axis=0)

                filled += s
                pbar.update(s)

        self.distribution = LossDistribution(
            losses=losses_out,
            notional=float(loan_book.ead.sum()),
            name='Portfolio',
            analytic_el=self._analytic_el(),
        )

        return self.distribution

    def _analytic_el(self):
        lb = self.loan_book
        if self.qsq is None:
            return float(np.sum(lb.pd * lb.lgd * lb.ead))
        return float(np.sum(self._expected_default_loss(self._default_thresholds(lb.pd))))

    def analytic_EL(self):
        """Closed-form expected loss of the loan book.

        With deterministic LGD this is simply ``sum(PD * LGD * EAD)``,
        under either copula: the t copula changes how defaults cluster,
        not any loan's marginal default probability. When
        `simulate_losses` was given a `QSQ`, default and realised LGD
        are correlated, so the expectation of their product is taken by
        quadrature over the shared systematic shock instead — and over
        the t copula's mixing variable as well — exceeding
        ``sum(PD * LGD * EAD)`` whenever ``QSQ > 0``.

        Returns
        -------
        float
            Expected portfolio loss across all loans in
            `self.loan_book`.

        Raises
        ------
        RuntimeError
            If called before `simulate_losses`.
        """
        self._require_simulated()
        return self._analytic_el()
