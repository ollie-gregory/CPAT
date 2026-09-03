"""Default-only single-factor Gaussian copula Monte Carlo engine for credit portfolio loss."""

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.inputs.loan_book import LoanBook
from src.inputs.correlation_matrix import CorrelationMatrix

everyday_green = '#11B67A'
racing_green = '#024731'
heritage_green = '#006A4D'

MIN_VARIANCE = 1e-12
QUADRATURE_NODES = 64


class DefaultSimulator:
    """Monte Carlo engine for a single-factor Gaussian copula credit loss model.

    Simulates portfolio credit losses under a Vasicek/ASRF-style
    single-factor model: each loan defaults when a latent,
    factor-correlated asset value falls below a threshold implied by
    its probability of default, and systematic factors are correlated
    according to a `CorrelationMatrix`. Optionally, LGD can be made
    stochastic and correlated with default through a second latent
    driver sharing the same systematic shock (see the `QSQ` argument
    of `simulate_losses`). Call `simulate_losses` first; the
    summary-statistic and plotting methods then operate on the
    resulting loss distribution.

    Attributes
    ----------
    loan_book : LoanBook or None
        The loan book used in the most recent call to
        `simulate_losses`.
    correlation_matrix : CorrelationMatrix or None
        The correlation matrix used in the most recent call to
        `simulate_losses`.
    qsq : numpy.ndarray or None
        Per-loan LGD systematic R-squared used in the most recent call
        to `simulate_losses`; None when LGD was deterministic.
    lgd_k : float or None
        Latent LGD scale used in the most recent call to
        `simulate_losses`; None when LGD was deterministic.
    losses : pandas.Series or None
        Simulated portfolio loss for each scenario, set by
        `simulate_losses`.
    """

    def __init__(self):
        self.loan_book = None
        self.correlation_matrix = None
        self.qsq = None
        self.lgd_k = None
        self.losses = None

    def _require_simulated(self):
        if self.losses is None:
            raise RuntimeError("simulate_losses() must be called before accessing results")

    def _compute_thresholds(self, PD, eps=1e-12):
        PD_clipped = np.clip(PD, eps, 1.0 - eps)
        return norm.ppf(PD_clipped)

    def _prep_sim_vectors(self, PD, RSQ, LGD, EAD):
        thr = self._compute_thresholds(PD)[:, None]
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
        independent. Requires `self.qsq` to be set.
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

        pd_given_phi = norm.cdf((thr[:, None] - sqrt_rsq * nodes[None, :]) / sd_eps)
        lgd_given_phi = norm.cdf((a - k * sqrt_qsq * nodes[None, :]) / sd_eta)

        return lb.ead * ((pd_given_phi * lgd_given_phi) @ weights)

    def _build_factor_loadings(self, loan_book, correlation_matrix):
        M = loan_book.n_loans
        N = correlation_matrix.n_factors

        F = np.zeros((M, N), dtype=float)

        for column in loan_book.factor_columns:
            values = loan_book.factor_values(column)
            for i, value in enumerate(values):
                idx = correlation_matrix.factor_index.get(value)
                if idx is None:
                    raise ValueError(
                        f"LoanBook factor column '{column}' contains value {value!r} "
                        f"at row {i} that is not a known factor in the CorrelationMatrix "
                        f"(known factors: {sorted(correlation_matrix.factor_index)})"
                    )
                F[i, idx] = 1.0

        FC = F @ correlation_matrix.cholesky
        row_norms = np.linalg.norm(FC, axis=1, keepdims=True)
        if np.any(row_norms == 0):
            raise ValueError(
                "One or more loans have zero total factor loading; check that "
                "factor_columns values match the CorrelationMatrix's factors"
            )
        FC_hat = FC / row_norms
        return FC_hat

    def simulate_losses(
        self,
        loan_book: LoanBook,
        correlation_matrix: CorrelationMatrix,
        n_scenarios=100_000,
        chunk=1_000,
        seed=42,
        QSQ=None,
        lgd_k=1.0,
    ):
        """Simulate the portfolio loss distribution via Monte Carlo.

        Draws correlated systematic factor shocks (via
        `correlation_matrix`) and idiosyncratic shocks for every loan
        in `loan_book`, flags a loan as defaulted when its simulated
        asset value falls below its default threshold, and sums
        LGD-weighted exposure across defaults to get one portfolio
        loss per scenario. Each loan's asset value is

        ``X_i = phi_i * sqrt(RSQ_i) + eps_i * sqrt(1 - RSQ_i)``

        where ``phi_i`` is the loan's correlated systematic shock and
        ``eps_i`` its idiosyncratic shock.

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
        the marginal LGD.

        Parameters
        ----------
        loan_book : LoanBook
            The loan-level portfolio to simulate.
        correlation_matrix : CorrelationMatrix
            Systematic factor correlation structure. Must cover every
            category value referenced by `loan_book.factor_columns`.
        n_scenarios : int, default 100_000
            Number of Monte Carlo scenarios to simulate.
        chunk : int, default 1_000
            Number of scenarios simulated per batch, to bound peak
            memory.
        seed : int, default 42
            Seed for the random number generator.
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
        pandas.Series
            Simulated portfolio loss for each of `n_scenarios`
            scenarios. Also stored on `self.losses`.

        Raises
        ------
        ValueError
            If a loan's factor category isn't a known factor in
            `correlation_matrix`, a loan ends up with zero total
            factor loading, `QSQ` has the wrong length or falls
            outside [0, 1], or `lgd_k` is negative.
        """
        self.loan_book = loan_book
        self.correlation_matrix = correlation_matrix
        self.qsq = None
        self.lgd_k = None

        M = loan_book.n_loans
        N = correlation_matrix.n_factors

        FC_hat = self._build_factor_loadings(loan_book, correlation_matrix)
        thr, sqrt_rsq, weight = self._prep_sim_vectors(
            loan_book.pd, loan_book.rsq, loan_book.lgd, loan_book.ead
        )
        lgd_driver = self._prep_lgd_driver(QSQ, lgd_k, loan_book.lgd, M)
        ead = loan_book.ead[:, None]

        rng = np.random.default_rng(seed)
        # Spawned rather than drawn from `rng`, so the systematic and
        # default shocks are identical with and without a PD-LGD component
        # and the two runs stay directly comparable.
        lgd_rng = rng.spawn(1)[0]

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

        self.losses = pd.Series(losses_out)

        return self.losses

    def analytic_EL(self):
        """Closed-form expected loss of the loan book.

        With deterministic LGD this is simply ``sum(PD * LGD * EAD)``.
        When `simulate_losses` was given a `QSQ`, default and realised
        LGD are correlated, so the expectation of their product is
        taken by quadrature over the shared systematic shock instead —
        exceeding ``sum(PD * LGD * EAD)`` whenever ``QSQ > 0``.

        Returns
        -------
        float
            Expected portfolio loss across all loans in
            `self.loan_book`.
        """
        lb = self.loan_book
        if self.qsq is None:
            return float(np.sum(lb.pd * lb.lgd * lb.ead))
        return float(np.sum(self._expected_default_loss(self._compute_thresholds(lb.pd))))

    def var_quantile(self, alpha=0.99):
        """Value-at-Risk of the simulated loss distribution.

        Parameters
        ----------
        alpha : float, default 0.99
            Confidence level in (0, 1).

        Returns
        -------
        float
            The `alpha`-quantile of the simulated portfolio losses.

        Raises
        ------
        RuntimeError
            If called before `simulate_losses`.
        """
        self._require_simulated()
        return float(np.quantile(self.losses.to_numpy(), alpha))

    def expected_shortfall(self, alpha=0.99):
        """Expected shortfall (conditional VaR) of the simulated loss distribution.

        Parameters
        ----------
        alpha : float, default 0.99
            Confidence level in (0, 1).

        Returns
        -------
        float
            Mean simulated loss among scenarios at or beyond the
            `alpha`-quantile.

        Raises
        ------
        RuntimeError
            If called before `simulate_losses`.
        """
        self._require_simulated()
        arr = self.losses.to_numpy()
        q = np.quantile(arr, alpha)
        tail = arr[arr >= q]
        return float(tail.mean()) if len(tail) > 0 else float(q)

    def plot_loss_distribution(self):
        """Plot a histogram of simulated loss rates with EL/VaR markers.

        Returns
        -------
        matplotlib.figure.Figure
            Histogram of simulated loss rate (loss / total EAD),
            annotated with analytic EL, simulated EL, and 95%/99% VaR
            reference lines.

        Raises
        ------
        RuntimeError
            If called before `simulate_losses`.
        """
        self._require_simulated()

        arr = self.losses.to_numpy()
        denom = self.loan_book.ead.sum()

        fig, ax = plt.subplots(figsize=(6.75, 3.75), dpi=300)

        ax.hist(arr / denom, bins=60, color=everyday_green, edgecolor=None)

        ax.axvline(self.analytic_EL() / denom, color=racing_green, lw=2, label='Analytic EL')
        ax.axvline(arr.mean() / denom, color=racing_green, lw=2, ls=':', label='Simulated EL')
        ax.axvline(self.var_quantile(0.95) / denom, color=racing_green, lw=2, ls='--', label='VaR 95%')
        ax.axvline(self.var_quantile(0.99) / denom, color=racing_green, lw=2, ls='-.', label='VaR 99%')

        ax.set_title('Portfolio Loss Distribution')
        ax.set_xlabel('Loss Rate')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid()
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))
        fig.tight_layout()
        return fig
