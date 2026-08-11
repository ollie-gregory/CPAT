"""Single-factor Gaussian copula Monte Carlo engine for credit portfolio loss."""

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.inputs.loan_book import LoanBook
from src.inputs.correlation_matrix import CorrelationMatrix

everyday_green = '#11B67A'
racing_green = '#024731'
heritage_green = '#006A4D'


class Simulator:
    """Monte Carlo engine for a single-factor Gaussian copula credit loss model.

    Simulates portfolio credit losses under a Vasicek/ASRF-style
    single-factor model: each loan defaults when a latent,
    factor-correlated asset value falls below a threshold implied by
    its probability of default, and systematic factors are correlated
    according to a `CorrelationMatrix`. Call `simulate_losses` first;
    the summary-statistic and plotting methods then operate on the
    resulting loss distribution.

    Attributes
    ----------
    loan_book : LoanBook or None
        The loan book used in the most recent call to
        `simulate_losses`.
    correlation_matrix : CorrelationMatrix or None
        The correlation matrix used in the most recent call to
        `simulate_losses`.
    losses : pandas.Series or None
        Simulated portfolio loss for each scenario, set by
        `simulate_losses`.
    """

    def __init__(self):
        self.loan_book = None
        self.correlation_matrix = None
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
    ):
        """Simulate the portfolio loss distribution via Monte Carlo.

        Draws correlated systematic factor shocks (via
        `correlation_matrix`) and idiosyncratic shocks for every loan
        in `loan_book`, flags a loan as defaulted when its simulated
        asset value falls below its default threshold, and sums
        LGD-weighted exposure across defaults to get one portfolio
        loss per scenario.

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

        Returns
        -------
        pandas.Series
            Simulated portfolio loss for each of `n_scenarios`
            scenarios. Also stored on `self.losses`.

        Raises
        ------
        ValueError
            If a loan's factor category isn't a known factor in
            `correlation_matrix`, or a loan ends up with zero total
            factor loading.
        """
        self.loan_book = loan_book
        self.correlation_matrix = correlation_matrix

        M = loan_book.n_loans
        N = correlation_matrix.n_factors

        FC_hat = self._build_factor_loadings(loan_book, correlation_matrix)
        thr, sqrt_rsq, weight = self._prep_sim_vectors(
            loan_book.pd, loan_book.rsq, loan_book.lgd, loan_book.ead
        )

        rng = np.random.default_rng(seed)

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

                losses_out[sl] = (defaults * weight).sum(axis=0)

                filled += s
                pbar.update(s)

        self.losses = pd.Series(losses_out)

        return self.losses

    def analytic_EL(self):
        """Closed-form expected loss of the loan book.

        Returns
        -------
        float
            ``sum(PD * LGD * EAD)`` across all loans in
            `self.loan_book`.
        """
        lb = self.loan_book
        return float(np.sum(lb.pd * lb.lgd * lb.ead))

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
