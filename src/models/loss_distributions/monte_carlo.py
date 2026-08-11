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
        lb = self.loan_book
        return float(np.sum(lb.pd * lb.lgd * lb.ead))

    def var_quantile(self, alpha=0.99):
        self._require_simulated()
        return float(np.quantile(self.losses.to_numpy(), alpha))

    def expected_shortfall(self, alpha=0.99):
        self._require_simulated()
        arr = self.losses.to_numpy()
        q = np.quantile(arr, alpha)
        tail = arr[arr >= q]
        return float(tail.mean()) if len(tail) > 0 else float(q)

    def plot_loss_distribution(self):
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
