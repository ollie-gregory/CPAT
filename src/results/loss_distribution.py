"""LossDistribution result type: a set of scenario losses and the statistics read off it."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.tools.palette import everyday_green, racing_green

VAR_LINE_STYLES = ("--", "-.", (0, (3, 1, 1, 1)))


@dataclass(frozen=True)
class LossDistribution:
    """A simulated loss distribution and the statistics derived from it.

    Wraps one loss per scenario together with the notional those
    losses are measured against, and owns every summary statistic and
    plot that a credit loss distribution supports. Any model or
    transformation that produces scenario losses - a Monte Carlo
    simulator over a whole portfolio, or a `Tranche` carving one band
    out of that portfolio - returns a `LossDistribution`, so the
    statistics are defined once here rather than reimplemented per
    producer.

    Parameters
    ----------
    losses : array_like
        Loss for each scenario, in currency units. Must be 1-D,
        non-empty and finite.
    notional : float
        The exposure these losses are measured against, used as the
        denominator for every loss rate: total EAD for a whole
        portfolio, or its size for a tranche. Must be finite and
        positive.
    name : str, default 'Portfolio'
        Label for this distribution, used in plot titles and as the
        name of the `summary` series.
    analytic_el : float, optional
        Closed-form expected loss, where the producing model can
        supply one. Drawn as a reference line by `plot` and included
        in `summary` when present; left as None otherwise (a tranche
        of a simulated distribution has no closed form).

    Attributes
    ----------
    losses : numpy.ndarray
        The scenario losses, as a 1-D float array.

    Raises
    ------
    ValueError
        If `losses` is not 1-D, is empty, or contains non-finite
        values, or if `notional` is not finite and positive.
    """

    losses: np.ndarray
    notional: float
    name: str = "Portfolio"
    analytic_el: float | None = None

    def __post_init__(self):
        losses = np.asarray(self.losses, dtype=float)

        if losses.ndim != 1:
            raise ValueError(f"LossDistribution.losses must be 1-D, got shape {losses.shape}")

        if losses.size == 0:
            raise ValueError("LossDistribution.losses must contain at least one scenario")

        if not np.isfinite(losses).all():
            raise ValueError("LossDistribution.losses must be finite")

        notional = float(self.notional)
        if not np.isfinite(notional) or notional <= 0.0:
            raise ValueError(
                f"LossDistribution.notional must be finite and positive, got {self.notional}"
            )

        object.__setattr__(self, "losses", losses)
        object.__setattr__(self, "notional", notional)

        if self.analytic_el is not None:
            object.__setattr__(self, "analytic_el", float(self.analytic_el))

    def __len__(self) -> int:
        return self.losses.size

    @property
    def n_scenarios(self) -> int:
        """int: Number of simulated scenarios."""
        return self.losses.size

    @property
    def loss_rates(self) -> np.ndarray:
        """numpy.ndarray: Scenario losses as a fraction of `notional`."""
        return self.losses / self.notional

    @property
    def expected_loss(self) -> float:
        """float: Mean loss across scenarios."""
        return float(self.losses.mean())

    @property
    def expected_loss_rate(self) -> float:
        """float: Mean loss as a fraction of `notional`."""
        return self.expected_loss / self.notional

    @property
    def unexpected_loss(self) -> float:
        """float: Standard deviation of loss across scenarios."""
        return float(self.losses.std())

    def to_series(self) -> pd.Series:
        """Return the scenario losses as a pandas Series.

        Returns
        -------
        pandas.Series
            One loss per scenario, named after this distribution.
        """
        return pd.Series(self.losses, name=self.name)

    def var(self, alpha=0.99) -> float:
        """Value-at-Risk of the loss distribution.

        Parameters
        ----------
        alpha : float, default 0.99
            Confidence level in (0, 1).

        Returns
        -------
        float
            The `alpha`-quantile of the scenario losses.
        """
        return float(np.quantile(self.losses, alpha))

    def expected_shortfall(self, alpha=0.99) -> float:
        """Expected shortfall (conditional VaR) of the loss distribution.

        Parameters
        ----------
        alpha : float, default 0.99
            Confidence level in (0, 1).

        Returns
        -------
        float
            Mean loss among scenarios at or beyond the `alpha`-quantile,
            falling back to the quantile itself when no scenario reaches
            it.
        """
        q = np.quantile(self.losses, alpha)
        tail = self.losses[self.losses >= q]
        return float(tail.mean()) if len(tail) > 0 else float(q)

    def summary(self, alphas=(0.95, 0.99)) -> pd.Series:
        """Headline statistics of the loss distribution.

        Parameters
        ----------
        alphas : sequence of float, default (0.95, 0.99)
            Confidence levels to report VaR and expected shortfall at.

        Returns
        -------
        pandas.Series
            Expected loss, unexpected loss, and VaR/ES at each level in
            `alphas`, each as both a currency amount and a rate on
            `notional`. Includes the analytic expected loss when
            `analytic_el` is set.
        """
        stats = {"EL": self.expected_loss}

        if self.analytic_el is not None:
            stats["Analytic EL"] = self.analytic_el

        stats["EL Rate"] = self.expected_loss_rate
        stats["UL"] = self.unexpected_loss

        for alpha in alphas:
            label = f"{alpha:.0%}"
            var = self.var(alpha)
            es = self.expected_shortfall(alpha)
            stats[f"VaR {label}"] = var
            stats[f"VaR {label} Rate"] = var / self.notional
            stats[f"ES {label}"] = es
            stats[f"ES {label} Rate"] = es / self.notional

        return pd.Series(stats, name=self.name)

    def plot(self, ax=None, bins=60, alphas=(0.95, 0.99), color=everyday_green, title=None):
        """Plot a histogram of loss rates with EL and VaR markers.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created when omitted;
            pass one to overlay distributions or to build a grid of
            panels.
        bins : int, default 60
            Number of histogram bins.
        alphas : sequence of float, default (0.95, 0.99)
            Confidence levels to mark with VaR reference lines.
        color : str, default '#11B67A'
            Fill colour of the histogram.
        title : str, optional
            Axes title. Defaults to ``'<name> Loss Distribution'``.

        Returns
        -------
        matplotlib.figure.Figure
            Histogram of loss rate (loss / `notional`), annotated with
            simulated EL, the analytic EL when one is set, and a VaR
            line per level in `alphas`.
        """
        owns_figure = ax is None
        if owns_figure:
            fig, ax = plt.subplots(figsize=(6.75, 3.75), dpi=300)
        else:
            fig = ax.figure

        ax.hist(self.loss_rates, bins=bins, color=color, edgecolor=None)

        if self.analytic_el is not None:
            ax.axvline(
                self.analytic_el / self.notional, color=racing_green, lw=2, label='Analytic EL'
            )

        ax.axvline(self.expected_loss_rate, color=racing_green, lw=2, ls=':', label='Simulated EL')

        for i, alpha in enumerate(alphas):
            ax.axvline(
                self.var(alpha) / self.notional,
                color=racing_green,
                lw=2,
                ls=VAR_LINE_STYLES[i % len(VAR_LINE_STYLES)],
                label=f'VaR {alpha:.0%}',
            )

        ax.set_title(f'{self.name} Loss Distribution' if title is None else title)
        ax.set_xlabel('Loss Rate')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid()
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))

        if owns_figure:
            fig.tight_layout()

        return fig
