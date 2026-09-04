"""CapitalStructure: a stack of tranches tiling a portfolio loss distribution."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.results.loss_distribution import LossDistribution
from src.models.tranching.tranche import Tranche
from src.tools.palette import everyday_green, racing_green


@dataclass
class CapitalStructure:
    """A full stack of tranches covering a portfolio from 0% to 100%.

    Built from the interior subordination points of the structure, so
    the tranches tile the pool exactly: every currency of portfolio
    loss lands in one and only one tranche, and tranche losses sum
    back to the portfolio loss in every scenario. Comparing the
    tranches shows how subordination redistributes the same losses -
    the junior tranche absorbing a high expected loss rate so the
    senior one can carry almost none.

    Parameters
    ----------
    portfolio : LossDistribution
        The pool's loss distribution to carve up.
    cut_points : sequence of float
        Interior subordination points as fractions of pool notional,
        strictly increasing and strictly inside (0, 1). ``[0.03,
        0.07]`` builds three tranches: 0-3%, 3-7% and 7-100%.
    names : sequence of str, optional
        Labels for the tranches, from most junior to most senior. Must
        have one more entry than `cut_points`. Defaults to each
        tranche's subordination points.

    Attributes
    ----------
    tranches : list of Tranche
        The tranches, ordered from most junior to most senior.

    Raises
    ------
    ValueError
        If `cut_points` is empty, not strictly increasing, or contains
        values outside (0, 1), or if `names` has the wrong length.
    """

    portfolio: LossDistribution
    cut_points: Sequence[float]
    names: Sequence[str] | None = None

    tranches: list[Tranche] = field(init=False, repr=False)

    def __post_init__(self):
        cut_points = [float(c) for c in self.cut_points]

        if not cut_points:
            raise ValueError("CapitalStructure.cut_points must contain at least one point")

        if not all(0.0 < c < 1.0 for c in cut_points):
            raise ValueError(
                f"CapitalStructure.cut_points must lie strictly within (0, 1), got {cut_points}"
            )

        if any(b <= a for a, b in zip(cut_points, cut_points[1:])):
            raise ValueError(
                f"CapitalStructure.cut_points must be strictly increasing, got {cut_points}"
            )

        n_tranches = len(cut_points) + 1

        if self.names is None:
            names = [""] * n_tranches
        else:
            names = list(self.names)
            if len(names) != n_tranches:
                raise ValueError(
                    f"CapitalStructure.names must have {n_tranches} entries for "
                    f"{len(cut_points)} cut point(s), got {len(names)}"
                )

        self.cut_points = cut_points

        boundaries = [0.0, *cut_points, 1.0]
        self.tranches = [
            Tranche(self.portfolio, attachment, detachment, name)
            for attachment, detachment, name in zip(boundaries, boundaries[1:], names)
        ]

    def __len__(self) -> int:
        return len(self.tranches)

    def __iter__(self):
        return iter(self.tranches)

    def __getitem__(self, key) -> Tranche:
        """Return a tranche by position or by name.

        Parameters
        ----------
        key : int or str
            Position in the stack (0 is the most junior), or a
            tranche's `name`.

        Returns
        -------
        Tranche

        Raises
        ------
        KeyError
            If `key` is a name no tranche in the stack carries.
        """
        if isinstance(key, str):
            for tranche in self.tranches:
                if tranche.name == key:
                    return tranche
            raise KeyError(
                f"no tranche named {key!r} (available: {[t.name for t in self.tranches]})"
            )
        return self.tranches[key]

    def summary(self, alphas=(0.95, 0.99)) -> pd.DataFrame:
        """Headline statistics of every tranche in the stack.

        Parameters
        ----------
        alphas : sequence of float, default (0.95, 0.99)
            Confidence levels to report VaR and expected shortfall at.

        Returns
        -------
        pandas.DataFrame
            One row per tranche, junior first, with the columns of
            `Tranche.summary`.
        """
        return pd.DataFrame([tranche.summary(alphas) for tranche in self.tranches])

    def plot_expected_loss(self, ax=None, color=everyday_green, title=None):
        """Plot each tranche's expected loss rate as a horizontal bar chart.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created when omitted.
        color : str, default '#11B67A'
            Bar fill colour.
        title : str, default 'Expected Loss by Tranche'
            Axes title.

        Returns
        -------
        matplotlib.figure.Figure
            Expected loss as a fraction of each tranche's own size,
            junior at the top, with the pool's expected loss rate
            marked for reference.
        """
        owns_figure = ax is None
        if owns_figure:
            fig, ax = plt.subplots(figsize=(6.75, 3.75), dpi=300)
        else:
            fig = ax.figure

        names = [tranche.name for tranche in self.tranches]
        rates = [tranche.distribution.expected_loss_rate for tranche in self.tranches]
        positions = np.arange(len(names))

        ax.barh(positions, rates, color=color)
        ax.axvline(
            self.portfolio.expected_loss_rate,
            color=racing_green,
            lw=2,
            ls=':',
            label='Portfolio EL',
        )

        ax.set_yticks(positions, names)
        ax.invert_yaxis()
        ax.set_title('Expected Loss by Tranche' if title is None else title)
        ax.set_xlabel('Expected Loss Rate')
        ax.set_ylabel('Tranche')
        ax.legend()
        ax.grid(axis='x')
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))

        if owns_figure:
            fig.tight_layout()

        return fig

    def plot_distributions(self, bins=60, alphas=(0.95, 0.99), color=everyday_green):
        """Plot every tranche's loss distribution as a column of panels.

        Each panel is `LossDistribution.plot` on the tranche's own
        distribution, so the panels are drawn by the same code as the
        portfolio's own histogram.

        Parameters
        ----------
        bins : int, default 60
            Number of histogram bins per panel.
        alphas : sequence of float, default (0.95, 0.99)
            Confidence levels to mark with VaR reference lines.
        color : str, default '#11B67A'
            Fill colour of the histograms.

        Returns
        -------
        matplotlib.figure.Figure
            One panel per tranche, junior first, each showing loss as
            a fraction of that tranche's own size.
        """
        n = len(self.tranches)
        fig, axes = plt.subplots(n, 1, figsize=(6.75, 2.75 * n), dpi=300, squeeze=False)

        for tranche, ax in zip(self.tranches, axes[:, 0]):
            tranche.distribution.plot(ax=ax, bins=bins, alphas=alphas, color=color)

        fig.tight_layout()
        return fig
