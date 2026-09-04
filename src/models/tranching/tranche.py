"""Tranche: one attachment-to-detachment slice of a portfolio loss distribution."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from src.results.loss_distribution import LossDistribution


@dataclass(frozen=True)
class Tranche:
    """One loss band of a portfolio, carved between two subordination points.

    A tranche absorbs portfolio losses only once they exceed its
    attachment point, and only up to its detachment point, so its
    scenario losses are the portfolio's losses shifted down by the
    attachment amount and capped at the tranche's size. That makes a
    tranche a transformation from one `LossDistribution` to another:
    the statistics and plots come from `distribution`, exactly as they
    do for the portfolio it was carved from, while this class adds
    only what is specific to a tranche.

    Parameters
    ----------
    portfolio : LossDistribution
        The pool's loss distribution, whose `notional` sets what the
        attachment and detachment fractions are measured against.
    attachment : float
        Lower subordination point as a fraction of pool notional, in
        [0, 1). Losses below it are absorbed by more junior tranches.
    detachment : float
        Upper subordination point as a fraction of pool notional, in
        (`attachment`, 1]. Losses above it pass to more senior
        tranches.
    name : str, optional
        Label for the tranche, used in plot titles and summary tables.
        Defaults to its subordination points, e.g. ``'3%-7%'``.

    Raises
    ------
    ValueError
        If `attachment` or `detachment` is not finite, or they do not
        satisfy ``0 <= attachment < detachment <= 1``.
    """

    portfolio: LossDistribution
    attachment: float
    detachment: float
    name: str = ""

    def __post_init__(self):
        attachment = float(self.attachment)
        detachment = float(self.detachment)

        if not np.isfinite([attachment, detachment]).all():
            raise ValueError(
                f"Tranche attachment and detachment must be finite, "
                f"got ({self.attachment}, {self.detachment})"
            )

        if not 0.0 <= attachment < detachment <= 1.0:
            raise ValueError(
                f"Tranche points must satisfy 0 <= attachment < detachment <= 1, "
                f"got attachment={attachment}, detachment={detachment}"
            )

        object.__setattr__(self, "attachment", attachment)
        object.__setattr__(self, "detachment", detachment)

        if not self.name:
            object.__setattr__(self, "name", f"{attachment:.0%}-{detachment:.0%}")

    @property
    def attachment_amount(self) -> float:
        """float: Attachment point in currency units."""
        return self.attachment * self.portfolio.notional

    @property
    def detachment_amount(self) -> float:
        """float: Detachment point in currency units."""
        return self.detachment * self.portfolio.notional

    @property
    def thickness(self) -> float:
        """float: Tranche width as a fraction of pool notional."""
        return self.detachment - self.attachment

    @property
    def size(self) -> float:
        """float: Tranche width in currency units, the most it can lose."""
        return self.thickness * self.portfolio.notional

    @cached_property
    def distribution(self) -> LossDistribution:
        """LossDistribution: The tranche's own losses, with its size as notional.

        Portfolio losses shifted down by `attachment_amount` and
        capped at `size`, so loss rates are measured against what the
        tranche itself has at risk. Carries no analytic EL: unlike the
        portfolio's, a tranche's expected loss has no closed form here.
        """
        return LossDistribution(
            losses=np.clip(self.portfolio.losses - self.attachment_amount, 0.0, self.size),
            notional=self.size,
            name=self.name,
        )

    @property
    def prob_attachment(self) -> float:
        """float: Share of scenarios in which the tranche takes any loss."""
        return float((self.portfolio.losses > self.attachment_amount).mean())

    @property
    def prob_wipeout(self) -> float:
        """float: Share of scenarios in which the tranche is written down in full."""
        return float((self.portfolio.losses >= self.detachment_amount).mean())

    def summary(self, alphas=(0.95, 0.99)) -> pd.Series:
        """Headline statistics of the tranche.

        Parameters
        ----------
        alphas : sequence of float, default (0.95, 0.99)
            Confidence levels to report VaR and expected shortfall at.

        Returns
        -------
        pandas.Series
            The tranche's subordination points and size, every
            statistic from `LossDistribution.summary`, and the
            probabilities of attachment and of full write-down.
        """
        points = pd.Series(
            {
                "Attachment": self.attachment,
                "Detachment": self.detachment,
                "Size": self.size,
            },
            name=self.name,
        )
        probabilities = pd.Series(
            {
                "P(Attachment)": self.prob_attachment,
                "P(Wipeout)": self.prob_wipeout,
            },
            name=self.name,
        )
        return pd.concat([points, self.distribution.summary(alphas), probabilities])
