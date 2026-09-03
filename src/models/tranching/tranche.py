class Tranche:
    def __init__(self, portfolio_losses, attachment, detachment):
        self.portfolio_losses = portfolio_losses
        self.attachment = attachment
        self.detachment = detachment

    @property
    def losses(self):
        size = self.detachment - self.attachment
        return np.clip(self.portfolio_losses - self.attachment, 0, size)

    @property
    def expected_loss(self):
        return self.losses.mean()