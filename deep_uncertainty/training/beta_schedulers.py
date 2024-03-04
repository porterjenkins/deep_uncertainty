import math


class BetaScheduler:
    """Base class for all beta schedulers (used to modify beta-NLL loss functions during training).

    Attributes:
        beta_0 (float, optional): Initial value of beta (should be in [0, 1]). Defaults to 0.
        beta_1 (float, optional): Final value of beta (should be in [0, 1]). Defaults to 0.
        last_epoch (int, optional): Epoch at which beta should achieve `beta_1`. Defaults to 1000.
    """

    def __init__(self, beta_0: float = 0.0, beta_1: float = 0.0, last_epoch: int = 1000):

        if (beta_0 < 0 or beta_0 > 1) or (beta_1 < 0 or beta_1 > 1):
            raise ValueError(
                f"Both `beta_0` and `beta_1` should be in [0, 1]. Got {beta_0, beta_1}"
            )

        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.last_epoch = last_epoch

        self.current_value = beta_0
        self.current_epoch = 0

    def step(self):
        """Update internal counter for the current epoch, and the current value of beta."""
        self.current_epoch += 1
        self._update_current_value()

    def _update_current_value(self):
        """Update `self.current_value` according to the specified schedule."""
        raise NotImplementedError("Should be implemented by subclass.")


class CosineAnnealingBetaScheduler(BetaScheduler):
    """A Beta scheduler that uses cosine annealing to gradually change from `beta_0` to `beta_1`."""

    def _update_current_value(self):
        self.current_value = self.beta_1 + 0.5 * (self.beta_0 - self.beta_1) * (
            1 + math.cos((self.current_epoch * math.pi) / self.last_epoch)
        )


class LinearBetaScheduler(BetaScheduler):
    """A Beta scheduler that uses linear steps to gradually change from `beta_0` to `beta_1`."""

    def _update_current_value(self):
        self.current_value = self.beta_0 - self.current_epoch * (
            (self.beta_0 - self.beta_1) / self.last_epoch
        )
