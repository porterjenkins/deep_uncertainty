import torch

from deep_uncertainty.models import DoublePoissonHomoscedasticNN
from deep_uncertainty.models import DoublePoissonNN
from deep_uncertainty.models.ensembles.deep_regression_ensemble import DeepRegressionEnsemble
from deep_uncertainty.random_variables import DiscreteMixture
from deep_uncertainty.random_variables import DoublePoisson


class DoublePoissonMixtureNN(
    DeepRegressionEnsemble[DoublePoissonNN | DoublePoissonHomoscedasticNN]
):
    """An ensemble of Double Poisson Neural Nets that outputs a discrete probability distribution over [0, infinity] (truncated at 2000).

    The ensemble's predictions are combined as a mixture model with uniform weights.
    """

    max_value = 2000
    member_type = DoublePoissonNN | DoublePoissonHomoscedasticNN

    def _predict_impl(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Make a forward pass through the ensemble.

        Args:
            x (torch.Tensor): Batched input tensor, with shape (N, ...).

        Returns:
            torch.Tensor: Output probability tensor over [0, `max_value`], with shape (N, `self.max_value`).
            torch.Tensor: Aleatoric and epistemic uncertainties per input, with shape (N, 2).
        """
        dists = []
        means = []
        variances = []
        for member in self.members:
            mu, phi = torch.split(member._predict_impl(x), [1, 1], dim=-1)
            means.append(mu.flatten())
            variances.append((mu / phi).flatten())
            dist = DoublePoisson(mu=mu.flatten(), phi=phi.flatten())
            dists.append(dist)

        mixture = DiscreteMixture(distributions=dists, weights=torch.ones(len(dists)))
        probabilities = mixture.pmf(torch.arange(self.max_value).unsqueeze(1)).transpose(0, 1)

        means = torch.cat(means, dim=1)
        variances = torch.cat(variances, dim=1)
        aleatoric = variances.mean(dim=1)
        epistemic = means.var(dim=1)
        uncertainties = torch.stack([aleatoric, epistemic], dim=1)

        return probabilities, uncertainties

    def _update_test_metrics(self, y_hat: tuple[torch.Tensor, torch.Tensor], y: torch.Tensor):
        probabilities, uncertainties = y_hat
        N = probabilities.shape[0]
        preds = probabilities.argmax(dim=1)
        targets = y.flatten()
        target_probs = probabilities[torch.arange(N, device=self.device), targets.long()]

        support = torch.arange(self.max_value, device=self.device)
        mu = (probabilities * support).sum(dim=1)
        var = (y_hat * (support.unsqueeze(1) - mu).transpose(0, 1) ** 2).sum(dim=1)
        precision = 1 / var

        self.rmse.update(preds, targets)
        self.mae.update(preds, targets)
        self.nll.update(target_probs)
        self.mp.update(precision)
