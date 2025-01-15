import torch

from deep_uncertainty.models import NegBinomNN
from deep_uncertainty.models.ensembles.deep_regression_ensemble import DeepRegressionEnsemble
from deep_uncertainty.random_variables import DiscreteMixture


class NegBinomMixtureNN(DeepRegressionEnsemble[NegBinomNN]):
    """An ensemble of NegBinom Neural Nets that outputs a discrete probability distribution over [0, infinity] (truncated at 2000).

    The ensemble's predictions are combined as a mixture model with uniform weights.
    """

    member_type = NegBinomNN
    max_value = 2000

    def _predict_impl(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Make a forward pass through the ensemble.

        Args:
            x (torch.Tensor): Batched input tensor, with shape (N, ...).

        Returns:
            torch.Tensor: Output probability tensor over [0, `max_value`], with shape (N, `self.max_value`).
            torch.Tensor: Aleatoric and epistemic uncertainties per input, with shape (N, 2).
        """
        means = []
        variances = []
        dists = []
        for member in self.members:
            y_hat = member.predict(x)
            eps = torch.tensor(1e-3, device=y_hat.device)

            mu, alpha = torch.split(y_hat, [1, 1], dim=-1)
            mu = mu.flatten()
            alpha = torch.clamp(alpha, min=eps).flatten()

            # Convert to standard parametrization.
            var = mu + alpha * mu**2
            p = mu / torch.maximum(var, eps)

            # Torch docs lie and say this should be P(success).
            failure_prob = torch.clamp(1 - p, min=eps, max=1 - eps)

            n = mu**2 / torch.maximum(var - mu, eps)
            dist = torch.distributions.NegativeBinomial(total_count=n, probs=failure_prob)
            means.append(dist.mean)
            variances.append(var)
            dists.append(dist)

        mixture = DiscreteMixture(distributions=dists, weights=torch.ones(len(dists)))
        probabilities = mixture.pmf(torch.arange(self.max_value).unsqueeze(1)).transpose(0, 1)

        means = torch.stack(means, dim=1)
        variances = torch.stack(variances, dim=1)
        aleatoric = variances.mean(dim=1)
        epistemic = means.var(dim=1)
        uncertainties = torch.stack([aleatoric, epistemic], dim=1)

        return probabilities, uncertainties

    def _update_test_metrics(self, y_hat: tuple[torch.Tensor, torch.Tensor], y: torch.Tensor):
        probabilities, uncertainties = y_hat
        preds = probabilities.argmax(dim=1)
        targets = y.flatten()

        support = torch.arange(self.max_value, device=self.device)
        mu = (probabilities * support).sum(dim=1)
        var = (probabilities * (support.unsqueeze(1) - mu).transpose(0, 1) ** 2).sum(dim=1)
        precision = 1 / var

        self.rmse.update(preds, targets)
        self.mae.update(preds, targets)
        self.mp.update(precision)
        self.crps.update(probabilities, targets)
