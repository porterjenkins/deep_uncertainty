import torch

from deep_uncertainty.evaluation.custom_torchmetrics import ContinuousRankedProbabilityScore
from deep_uncertainty.models import GaussianNN
from deep_uncertainty.models.ensembles.deep_regression_ensemble import DeepRegressionEnsemble


class GaussianMixtureNN(DeepRegressionEnsemble[GaussianNN]):
    """An ensemble of Gaussian Neural Nets, formed into a mixture as specified in https://arxiv.org/abs/1612.01474.

    Aleatoric / epistemic uncertainty is calculated according to https://arxiv.org/abs/1703.04977.

    This model does not "train" and should strictly be used for prediction.
    """

    member_type = GaussianNN

    def __init__(self, members: list[GaussianNN]):
        super().__init__(members=members)
        self.crps = ContinuousRankedProbabilityScore(mode="gaussian")

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the ensemble.

        Args:
            x (torch.Tensor): Batched input tensor, with shape (N, ...).

        Returns:
            torch.Tensor: Output tensor, with shape (N, 4). Output is (mean, predictive_uncertainty, aleatoric_uncertainty, epistemic_uncertainty) for each input.
        """
        mu_vals = []
        var_vals = []
        for member in self.members:
            mu, var = torch.split(member._predict_impl(x), [1, 1], dim=-1)
            mu_vals.append(mu)
            var_vals.append(var)
        mu_vals = torch.cat(mu_vals, dim=1)
        var_vals = torch.cat(var_vals, dim=1)

        pred_mean = mu_vals.mean(dim=1)
        aleatoric_uncertainty = var_vals.mean(dim=1)
        epistemic_uncertainty = mu_vals.var(dim=1)
        pred_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
        output = torch.stack(
            [
                pred_mean,
                pred_uncertainty,
                aleatoric_uncertainty,
                epistemic_uncertainty,
            ],
            dim=1,
        )
        return output

    def _update_test_metrics(self, y_hat: torch.Tensor, y: torch.Tensor):
        mu, var, aleatoric, epistemic = torch.split(y_hat, [1, 1, 1, 1], dim=-1)
        mu = mu.flatten()
        var = var.flatten()
        precision = 1 / var
        std = torch.sqrt(var)
        targets = y.flatten()

        preds = torch.round(mu)  # Since we have to predict counts.

        # We compute "probability" by normalizing density over the discrete counts.
        dist = torch.distributions.Normal(loc=mu, scale=std)
        target_probs = dist.cdf(targets + 0.5) - dist.cdf(targets - 0.5)
        self.rmse.update(preds, targets)
        self.mae.update(preds, targets)
        self.nll.update(target_probs=target_probs)
        self.mp.update(precision)
        self.crps.update(torch.stack([mu, var], dim=1), targets)
