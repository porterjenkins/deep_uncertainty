import torch

from deep_uncertainty.evaluation.custom_torchmetrics import ContinuousRankedProbabilityScore
from deep_uncertainty.models import NaturalGaussianNN
from deep_uncertainty.models.ensembles.deep_regression_ensemble import DeepRegressionEnsemble


class NaturalGaussianMixtureNN(DeepRegressionEnsemble[NaturalGaussianNN]):
    """An ensemble of naturally-parametrized Gaussian NNs that outputs the predictive mean and variance of the implied uniform mixture.

    See https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html for details.
    """

    member_type = NaturalGaussianNN

    def __init__(self, members: list[NaturalGaussianNN]):
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
            eta_1, eta_2 = torch.split(member.predict(x), [1, 1], dim=-1)
            mu = member._natural_to_mu(eta_1, eta_2)
            var = member._natural_to_var(eta_2)
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
        targets = y.flatten()

        # We compute "probability" by normalizing density over the discrete counts.
        self.rmse.update(mu, targets)
        self.mae.update(mu, targets)
        self.mp.update(precision)
        self.crps.update(torch.stack([mu, var], dim=1), targets)
