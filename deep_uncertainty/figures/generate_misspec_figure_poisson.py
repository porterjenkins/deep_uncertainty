from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from deep_uncertainty.models import DoublePoissonNN
from deep_uncertainty.models import PoissonNN
from deep_uncertainty.random_variables import DoublePoisson


@torch.inference_mode()
def main(save_path: Path):
    data = np.load("data/discrete-parabola/discrete_parabola.npz")
    X_test = data["X_test"].flatten()
    y_test = data["y_test"].flatten()
    order = np.argsort(X_test)

    X_test_tensor = torch.tensor(X_test).float().unsqueeze(1)

    ddpn = DoublePoissonNN.load_from_checkpoint(
        "chkp/discrete-parabola/ddpn/version_0/best_loss.ckpt"
    )
    poisson = PoissonNN.load_from_checkpoint(
        "chkp/discrete-parabola/poisson/version_0/best_loss.ckpt"
    )

    with open("results/discrete-parabola/ddpn/version_0/test_metrics.yaml") as f:
        ddpn_metrics = yaml.safe_load(f)
    with open("results/discrete-parabola/poisson/version_0/test_metrics.yaml") as f:
        poisson_metrics = yaml.safe_load(f)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey="row")

    poisson_predictive = poisson.predictive_dist(poisson.predict(X_test_tensor))
    support = torch.arange(2000).view(-1, 1)
    poisson_probs = torch.exp(poisson_predictive.log_prob(support)).T
    poisson_cdf = torch.cumsum(poisson_probs, dim=1)
    poisson_lower = torch.tensor(
        [torch.searchsorted(poisson_cdf[i], 0.025) for i in range(len(poisson_cdf))]
    )
    poisson_upper = torch.tensor(
        [torch.searchsorted(poisson_cdf[i], 0.975) for i in range(len(poisson_cdf))]
    )

    ddpn_predictive: DoublePoisson = ddpn.predictive_dist(ddpn.predict(X_test_tensor))
    ddpn_lower = ddpn_predictive.ppf(0.025)
    ddpn_upper = ddpn_predictive.ppf(0.975)

    axs[0].scatter(X_test, y_test, c="cornflowerblue", alpha=0.4, s=20)
    axs[0].plot(
        X_test[order], poisson_predictive.mean[order].detach().numpy(), c="black", label="Mean"
    )
    axs[0].fill_between(
        X_test[order],
        poisson_lower[order],
        poisson_upper[order],
        color="cornflowerblue",
        alpha=0.2,
        zorder=0,
    )
    axs[0].annotate(f"MAE: {poisson_metrics['test_mae']:.3f}", (-4.9, 16.7))
    axs[0].annotate(f"CRPS: {poisson_metrics['crps']:.3f}", (-4.9, 15.7))
    axs[0].set_title("Poisson DNN")

    axs[1].scatter(X_test, y_test, c="cornflowerblue", alpha=0.4, s=20)
    axs[1].plot(X_test[order], ddpn_predictive.mu[order].detach().numpy(), c="black", label="Mean")
    axs[1].fill_between(
        X_test[order],
        ddpn_lower[order],
        ddpn_upper[order],
        color="cornflowerblue",
        alpha=0.2,
        zorder=0,
    )
    axs[1].annotate(f"MAE: {ddpn_metrics['test_mae']:.3f}", (-4.9, 16.7))
    axs[1].annotate(f"CRPS: {ddpn_metrics['crps']:.3f}", (-4.9, 15.7))
    axs[1].set_title("DDPN (Ours)")

    fig.tight_layout()
    fig.savefig(save_path, format="pdf", dpi=150)


if __name__ == "__main__":
    save_path = "deep_uncertainty/figures/artifacts/misspecification_recovery_poisson.pdf"
    main(save_path=save_path)
