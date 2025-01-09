from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from deep_uncertainty.models import DoublePoissonNN
from deep_uncertainty.models import NegBinomNN
from deep_uncertainty.random_variables import DoublePoisson


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


@torch.inference_mode()
def main(save_path: Path):
    data = np.load("data/bowtie/bowtie.npz")

    X_test = data["X_test"].flatten()
    y_test = data["y_test"].flatten()
    order = np.argsort(X_test)

    X_test_tensor = torch.tensor(X_test).float().unsqueeze(1)

    ddpn = DoublePoissonNN.load_from_checkpoint("chkp/bowtie/ddpn/version_0/best_loss.ckpt")
    nbinom = NegBinomNN.load_from_checkpoint("chkp/bowtie/nbinom/version_0/best_loss.ckpt")

    fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey="row")

    nbinom_predictive = nbinom.predictive_dist(nbinom.predict(X_test_tensor))
    support = torch.arange(2000).view(-1, 1)
    nbinom_probs = torch.exp(nbinom_predictive.log_prob(support)).T
    nbinom_cdf = torch.cumsum(nbinom_probs, dim=1)
    nbinom_lower = torch.tensor(
        [torch.searchsorted(nbinom_cdf[i], 0.025) for i in range(len(nbinom_cdf))]
    )
    nbinom_upper = torch.tensor(
        [torch.searchsorted(nbinom_cdf[i], 0.975) for i in range(len(nbinom_cdf))]
    )

    ddpn_predictive: DoublePoisson = ddpn.predictive_dist(ddpn.predict(X_test_tensor))
    ddpn_lower = ddpn_predictive.ppf(0.025)
    ddpn_upper = ddpn_predictive.ppf(0.975)

    axs[0].scatter(X_test, y_test, c="cornflowerblue", alpha=0.4, s=20)
    axs[0].plot(
        X_test[order], nbinom_predictive.mean[order].detach().numpy(), c="black", label="Mean"
    )
    axs[0].fill_between(
        X_test[order],
        nbinom_lower[order],
        nbinom_upper[order],
        color="cornflowerblue",
        alpha=0.2,
        zorder=0,
    )
    axs[0].set_title("NB DNN")

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
    axs[1].set_title("DDPN (Ours)")

    fig.tight_layout()
    fig.savefig(save_path, format="pdf", dpi=150)


if __name__ == "__main__":
    save_path = "deep_uncertainty/figures/artifacts/misspecification_recovery_nbinom.pdf"
    main(save_path=save_path)
