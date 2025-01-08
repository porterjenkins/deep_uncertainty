from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import interp1d

from deep_uncertainty.datamodules.tabular_datamodule import TabularDataModule
from deep_uncertainty.models.ensembles import DoublePoissonMixtureNN
from deep_uncertainty.models.ensembles import NegBinomMixtureNN
from deep_uncertainty.models.ensembles import PoissonMixtureNN
from deep_uncertainty.models.ensembles.deep_regression_ensemble import DeepRegressionEnsemble
from deep_uncertainty.utils.configs import EnsembleConfig


def produce_figure(
    ensembles: list[DeepRegressionEnsemble],
    names: list[str],
    save_path: Path | str,
):
    save_path = Path(save_path)
    if save_path.suffix not in (".pdf", ".png"):
        raise ValueError("Must specify a save path that is either a PDF or a PNG.")
    fig, axs = plt.subplots(
        3, len(ensembles), figsize=(9, 2 * len(ensembles)), sharex="col", sharey="row"
    )
    axs[0, 0].set_ylabel("Predictive Dist.")
    axs[1, 0].set_ylabel("Aleatoric")
    axs[2, 0].set_ylabel("Epistemic")
    [axs[0, i].set_title(name) for i, name in enumerate(names)]

    datamodule = TabularDataModule(
        "data/discrete-wave/discrete_sine_wave.npz",
        batch_size=1,
        num_workers=0,
        persistent_workers=False,
    )
    datamodule.setup("")
    datamodule.prepare_data()
    gt_uncertainty: dict[str, np.ndarray] = np.load("data/discrete-wave-ii/gt_uncertainty.npz")

    gt_aleatoric = interp1d(
        x=gt_uncertainty["X"].flatten(),
        y=gt_uncertainty["variance"].flatten(),
        fill_value="extrapolate",
    )
    orig_X = datamodule.test.tensors[0]
    orig_y = datamodule.test.tensors[1]
    X_with_ood = torch.cat(
        [
            orig_X,
            torch.linspace(orig_X.min() - 0.5, orig_X.min(), 50).unsqueeze(1),
            torch.linspace(orig_X.max(), orig_X.max() + 0.5, 50).unsqueeze(1),
        ]
    )
    order = torch.argsort(X_with_ood.flatten())

    for j, ensemble in enumerate(ensembles):
        probs, uncertainties = ensemble(X_with_ood)
        aleatoric, epistemic = uncertainties[:, 0], uncertainties[:, 1]
        cdf = torch.cumsum(probs, dim=1)
        lower = torch.tensor([torch.searchsorted(cdf[i], 0.025) for i in range(len(cdf))])
        upper = torch.tensor([torch.searchsorted(cdf[i], 0.975) for i in range(len(cdf))])
        mu = (probs * torch.arange(2000).view(1, -1)).sum(dim=1)

        axs[0, j].scatter(orig_X.flatten(), orig_y.flatten(), c="cornflowerblue", alpha=0.4, s=10)
        axs[0, j].plot(X_with_ood[order], mu[order].detach().numpy(), c="black", label="Mean")
        axs[0, j].fill_between(
            X_with_ood[order].flatten(),
            lower[order],
            upper[order],
            color="cornflowerblue",
            alpha=0.2,
            zorder=0,
        )
        axs[1, j].plot(
            X_with_ood[order], gt_aleatoric(X_with_ood[order]), color="black", linestyle="--"
        )
        axs[1, j].plot(X_with_ood[order], aleatoric[order].detach().numpy(), color="black")
        axs[2, j].plot(X_with_ood[order], epistemic[order].detach().numpy(), color="black")

    [ax.set_xticks([]) for ax in axs.ravel()]
    [ax.set_yticks([]) for ax in axs.ravel()]
    fig.tight_layout()
    fig.savefig(save_path, format="pdf", dpi=150)


if __name__ == "__main__":
    save_path = "deep_uncertainty/figures/artifacts/synthetic_demo_ii.pdf"
    ensembles = [
        PoissonMixtureNN.from_config(
            EnsembleConfig.from_yaml("configs/discrete-wave/ensembles/poisson.yaml")
        ),
        NegBinomMixtureNN.from_config(
            EnsembleConfig.from_yaml("configs/discrete-wave/ensembles/nbinom.yaml")
        ),
        DoublePoissonMixtureNN.from_config(
            EnsembleConfig.from_yaml("configs/discrete-wave/ensembles/ddpn.yaml")
        ),
    ]
    names = ["Poisson Mixture", "NB Mixture", "DDPN Mixture (Ours)"]
    produce_figure(ensembles, names, save_path)
