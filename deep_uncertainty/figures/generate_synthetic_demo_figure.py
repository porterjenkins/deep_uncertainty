from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
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
    metrics_files: list[str],
    save_path: Path | str,
):
    save_path = Path(save_path)
    if save_path.suffix not in (".pdf", ".png"):
        raise ValueError("Must specify a save path that is either a PDF or a PNG.")
    fig, axs = plt.subplots(
        3, len(ensembles), figsize=(3 * len(ensembles), 9), sharex="col", sharey="row"
    )
    [axs[0, j].set_title(name, fontsize=14) for j, name in enumerate(names)]

    datamodule = TabularDataModule(
        "data/discrete-wave/discrete_sine_wave.npz",
        batch_size=1,
        num_workers=0,
        persistent_workers=False,
    )
    datamodule.setup("")
    datamodule.prepare_data()
    gt_uncertainty: dict[str, np.ndarray] = np.load("data/discrete-wave/gt_uncertainty.npz")

    X = datamodule.test.tensors[0]
    y = datamodule.test.tensors[1]

    domain = np.linspace(X.min(), X.max())
    gt_aleatoric = interp1d(
        x=gt_uncertainty["X"].flatten(),
        y=gt_uncertainty["aleatoric"].flatten(),
        kind="cubic",
        fill_value="extrapolate",
    )(domain)
    order = torch.argsort(X.flatten())

    for j, ensemble in enumerate(ensembles):
        with open(metrics_files[j]) as f:
            metrics = yaml.safe_load(f)

        mae = metrics["mae"]
        crps = metrics["crps"]
        probs, uncertainties = ensemble.predict(X)
        aleatoric = uncertainties[:, 0].detach()
        epistemic = uncertainties[:, 1].detach()
        cdf = torch.cumsum(probs, dim=1)
        lower = torch.tensor([torch.searchsorted(cdf[i], 0.025) for i in range(len(cdf))])
        upper = torch.tensor([torch.searchsorted(cdf[i], 0.975) for i in range(len(cdf))])
        mu = (probs * torch.arange(2000).view(1, -1)).sum(dim=1)

        data_label = "Test Data" if j == len(ensembles) - 1 else None
        axs[0, j].scatter(
            X.flatten(), y.flatten(), c="cornflowerblue", alpha=0.4, s=20, label=data_label
        )
        axs[0, j].plot(X[order], mu[order].detach().numpy(), c="black", label="Predictive Mean")
        axs[0, j].fill_between(
            X[order].flatten(),
            lower[order],
            upper[order],
            color="cornflowerblue",
            alpha=0.2,
            zorder=0,
            label="Predictive Uncertainty",
        )
        axs[0, j].annotate(f"MAE: {mae:.3f}", (0, 40))
        axs[0, j].annotate(f"CRPS: {crps:.3f}", (0, 37))

        axs[1, j].plot(X[order], aleatoric[order], label="Predicted")
        axs[1, j].plot(domain, gt_aleatoric, linestyle="--", label="Ground Truth")
        axs[1, j].legend(loc="upper left", prop={"size": 9})
        axs[2, j].plot(X[order], epistemic[order])

    axs[0, 0].set_ylabel("Predictive Dist.", fontsize=14)
    axs[1, 0].set_ylabel("Aleatoric", fontsize=14)
    axs[2, 0].set_ylabel("Epistemic", fontsize=14)
    [ax.set_xticks([0, np.pi, 2 * np.pi]) for ax in axs[-1].ravel()]
    [
        ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$"]) for ax in axs[-1].ravel()
    ]  # Optional manual labels (or use formatter below)
    axs[0, 0].set_yticks([0, 20, 40])
    axs[1, 0].set_yticks([0, 15, 30])
    [ax.tick_params(axis="both", which="major", labelsize=8) for ax in axs.ravel()]
    [ax.tick_params(axis="both", which="minor", labelsize=8) for ax in axs.ravel()]

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
    names = ["Poisson Ensemble", "NB Ensemble", "DDPN Ensemble (Ours)"]
    metrics_files = [
        "results/discrete-wave/ensembles/poisson_ensemble/test_metrics.yaml",
        "results/discrete-wave/ensembles/nbinom_ensemble/test_metrics.yaml",
        "results/discrete-wave/ensembles/ddpn_ensemble/test_metrics.yaml",
    ]
    produce_figure(ensembles, names, metrics_files, save_path)
