from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import nbinom
from scipy.stats import norm
from scipy.stats import poisson
from scipy.stats import rv_discrete

from deep_uncertainty.datamodules import COCOPeopleDataModule
from deep_uncertainty.models import DoublePoissonNN
from deep_uncertainty.models import FaithfulGaussianNN
from deep_uncertainty.models import GaussianNN
from deep_uncertainty.models import NegBinomNN
from deep_uncertainty.models import PoissonNN
from deep_uncertainty.models.ensembles import DoublePoissonMixtureNN
from deep_uncertainty.models.ensembles import GaussianMixtureNN
from deep_uncertainty.models.ensembles import NegBinomMixtureNN
from deep_uncertainty.models.ensembles import PoissonMixtureNN
from deep_uncertainty.random_variables import DoublePoisson


def main(chkp_dir: str | Path, save_path: str | Path, image_indices: list[int], xlims: list[int]):

    chkp_dir = Path(chkp_dir)
    save_path = Path(save_path)
    eps = 1e-6

    datamodule = COCOPeopleDataModule(
        root_dir="data/coco-people",
        batch_size=1,
        num_workers=1,
        persistent_workers=False,
        surface_image_path=True,
    )
    datamodule.prepare_data()
    datamodule.setup("test")
    dataloader = datamodule.test_dataloader()

    batches = []
    for i, x in enumerate(dataloader):
        if i > max(image_indices):
            break
        if i in image_indices:
            batches.append(x)

    beta_ddpn_model = DoublePoissonNN.load_from_checkpoint(
        chkp_dir / "coco_people_beta_ddpn_1.0" / "version_0" / "best_mae.ckpt"
    )
    faithful_gaussian_model = FaithfulGaussianNN.load_from_checkpoint(
        chkp_dir / "coco_people_faithful_gaussian" / "version_0" / "best_mae.ckpt"
    )
    poisson_model = PoissonNN.load_from_checkpoint(
        chkp_dir / "coco_people_poisson" / "version_0" / "best_mae.ckpt"
    )
    nbinom_model = NegBinomNN.load_from_checkpoint(
        chkp_dir / "coco_people_nbinom" / "version_0" / "best_mae.ckpt"
    )

    beta_ddpn_members = [
        DoublePoissonNN.load_from_checkpoint(
            chkp_dir / "coco_people_beta_ddpn_1.0" / f"version_{i}" / "best_mae.ckpt"
        )
        for i in range(5)
    ]
    gaussian_members = [
        GaussianNN.load_from_checkpoint(
            chkp_dir / "coco_people_gaussian" / f"version_{i}" / "best_mae.ckpt"
        )
        for i in range(5)
    ]
    poisson_members = [
        PoissonNN.load_from_checkpoint(
            chkp_dir / "coco_people_poisson" / f"version_{i}" / "best_mae.ckpt"
        )
        for i in range(5)
    ]
    nbinom_members = [
        NegBinomNN.load_from_checkpoint(
            chkp_dir / "coco_people_nbinom" / f"version_{i}" / "best_mae.ckpt"
        )
        for i in range(5)
    ]

    beta_ddpn_mixture = DoublePoissonMixtureNN(beta_ddpn_members)
    gaussian_quasi_mixture = GaussianMixtureNN(gaussian_members)
    poisson_mixture = PoissonMixtureNN(poisson_members)
    nbinom_mixture = NegBinomMixtureNN(nbinom_members)

    xlims_vals = [(xlims[i], xlims[i + 1]) for i in range(0, len(xlims), 2)]
    fig, all_axs = plt.subplots(len(image_indices), 3, figsize=(20, 20))
    for batch, axs_row, xlims in zip(batches, all_axs, xlims_vals):
        image_tensor, (image_path, label) = batch
        image = plt.imread(image_path[0])

        img_ax: plt.Axes = axs_row[0]
        single_dist_ax: plt.Axes = axs_row[1]
        ensemble_dist_ax: plt.Axes = axs_row[2]

        mu, phi = (
            beta_ddpn_model._predict_impl(image_tensor.to(beta_ddpn_model.device))
            .flatten()
            .detach()
            .cpu()
            .numpy()
        )
        mean, var = (
            faithful_gaussian_model._predict_impl(image_tensor.to(faithful_gaussian_model.device))
            .flatten()
            .detach()
            .cpu()
            .numpy()
        )
        lmbda = (
            poisson_model._predict_impl(image_tensor.to(poisson_model.device))
            .flatten()
            .detach()
            .cpu()
            .numpy()
        )
        nbinom_mu, alpha = (
            nbinom_model._predict_impl(image_tensor.to(nbinom_model.device))
            .flatten()
            .detach()
            .cpu()
            .numpy()
        )
        nbinom_var = nbinom_mu + alpha * nbinom_mu**2
        n = nbinom_mu**2 / np.maximum(nbinom_var - nbinom_mu, eps)
        p = nbinom_mu / np.maximum(nbinom_var, eps)

        ddpn_dist = DoublePoisson(mu, phi)
        gauss_dist = norm(loc=mean, scale=np.sqrt(var))
        poiss_dist = poisson(mu=lmbda)
        nbinom_dist = nbinom(n=n, p=p)
        min_val = xlims[0]
        max_val = xlims[1]
        disc_support = np.arange(min_val, max_val + 1)
        cont_support = np.linspace(min_val, max_val, num=500)

        img_ax.imshow(image)
        img_ax.axis("off")

        single_dist_ax.plot(
            disc_support,
            ddpn_dist.pmf(disc_support),
            ".-",
            label=r"$\beta_{1.0}$-DDPN (ours)",
            alpha=0.5,
        )
        single_dist_ax.plot(
            disc_support, poiss_dist.pmf(disc_support), ".-", label="Poisson DNN", alpha=0.5
        )
        single_dist_ax.plot(
            disc_support, nbinom_dist.pmf(disc_support), ".-", label="NB DNN", alpha=0.5
        )
        single_dist_ax.plot(
            cont_support, gauss_dist.pdf(cont_support), label="Stirn et al. ('23)", alpha=0.5
        )
        single_dist_ax.scatter(label, 0.02, marker="*", s=150, c="black", zorder=10)
        single_dist_ax.set_ylim(-0.01, 1.1)
        single_dist_ax.legend()

        beta_ddpn_probs = (
            beta_ddpn_mixture(image_tensor.to(beta_ddpn_model.device))
            .flatten()
            .detach()
            .cpu()
            .numpy()
        )
        mu, var, _, _ = (
            gaussian_quasi_mixture(image_tensor.to(faithful_gaussian_model.device))
            .flatten()
            .detach()
            .cpu()
            .numpy()
        )
        poisson_probs = (
            poisson_mixture(image_tensor.to(poisson_model.device)).flatten().detach().cpu().numpy()
        )
        nbinom_probs = (
            nbinom_mixture(image_tensor.to(nbinom_model.device)).flatten().detach().cpu().numpy()
        )

        ddpn_dist = rv_discrete(0, 2000, values=(range(2000), beta_ddpn_probs))
        poiss_dist = rv_discrete(0, 2000, values=(range(2000), poisson_probs))
        nbinom_dist = rv_discrete(0, 2000, values=(range(2000), nbinom_probs))
        gauss_dist = norm(loc=mu, scale=np.sqrt(var))

        ensemble_dist_ax.plot(
            disc_support,
            ddpn_dist.pmf(disc_support),
            ".-",
            label=r"$\beta_{1.0}$-DDPN Mixture (ours)",
            alpha=0.5,
        )
        ensemble_dist_ax.plot(
            disc_support,
            poiss_dist.pmf(disc_support),
            ".-",
            label="Poisson DNN Mixture",
            alpha=0.5,
        )
        ensemble_dist_ax.plot(
            disc_support, nbinom_dist.pmf(disc_support), ".-", label="NB DNN Mixture", alpha=0.5
        )
        ensemble_dist_ax.plot(
            cont_support, gauss_dist.pdf(cont_support), label="Laksh. et al. ('17)", alpha=0.5
        )
        ensemble_dist_ax.scatter(label, 0.02, marker="*", s=150, c="black", zorder=10)
        ensemble_dist_ax.set_ylim(-0.01, 1.1)
        ensemble_dist_ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, format="pdf", dpi=150)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--chkp-dir", type=str, default="chkp")
    parser.add_argument(
        "--save-path",
        type=str,
        default="deep_uncertainty/figures/artifacts/coco_supplementary_comparison.pdf",
    )
    parser.add_argument(
        "--image-indices",
        nargs="+",
        type=int,
        default=[6, 10, 14],
        help="Indices of the images in the test split of COCO-People to show. Specify as `idx_0 idx_1 ...`. Must be in ascending order.",
    )
    parser.add_argument(
        "--xlims",
        nargs="+",
        type=int,
        default=[0, 6, 0, 6, 0, 12],
        help="The xlims to set for the dist. plot with each image. Specify as `xmin_0 xmax_0 xmin_1 xmax_1 ...`.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.chkp_dir, args.save_path, args.image_indices, args.xlims)
