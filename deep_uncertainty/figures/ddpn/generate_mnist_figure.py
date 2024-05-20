import math
from argparse import ArgumentParser
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import nbinom
from scipy.stats import norm
from scipy.stats import poisson
from torch.utils.data import DataLoader
from torchvision.transforms.functional import rotate

from deep_uncertainty.datamodules import MNISTDataModule
from deep_uncertainty.models import DoublePoissonNN
from deep_uncertainty.models import GaussianNN
from deep_uncertainty.models import NegBinomNN
from deep_uncertainty.models import PoissonNN
from deep_uncertainty.random_variables import DoublePoisson
from deep_uncertainty.utils.model_utils import get_hdi


def get_mnist_preds(
    digits: list[int],
    angles: list[float | int],
    test_loader: DataLoader,
    gaussian_nn: GaussianNN,
    poisson_nn: PoissonNN,
    nbinom_nn: NegBinomNN,
    ddpn: DoublePoissonNN,
) -> tuple[list, list, list, list]:
    """Given a list of digits, get each model's predictions for all instances of each digit rotated by the specified angles.

    Args:
        digits (list[int]): List of digits to get predictions for.
        angles (list[float | int]): List of the rotation angles to get predictions on for each digit instance.
        test_loader (DataLoader): DataLoader with images/labels.
        gaussian_nn (GaussianNN): The GaussianNN to get predictions from.
        poisson_nn (PoissonNN): The PoissonNN to get predictions from.
        nbinom_nn (NegBinomNN): The NegBinomNN to get predictions from.
        ddpn (DoublePoissonNN): The DoublePoissonNN to get predictions from.

    Returns:
        tuple[list, list, list, list]: Predictions for the given digits, in the order (`gaussian_nn`, `poisson_nn`, `nbinom_nn`, `ddpn`).

        Element `i` of a specific list is a (num_angles, num_instances, model_output_dim) tensor.
    """
    all_gaussian_preds = []
    all_poisson_preds = []
    all_nbinom_preds = []
    all_ddpn_preds = []
    num_angles = len(angles)

    with torch.inference_mode():
        for digit in digits:

            instances_of_target = [img for (img, label) in test_loader if label == digit]
            num_images = len(instances_of_target)

            gaussian_preds = torch.zeros(num_angles, num_images, 2)
            poisson_preds = torch.zeros(num_angles, num_images, 1)
            neg_binom_preds = torch.zeros(num_angles, num_images, 2)
            double_poisson_preds = torch.zeros(num_angles, num_images, 2)

            for i, angle in enumerate(angles):
                for j, tensor in enumerate(instances_of_target):
                    rotated = rotate(tensor, angle)
                    gaussian_y_hat = gaussian_nn._predict_impl(rotated).squeeze()
                    gaussian_preds[i, j, :] = gaussian_y_hat

                    poisson_y_hat = poisson_nn._predict_impl(rotated).squeeze()
                    poisson_preds[i, j, :] = poisson_y_hat

                    nbinom_y_hat = nbinom_nn._predict_impl(rotated).squeeze()
                    neg_binom_preds[i, j, :] = nbinom_y_hat

                    dpo_y_hat = ddpn._predict_impl(rotated).squeeze()
                    double_poisson_preds[i, j, :] = dpo_y_hat

            all_gaussian_preds.append(gaussian_preds)
            all_poisson_preds.append(poisson_preds)
            all_nbinom_preds.append(neg_binom_preds)
            all_ddpn_preds.append(double_poisson_preds)

    return all_gaussian_preds, all_poisson_preds, all_nbinom_preds, all_ddpn_preds


def produce_figure(
    gaussian_path: str, poisson_path: str, nbinom_path: str, ddpn_path: str, save_path: str
):
    datamodule = MNISTDataModule(
        root_dir="./data/mnist", batch_size=1, num_workers=0, persistent_workers=False
    )
    datamodule.setup("predict")
    test_loader = datamodule.test_dataloader()

    gaussian_nn = GaussianNN.load_from_checkpoint(gaussian_path, map_location="cpu")
    poisson_nn = PoissonNN.load_from_checkpoint(poisson_path, map_location="cpu")
    nbinom_nn = NegBinomNN.load_from_checkpoint(nbinom_path, map_location="cpu")
    ddpn = DoublePoissonNN.load_from_checkpoint(ddpn_path, map_location="cpu")

    digits = [0, 5, 8]
    angles = [0, 45]

    all_gaussian_preds, all_poisson_preds, all_nbinom_preds, all_ddpn_preds = get_mnist_preds(
        digits=digits,
        angles=angles,
        test_loader=test_loader,
        gaussian_nn=gaussian_nn,
        poisson_nn=poisson_nn,
        nbinom_nn=nbinom_nn,
        ddpn=ddpn,
    )

    cont_support = np.linspace(-2, 12, 500)
    disc_support = np.arange(0, 20)

    fig, axs = plt.subplots(5, len(angles) * len(digits), figsize=(10, 7), sharey="row")
    for i, digit in enumerate(digits):

        instances_of_target = [img for (img, label) in test_loader if label == digit]

        avg_gaussian_preds = torch.mean(all_gaussian_preds[i], dim=1)
        avg_poisson_preds = torch.mean(all_poisson_preds[i], dim=1)
        avg_nbinom_preds = torch.mean(all_nbinom_preds[i], dim=1)
        avg_double_poisson_preds = torch.mean(all_ddpn_preds[i], dim=1)

        image_tensor = instances_of_target[7] / 255.0
        tensors = [rotate(image_tensor, angle, fill=image_tensor.min().item()) for angle in angles]

        for ax, tensor, angle in zip(axs[0, i * 2 : i * 2 + 2], tensors, angles):
            ax: plt.Axes
            ax.set_title(f"Rotation: ${angle}^\circ$", fontsize=8)
            ax.imshow(tensor.squeeze(), cmap="gray")
            ax.axis("off")

        for j, angle in enumerate(angles):

            idx = 2 * i + j
            gauss_mu, var = avg_gaussian_preds[j].numpy()
            gauss_dist = norm(loc=gauss_mu, scale=math.sqrt(var))
            lam = avg_poisson_preds[j].detach().numpy()
            poiss_dist = poisson(mu=lam)
            nbinom_mu, alpha = avg_nbinom_preds[j].numpy()
            nbinom_var = nbinom_mu + alpha * nbinom_mu**2
            p = nbinom_mu / nbinom_var
            n = nbinom_mu**2 / (nbinom_var - nbinom_mu)
            nbinom_dist = nbinom(n=n, p=p)
            dpo_mu, phi = avg_double_poisson_preds[j].numpy()
            dpo_dist = DoublePoisson(dpo_mu, phi)

            gaussian_ax: plt.Axes = axs[1, idx]
            poisson_ax: plt.Axes = axs[2, idx]
            nbinom_ax: plt.Axes = axs[3, idx]
            ddpn_ax: plt.Axes = axs[4, idx]

            gaussian_ax.plot(cont_support, gauss_dist.pdf(cont_support), linewidth=0.7)
            lower, upper = get_hdi(gauss_dist)
            gaussian_ax.set_ylim(-0.03, 2.3)
            gaussian_ax.text(
                9,
                2,
                f"95% HDI: [{lower:.2f}, {upper:.2f}]",
                horizontalalignment="right",
                fontsize=6,
            )
            gaussian_ax.fill_between(
                x=cont_support,
                y1=-0.1,
                y2=gauss_dist.pdf(cont_support),
                where=(cont_support >= lower) & (cont_support <= upper),
                color="gray",
                alpha=0.3,
            )
            gaussian_ax.set_xticks([])

            poisson_ax.plot(
                disc_support, poiss_dist.pmf(disc_support), ".-", linewidth=0.7, markersize=2
            )
            lower, upper = get_hdi(poiss_dist)
            poisson_ax.set_ylim(-0.03, 1.2)
            poisson_ax.text(
                9,
                1.05,
                f"95% HDI: [{int(lower[0])}, {int(upper[0])}]",
                horizontalalignment="right",
                fontsize=6,
            )
            poisson_ax.fill_between(
                x=disc_support,
                y1=-0.1,
                y2=poiss_dist.pmf(disc_support),
                where=(disc_support >= lower) & (disc_support <= upper),
                color="gray",
                alpha=0.3,
            )
            poisson_ax.set_xticks([])

            nbinom_ax.plot(
                disc_support, nbinom_dist.pmf(disc_support), ".-", linewidth=0.7, markersize=2
            )
            lower, upper = get_hdi(nbinom_dist)
            nbinom_ax.set_ylim(-0.03, 1.2)
            nbinom_ax.text(
                9,
                1.05,
                f"95% HDI: [{int(lower)}, {int(upper)}]",
                horizontalalignment="right",
                fontsize=6,
            )
            nbinom_ax.fill_between(
                x=disc_support,
                y1=-0.1,
                y2=nbinom_dist.pmf(disc_support),
                where=(disc_support >= lower) & (disc_support <= upper),
                color="gray",
                alpha=0.3,
            )
            nbinom_ax.set_xticks([])

            ddpn_ax.plot(
                disc_support, dpo_dist.pmf(disc_support), ".-", linewidth=0.7, markersize=2
            )
            lower, upper = get_hdi(dpo_dist, 0.95)
            ddpn_ax.set_ylim(-0.03, 1.2)
            ddpn_ax.text(
                9, 1.05, f"95% HDI: [{lower}, {upper}]", horizontalalignment="right", fontsize=6
            )
            ddpn_ax.fill_between(
                x=disc_support,
                y1=-0.1,
                y2=dpo_dist.pmf(disc_support),
                where=(disc_support >= lower) & (disc_support <= upper),
                color="gray",
                alpha=0.3,
            )
            ddpn_ax.set_xticks([0, 3, 6, 9])

            if idx == 0:
                gaussian_ax.set_ylabel("Gaussian DNN", fontsize=8)
                poisson_ax.set_ylabel("Poisson DNN", fontsize=8)
                nbinom_ax.set_ylabel("NB DNN", fontsize=8)
                ddpn_ax.set_ylabel("DDPN (Ours)", fontsize=8)

    for ax in axs.ravel()[6:]:
        ax.set_xlim(-1, 10)
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("gray")

    fig.tight_layout()
    fig.savefig(save_path, format="pdf", dpi=600)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--gaussian-path", default="chkp/mnist_gaussian/version_1/best_mae.ckpt")
    parser.add_argument("--poisson-path", default="chkp/mnist_poisson/version_0/best_mae.ckpt")
    parser.add_argument("--nbinom-path", default="chkp/mnist_nbinom/version_0/best_mae.ckpt")
    parser.add_argument("--ddpn-path", default="chkp/mnist_beta_ddpn/version_0/best_mae.ckpt")
    parser.add_argument(
        "--save-path",
        default="deep_uncertainty/figures/ddpn/artifacts/mnist_regression_with_rotation.pdf",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    produce_figure(
        gaussian_path=args.gaussian_path,
        poisson_path=args.poisson_path,
        nbinom_path=args.nbinom_path,
        ddpn_path=args.ddpn_path,
        save_path=args.save_path,
    )
