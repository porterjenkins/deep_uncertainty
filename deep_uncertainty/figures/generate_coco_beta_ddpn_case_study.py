from argparse import ArgumentParser
from argparse import Namespace

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

from deep_uncertainty.datamodules import COCOPeopleDataModule
from deep_uncertainty.models import DoublePoissonNN
from deep_uncertainty.random_variables import DoublePoisson
from deep_uncertainty.utils.model_utils import get_hdi


def generate_case_study_plot(
    batches: list[tuple[torch.Tensor, tuple[str, int]]],
    model: DoublePoissonNN,
    xlims: list[tuple[int]],
) -> plt.Figure:
    """Generate a case study plot for the given DDPN model with Coco-People.

    Args:
        batches (list[tuple[torch.Tensor, tuple[str, int]]]): Batches containing the images to showcase (along with their image paths and labels).
        model (DoublePoissonNN): The DDPN model to showcase.
        xlims (list[tuple[int]]): List of xlims for the respective plots. Should have the same length as `batches`.

    Returns:
        plt.Figure: The resultant figure.
    """
    fig, axs = plt.subplots(
        2,
        len(batches),
        figsize=(3.5 * len(batches), 6),
        sharey="row",
        gridspec_kw={"height_ratios": [1.5, 1]},
    )
    iterable = zip(batches, axs.T, xlims)

    for batch, ax_column, xlim_vals in iterable:

        image_ax: plt.Axes = ax_column[0]
        dist_ax: plt.Axes = ax_column[1]

        image_tensor, (image_path, label) = batch
        image = plt.imread(image_path[0])
        image = Image.open(image_path[0]).resize(size=(450, 350), resample=Image.BILINEAR)
        mu, phi = (
            model._predict_impl(image_tensor.to(model.device)).flatten().detach().cpu().numpy()
        )
        ddpn_dist = DoublePoisson(mu, phi)
        lower, upper = get_hdi(ddpn_dist, 0.95)

        image_ax.imshow(image)
        image_ax.axis("off")

        support = np.arange(xlim_vals[0], xlim_vals[1] + 1)
        dist_ax.set_xlim(xlim_vals[0] - 0.3, xlim_vals[1] + 0.3)
        dist_ax.set_ylim(-0.01, 1.05)
        dist_ax.plot(support, ddpn_dist.pmf(support), ".-")
        dist_ax.scatter(label, ddpn_dist.pmf(label), marker="*", s=150, c="black", zorder=10)
        dist_ax.fill_between(
            x=support,
            y1=-0.01,
            y2=ddpn_dist.pmf(support),
            where=(support >= lower) & (support <= upper),
            color="gray",
            alpha=0.3,
        )
        dist_ax.annotate(
            xy=(0.6 * (xlim_vals[1] - xlim_vals[0]) + xlim_vals[0], 0.9),
            text=f"95% HDI: [{lower}, {upper}]",
        )
        for spine in dist_ax.spines.values():
            spine.set_edgecolor("gray")

    fig.tight_layout()
    return fig


def produce_figure(chkp_path: str, save_path: str, image_indices: list[int], xlims: list[int]):
    beta_ddpn_model = DoublePoissonNN.load_from_checkpoint(chkp_path)
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

    xlims = [(xlims[i], xlims[i + 1]) for i in range(0, len(xlims), 2)]
    fig = generate_case_study_plot(batches, beta_ddpn_model, xlims=xlims)
    fig.savefig(fname=save_path, format="pdf", dpi=150)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--chkp-path",
        type=str,
        default="chkp/coco_people_beta_ddpn_1.0/version_0/best_mae.ckpt",
        help="Filepath where the model weights are saved.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="deep_uncertainty/figures/artifacts/coco_people_beta_ddpn_case_study.pdf",
        help="Filepath where the resultant figure should be saved.",
    )
    parser.add_argument(
        "--image-indices",
        nargs="+",
        type=int,
        default=[5, 9, 185],
        help="Indices of the images in the test split of COCO-People to show. Specify as `idx_0 idx_1 ...`. Must be in ascending order.",
    )
    parser.add_argument(
        "--xlims",
        nargs="+",
        type=int,
        default=[0, 5, 6, 16, 0, 8],
        help="The xlims to set for the dist. plot with each image. Specify as `xmin_0 xmax_0 xmin_1 xmax_1 ...`.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    produce_figure(args.chkp_path, args.save_path, args.image_indices, args.xlims)
