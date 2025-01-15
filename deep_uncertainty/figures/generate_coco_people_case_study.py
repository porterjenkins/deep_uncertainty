import textwrap
from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path
from typing import Type

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

from deep_uncertainty.custom_datasets.coco_people_dataset import COCOPeopleDataset
from deep_uncertainty.enums import HeadType
from deep_uncertainty.models.discrete_regression_nn import DiscreteRegressionNN
from deep_uncertainty.models.ensembles import DoublePoissonMixtureNN
from deep_uncertainty.models.ensembles import FaithfulGaussianMixtureNN
from deep_uncertainty.models.ensembles import GaussianMixtureNN
from deep_uncertainty.models.ensembles import NaturalGaussianMixtureNN
from deep_uncertainty.models.ensembles import NegBinomMixtureNN
from deep_uncertainty.models.ensembles import PoissonMixtureNN
from deep_uncertainty.models.ensembles.deep_regression_ensemble import DeepRegressionEnsemble
from deep_uncertainty.utils.configs import EnsembleConfig
from deep_uncertainty.utils.configs import TrainingConfig
from deep_uncertainty.utils.experiment_utils import get_model


MAX_COUNT = 20
GAUSSIAN_HEADS = (HeadType.GAUSSIAN, HeadType.FAITHFUL_GAUSSIAN, HeadType.NATURAL_GAUSSIAN)
DEMO_INDICES = [10, 25, 95, 249]


def wrap_text(text, width=25):
    wrapper = textwrap.TextWrapper(width=width)
    return wrapper.fill(text)


def load_model(
    config: TrainingConfig | EnsembleConfig, head_type: HeadType, chkp_path: Path | None
) -> DiscreteRegressionNN | DeepRegressionEnsemble:
    if chkp_path is not None:
        initializer: Type[DiscreteRegressionNN] = get_model(config, return_initializer=True)[1]
        return initializer.load_from_checkpoint(chkp_path)
    else:
        model_map: dict[HeadType, DeepRegressionEnsemble] = {
            HeadType.GAUSSIAN: GaussianMixtureNN,
            HeadType.FAITHFUL_GAUSSIAN: FaithfulGaussianMixtureNN,
            HeadType.NATURAL_GAUSSIAN: NaturalGaussianMixtureNN,
            HeadType.DOUBLE_POISSON: DoublePoissonMixtureNN,
            HeadType.POISSON: PoissonMixtureNN,
            HeadType.NEGATIVE_BINOMIAL: NegBinomMixtureNN,
        }
        if head_type not in model_map:
            raise ValueError("Unsupported head type specified.")
        return model_map[head_type].from_config(config)


def plot_individual_predictive_dist(
    x: torch.Tensor,
    y: torch.Tensor,
    model: DiscreteRegressionNN,
    head_type: HeadType,
    ax: plt.Axes,
    **plot_kwargs
) -> int:
    y_hat: torch.Tensor = model.predict(x)
    dist = model.predictive_dist(y_hat)
    is_torch_dist = isinstance(dist, torch.distributions.Distribution)
    if isinstance(dist, torch.distributions.Normal):
        max_val = torch.round(dist.icdf(torch.tensor(0.999, device=y_hat.device))).item() + 5
    elif isinstance(dist, torch.distributions.Distribution):
        probs_over_support = torch.exp(
            dist.log_prob(torch.arange(2000, device=model.device))
        ).flatten()
        cdf_over_support = torch.cumsum(probs_over_support, dim=0)
        max_val = torch.searchsorted(cdf_over_support, 0.999).item() + 5
    else:
        max_val = dist.ppf(0.999) + 5

    if head_type in GAUSSIAN_HEADS:
        support = torch.linspace(0, max_val, steps=500, device=model.device)
        marker = None
        markersize = None
    else:
        support = torch.arange(0, max_val, device=model.device)
        marker = "o"
        markersize = 2

    if is_torch_dist:
        probs = torch.exp(dist.log_prob(support))
    else:
        probs = dist.pmf(support)
    ax.plot(support.cpu(), probs.cpu(), "-", marker=marker, markersize=markersize, **plot_kwargs)
    ax.scatter(y, 0, marker="*", color="black")
    return max_val


def plot_ensemble_predictive_dist(
    x: torch.Tensor,
    y: torch.Tensor,
    model: DeepRegressionEnsemble,
    head_type: HeadType,
    ax: plt.Axes,
    color,
):
    y_hat: torch.Tensor = model(x)
    if head_type in GAUSSIAN_HEADS:
        dist = torch.distributions.Normal(loc=y_hat[:, 0], scale=y_hat[:, 1].sqrt())
        max_val = torch.round(dist.icdf(torch.tensor(0.999, device=y_hat.device))).item() + 5
        support = torch.linspace(0, max_val, steps=500, device=model.members[0].device)
        probs = torch.exp(dist.log_prob(support))
        marker = None
        markersize = None
    else:
        probs, _ = y_hat
        cdf = torch.cumsum(probs.flatten(), dim=0)
        max_val = torch.searchsorted(cdf, 0.999).item() + 5
        support = torch.arange(0, max_val)
        probs = probs[:, :max_val].flatten()
        marker = "o"
        markersize = 2
    ax.plot(
        support.cpu(),
        probs.cpu(),
        marker=marker,
        linestyle="-",
        markersize=markersize,
        color=color,
    )

    max_vals = []
    for member in model.members:
        max_val = plot_individual_predictive_dist(
            x,
            y,
            member,
            head_type=head_type,
            ax=ax,
            color="lightgray",
            zorder=-1,
            alpha=0.8,
        )
        max_vals.append(max_val)
    new_max = min(max_vals)
    ax.set_xlim(-0.5, new_max + 0.5)


@torch.inference_mode()
def produce_figure(index: int, config_path: Path, chkp_path: Path | None, save_path: Path):
    is_ensemble = chkp_path is None
    config = (
        TrainingConfig.from_yaml(config_path)
        if not is_ensemble
        else EnsembleConfig.from_yaml(config_path)
    )
    head_type = config.head_type if not is_ensemble else config.member_head_type
    resize = Resize((224, 224))
    normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    to_tensor = ToTensor()
    transform = Compose([resize, to_tensor, normalize])

    fig, axs = plt.subplots(1, 2, figsize=(4, 2), width_ratios=[2.5, 1.5])
    dataset = COCOPeopleDataset(
        root_dir="data/coco-people", split="test", surface_image_path=True, transform=transform
    )

    model = load_model(config, head_type, chkp_path)
    device = model.device if isinstance(model, DiscreteRegressionNN) else model.members[0].device
    image_tensor, (image_path, count) = dataset[index]
    image_tensor = image_tensor.to(device).unsqueeze(0)

    axs[0].imshow(plt.imread(image_path))
    axs[0].axis("off")
    if isinstance(model, DiscreteRegressionNN):
        plot_individual_predictive_dist(
            x=image_tensor, y=count, model=model, head_type=head_type, ax=axs[1]
        )
    else:
        plot_ensemble_predictive_dist(
            x=image_tensor, y=count, model=model, head_type=head_type, ax=axs[1], color="tab:blue"
        )

    axs[1].set_ylim(-0.05, max(1.1, axs[1].get_ylim()[1]))
    axs[1].tick_params(axis="both", which="major", labelsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, format="pdf")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config-path", type=str, help="Path to model config.")
    parser.add_argument(
        "--chkp-path",
        type=str,
        default="",
        help="Path to .ckpt where model weights are saved. Do not specify if the model is an ensemble.",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--save-path",
        type=str,
        default="deep_uncertainty/figures/artifacts/coco_people_case_study.pdf",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    produce_figure(
        index=args.index,
        config_path=Path(args.config_path),
        chkp_path=Path(args.chkp_path) if args.chkp_path != "" else None,
        save_path=Path(args.save_path),
    )
