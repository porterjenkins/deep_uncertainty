import textwrap
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Type

import matplotlib.pyplot as plt
import torch
from seaborn import color_palette

from deep_uncertainty.datamodules import BibleDataModule, ReviewsDataModule
from deep_uncertainty.enums import HeadType
from deep_uncertainty.models.discrete_regression_nn import DiscreteRegressionNN
from deep_uncertainty.models.ensembles import (
    DoublePoissonMixtureNN, GaussianMixtureNN, FaithfulGaussianMixtureNN,
    NaturalGaussianMixtureNN, NegBinomMixtureNN, PoissonMixtureNN
)
from deep_uncertainty.models.ensembles.deep_regression_ensemble import DeepRegressionEnsemble
from deep_uncertainty.utils.configs import EnsembleConfig, TrainingConfig
from deep_uncertainty.utils.experiment_utils import get_model


MAX_RATING = 5
GAUSSIAN_HEADS = (HeadType.GAUSSIAN, HeadType.FAITHFUL_GAUSSIAN, HeadType.NATURAL_GAUSSIAN)
DEMO_REVIEWS_INDICES = [10, 25, 95, 249]
DEMO_BIBLE_INDICES = [105, 419, 1654, 4809]


def wrap_text(text, width=25):
    wrapper = textwrap.TextWrapper(width=width)
    return wrapper.fill(text)


def load_model(
    config: TrainingConfig | EnsembleConfig,
    head_type: HeadType,
    chkp_path: Path | None
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
            HeadType.NEGATIVE_BINOMIAL: NegBinomMixtureNN
        }
        if head_type not in model_map:
            raise ValueError("Unsupported head type specified.")
        return model_map[head_type].from_config(config)


def plot_individual_predictive_dist(x: torch.Tensor, model: DiscreteRegressionNN, head_type: HeadType, ax: plt.Axes, **plot_kwargs):
    y_hat: torch.Tensor = model.predict(x)
    dist = model.predictive_dist(y_hat)
    if head_type in GAUSSIAN_HEADS:
        support = torch.linspace(0, MAX_RATING, steps=500, device=model.device)
        linestyle = "-"
    else:
        support = torch.arange(0, MAX_RATING + 1, device=model.device)
        linestyle = ".-"
    
    try:
        probs = torch.exp(dist.log_prob(support))
    except Exception:
        probs = dist.pmf(support)
    ax.plot(support.cpu(), probs.cpu(), linestyle, **plot_kwargs)


def plot_ensemble_predictive_dist(x: torch.Tensor, model: DeepRegressionEnsemble, head_type: HeadType, ax: plt.Axes, color):
    y_hat: torch.Tensor = model(x)
    if head_type in GAUSSIAN_HEADS:
        dist = torch.distributions.Normal(loc=y_hat[:, 0], scale=y_hat[:, 1].sqrt())
        support = torch.linspace(0, MAX_RATING, steps=500, device=model.members[0].device)
        probs = torch.exp(dist.log_prob(support))
        linestyle = "-"
    else:
        support = torch.arange(0, MAX_RATING + 1)
        probs, _ = y_hat
        probs = probs[:, :MAX_RATING + 1].flatten()
        linestyle = ".-"
    ax.plot(support.cpu(), probs.cpu(), linestyle, color=color)

    for member in model.members:
        plot_individual_predictive_dist(x, member, head_type=head_type, ax=ax, color="lightgray", zorder=-1, alpha=0.8)


def plot_predictive_dists(
    model: DiscreteRegressionNN | DeepRegressionEnsemble,
    datamodule: ReviewsDataModule | BibleDataModule,
    ax_row: int,
    head_type: HeadType,
    demo_samples_indices: list[int],
):
    palette = color_palette()
    id_color, ood_color = palette[0], palette[1]
    device = model.device if isinstance(model, DiscreteRegressionNN) else model.members[0].device
    ax_counter = 0
    datamodule.setup("test")
    loader = datamodule.test_dataloader()
    demo_type = "reviews" if isinstance(datamodule, ReviewsDataModule) else "bible"

    for i, (batch_encoding, _) in enumerate(loader):
        if i > max(demo_samples_indices):
            break
        if i in demo_samples_indices:
            text_ax: plt.Axes = ax_row[0, ax_counter]
            prob_ax: plt.Axes = ax_row[1, ax_counter]
            ax_counter += 1

            batch_encoding = batch_encoding.to(device)

            if demo_type == "reviews":
                text = datamodule.test[i][0]
            else:
                text = datamodule.full[i][0]
            text_ax.text(
                x=0,
                y=0.5,
                s=wrap_text(f'"{text}"'),
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=8,
                fontdict=dict(family="serif"),
            )
            text_ax.axis("off")

            color = id_color if demo_type == "reviews" else ood_color
            if isinstance(model, DiscreteRegressionNN):
                plot_individual_predictive_dist(x=batch_encoding, model=model, head_type=head_type, ax=prob_ax, color=color)
            else:
                plot_ensemble_predictive_dist(x=batch_encoding, model=model, head_type=head_type, ax=prob_ax, color=color)
            
            lower = -0.1
            upper = max(prob_ax.get_ylim()[1], 1.1)
            prob_ax.set_ylim(lower, upper)
            prob_ax.set_xticks(list(range(MAX_RATING + 1)))


@torch.inference_mode()
def produce_figure(config_path: Path, chkp_path: Path | None, save_path: Path):
    is_ensemble = chkp_path is None
    config = TrainingConfig.from_yaml(config_path) if not is_ensemble else EnsembleConfig.from_yaml(config_path)
    head_type = config.head_type if not is_ensemble else config.member_head_type

    fig, axs = plt.subplots(4, len(DEMO_REVIEWS_INDICES), figsize=(8, 5), sharey="row")

    reviews_datamodule = ReviewsDataModule(root_dir="./data/amazon-reviews", batch_size=1, num_workers=0, persistent_workers=False)
    bible_datamodule = BibleDataModule(root_dir="./data/bible", batch_size=1, num_workers=0, persistent_workers=False)

    model = load_model(config, head_type, chkp_path)
    plot_predictive_dists(
        model=model,
        datamodule=reviews_datamodule,
        ax_row=axs[:2],
        head_type=head_type,
        demo_samples_indices=DEMO_REVIEWS_INDICES,
    )
    plot_predictive_dists(
        model=model,
        datamodule=bible_datamodule,
        ax_row=axs[2:],
        head_type=head_type,
        demo_samples_indices=DEMO_BIBLE_INDICES,
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, format="pdf")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config-path", type=str, help="Path to model config.")
    parser.add_argument("--chkp-path", type=str, default="", help="Path to .ckpt where model weights are saved. Do not specify if the model is an ensemble.")
    parser.add_argument("--save-path", type=str, default="deep_uncertainty/figures/artifacts/reviews_case_study.pdf")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    produce_figure(
        config_path=Path(args.config_path),
        chkp_path=Path(args.chkp_path) if args.chkp_path != "" else None,
        save_path=Path(args.save_path),
    )
