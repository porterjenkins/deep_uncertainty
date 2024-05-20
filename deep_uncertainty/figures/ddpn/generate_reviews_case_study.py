import textwrap
from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path
from typing import Type

import matplotlib.pyplot as plt
import torch
from seaborn import color_palette

from deep_uncertainty.datamodules import BibleDataModule
from deep_uncertainty.datamodules import ReviewsDataModule
from deep_uncertainty.enums import HeadType
from deep_uncertainty.models.discrete_regression_nn import DiscreteRegressionNN
from deep_uncertainty.models.ensembles import DoublePoissonMixtureNN
from deep_uncertainty.models.ensembles import GaussianMixtureNN
from deep_uncertainty.models.ensembles import NegBinomMixtureNN
from deep_uncertainty.models.ensembles import PoissonMixtureNN
from deep_uncertainty.random_variables import DoublePoisson
from deep_uncertainty.utils.configs import EnsembleConfig
from deep_uncertainty.utils.configs import TrainingConfig
from deep_uncertainty.utils.experiment_utils import get_model


def wrap_text(text, width=25):
    wrapper = textwrap.TextWrapper(width=width)
    wrapped_text = wrapper.fill(text)
    return wrapped_text


def produce_figure(config_path: Path, chkp_path: Path | None, save_path: Path, title: str):
    is_ensemble = chkp_path is None
    palette = color_palette()
    id_color, ood_color = palette[0], palette[1]

    config = (
        TrainingConfig.from_yaml(config_path)
        if not is_ensemble
        else EnsembleConfig.from_yaml(config_path)
    )
    batch_size = 1
    num_workers = 0
    head_type = config.head_type if not is_ensemble else config.member_head_type

    demo_reviews_indices = [0, 5, 9, 15]
    demo_bible_indices = [105, 1654, 419, 4809]
    fig, axs = plt.subplots(4, len(demo_reviews_indices), figsize=(8, 5), sharey="row")

    reviews_datamodule = ReviewsDataModule(
        root_dir="./data/reviews",
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=False,
    )
    bible_datamodule = BibleDataModule(
        root_dir="./data/bible",
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=False,
    )
    reviews_datamodule.setup("test")
    bible_datamodule.setup("test")
    reviews_loader = reviews_datamodule.test_dataloader()
    bible_loader = bible_datamodule.test_dataloader()

    if not is_ensemble:
        initializer: Type[DiscreteRegressionNN] = get_model(config, return_initializer=True)[1]
        model = initializer.load_from_checkpoint(chkp_path)
        device = model.device
    else:
        if head_type == HeadType.GAUSSIAN:
            model = GaussianMixtureNN.from_config(config)
        elif head_type == HeadType.DOUBLE_POISSON:
            model = DoublePoissonMixtureNN.from_config(config)
        elif head_type == HeadType.POISSON:
            model = PoissonMixtureNN.from_config(config)
        elif head_type == HeadType.NEGATIVE_BINOMIAL:
            model = NegBinomMixtureNN.from_config(config)
        device = model.members[0].device

    eps = torch.tensor(1e-6, device=device)

    with torch.inference_mode():
        max_val = 500
        disc_support = torch.arange(0, 6, device=device)
        cont_support = torch.linspace(0, 5, steps=500, device=device)

        ax_counter = 0
        for i, (batch_encoding, _) in enumerate(reviews_loader):
            if i >= 120:
                break
            if i in demo_reviews_indices:
                review_text_ax: plt.Axes = axs[0, ax_counter]
                review_pmf_ax: plt.Axes = axs[1, ax_counter]
                ax_counter += 1

                batch_encoding = batch_encoding.to(device)
                y_hat = model._predict_impl(batch_encoding)
                verse = reviews_datamodule.test[i][0]
                review_text_ax.text(
                    x=0,
                    y=0.5,
                    s=wrap_text(f'"{verse}"'),
                    horizontalalignment="left",
                    verticalalignment="center",
                    fontsize=8,
                    fontdict=dict(family="serif"),
                )
                review_text_ax.axis("off")

                if head_type in (HeadType.GAUSSIAN, HeadType.FAITHFUL_GAUSSIAN):
                    mu, var = torch.split(y_hat, [1, 1], dim=-1)
                    mu = mu.flatten()
                    var = var.flatten()
                    dist = torch.distributions.Normal(mu, var.sqrt())
                    review_pmf_ax.plot(
                        cont_support.detach().cpu().numpy(),
                        torch.exp(dist.log_prob(cont_support)).detach().cpu().numpy(),
                        color=id_color,
                    )
                elif head_type == HeadType.NATURAL_GAUSSIAN:
                    eta_1, eta_2 = torch.split(y_hat, [1, 1], dim=-1)
                    mu = -0.5 * (eta_1 / eta_2)
                    var = -0.5 * torch.reciprocal(eta_2)
                    mu = mu.flatten()
                    var = var.flatten()
                    dist = torch.distributions.Normal(mu, var.sqrt())
                    review_pmf_ax.plot(
                        cont_support.detach().cpu().numpy(),
                        torch.exp(dist.log_prob(cont_support)).detach().cpu().numpy(),
                        color=id_color,
                    )
                elif not is_ensemble:
                    if head_type == HeadType.DOUBLE_POISSON:
                        mu, phi = torch.split(y_hat, [1, 1], dim=-1)
                        mu = mu.flatten()
                        phi = phi.flatten()
                        dist = DoublePoisson(mu, phi)
                        probs = dist.pmf(disc_support)
                    elif head_type == HeadType.NEGATIVE_BINOMIAL:
                        mu, alpha = torch.split(y_hat, [1, 1], dim=-1)
                        mu = mu.flatten()
                        alpha = alpha.flatten()
                        var = mu + alpha * mu**2
                        p = mu / torch.maximum(var, eps)
                        failure_prob = torch.minimum(1 - p, 1 - eps)
                        n = mu**2 / torch.maximum(var - mu, eps)
                        dist = torch.distributions.NegativeBinomial(
                            total_count=n, probs=failure_prob
                        )
                        probs = torch.exp(dist.log_prob(disc_support))
                    elif head_type == HeadType.POISSON:
                        lmbda = y_hat
                        dist = torch.distributions.Poisson(rate=lmbda.flatten())
                        probs = torch.exp(dist.log_prob(disc_support))
                    review_pmf_ax.plot(
                        disc_support.detach().cpu().numpy(),
                        probs.detach().cpu().numpy(),
                        ".-",
                        color=id_color,
                    )
                else:
                    if head_type in (
                        HeadType.DOUBLE_POISSON,
                        HeadType.POISSON,
                        HeadType.NEGATIVE_BINOMIAL,
                    ):
                        probs = y_hat[:, :max_val].permute(1, 0)
                        review_pmf_ax.plot(
                            disc_support.detach().cpu().numpy(),
                            probs[:6].detach().cpu().numpy(),
                            ".-",
                            color=id_color,
                        )

                review_pmf_ax.set_xticks([0, 1, 2, 3, 4, 5])

        ax_counter = 0
        for i, (batch_encoding, _) in enumerate(bible_loader):
            if i >= 5000:
                break
            if i in demo_bible_indices:
                verse_ax: plt.Axes = axs[2, ax_counter]
                rating_pmf_ax: plt.Axes = axs[3, ax_counter]
                ax_counter += 1

                batch_encoding = batch_encoding.to(device)
                y_hat = model._predict_impl(batch_encoding)
                verse = bible_datamodule.full[i][0]
                verse_ax.text(
                    x=0,
                    y=0.5,
                    s=wrap_text(f'"{verse}"'),
                    horizontalalignment="left",
                    verticalalignment="center",
                    fontsize=8,
                    fontdict=dict(family="serif"),
                )
                verse_ax.axis("off")

                if head_type in (HeadType.GAUSSIAN, HeadType.FAITHFUL_GAUSSIAN):
                    mu, var = torch.split(y_hat, [1, 1], dim=-1)
                    mu = mu.flatten()
                    var = var.flatten()
                    dist = torch.distributions.Normal(mu, var.sqrt())
                    rating_pmf_ax.plot(
                        cont_support.detach().cpu().numpy(),
                        torch.exp(dist.log_prob(cont_support)).detach().cpu().numpy(),
                        color=ood_color,
                    )
                elif head_type == HeadType.NATURAL_GAUSSIAN:
                    eta_1, eta_2 = torch.split(y_hat, [1, 1], dim=-1)
                    mu = -0.5 * (eta_1 / eta_2)
                    var = -0.5 * torch.reciprocal(eta_2)
                    mu = mu.flatten()
                    var = var.flatten()
                    dist = torch.distributions.Normal(mu, var.sqrt())
                    rating_pmf_ax.plot(
                        cont_support.detach().cpu().numpy(),
                        torch.exp(dist.log_prob(cont_support)).detach().cpu().numpy(),
                        color=ood_color,
                    )
                elif not is_ensemble:
                    if head_type == HeadType.DOUBLE_POISSON:
                        mu, phi = torch.split(y_hat, [1, 1], dim=-1)
                        mu = mu.flatten()
                        phi = phi.flatten()
                        dist = DoublePoisson(mu, phi)
                        probs = dist.pmf(disc_support)
                    elif head_type == HeadType.NEGATIVE_BINOMIAL:
                        mu, alpha = torch.split(y_hat, [1, 1], dim=-1)
                        mu = mu.flatten()
                        alpha = alpha.flatten()
                        var = mu + alpha * mu**2
                        p = mu / torch.maximum(var, eps)
                        failure_prob = torch.minimum(1 - p, 1 - eps)
                        n = mu**2 / torch.maximum(var - mu, eps)
                        dist = torch.distributions.NegativeBinomial(
                            total_count=n, probs=failure_prob
                        )
                        probs = torch.exp(dist.log_prob(disc_support))
                    elif head_type == HeadType.POISSON:
                        lmbda = y_hat
                        dist = torch.distributions.Poisson(rate=lmbda.flatten())
                        probs = torch.exp(dist.log_prob(disc_support))
                    rating_pmf_ax.plot(
                        disc_support.detach().cpu().numpy(),
                        probs.detach().cpu().numpy(),
                        ".-",
                        color=ood_color,
                    )
                else:
                    if head_type in (
                        HeadType.DOUBLE_POISSON,
                        HeadType.POISSON,
                        HeadType.NEGATIVE_BINOMIAL,
                    ):
                        probs = y_hat[:, :max_val].permute(1, 0)
                        rating_pmf_ax.plot(
                            disc_support.detach().cpu().numpy(),
                            probs[:6].detach().cpu().numpy(),
                            ".-",
                            color=ood_color,
                        )
                rating_pmf_ax.set_xticks([0, 1, 2, 3, 4, 5])

    lower = -0.1
    upper = max(review_pmf_ax.get_ylim()[1], 1.1)
    review_pmf_ax.set_ylim(lower, upper)

    lower = -0.1
    upper = max(rating_pmf_ax.get_ylim()[1], 1.1)
    rating_pmf_ax.set_ylim(lower, upper)

    fig.suptitle(title, fontsize=12)
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
    parser.add_argument(
        "--save-path",
        type=str,
        default="deep_uncertainty/figures/ddpn/artifacts/reviews_case_study.pdf",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Reviews Case Study",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    produce_figure(
        config_path=Path(args.config_path),
        chkp_path=Path(args.chkp_path) if args.chkp_path != "" else None,
        save_path=Path(args.save_path),
        title=args.title,
    )
