from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch

heads_names = {
    "ddpn": "DDPN (Ours)",
    "beta_ddpn_0.5": "$\\beta_{0.5}$-DDPN (Ours)",
    "beta_ddpn_1.0": "$\\beta_{1.0}$-DDPN (Ours)",
    "gaussian": "Gaussian DNN",
    "stirn": "Faithful Gaussian",
    "immer": "Natural Gaussian",
    "seitzer_0.5": "$\\beta_{0.5}$-Gaussian",
    "seitzer_1.0": "$\\beta_{1.0}$-Gaussian",
    "poisson": "Poisson DNN",
    "nbinom": "NB DNN",
}


def generate_plot(models: list[str], trial: int, prefix: str):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for model in models:
        all_results = torch.load(f"results/reviews/id-ood/{model}_ensemble/ood_results_{trial}.pt")
        fprs = all_results["fprs"]
        recalls = all_results["recalls"]
        ax.plot(fprs, recalls, label=heads_names[model])
    baseline = torch.linspace(0, 1, 50)
    ax.plot(baseline, baseline, linestyle="--", color="gray", label="Random")
    ax.set_ylabel("TPR")
    ax.set_xlabel("FPR")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"deep_uncertainty/figures/artifacts/{prefix}_roc_curves.pdf", dpi=150)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--trial", type=int, choices=list(range(10)))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(heads_names.keys()),
        help="List of models to process",
        required=True,
    )
    parser.add_argument("--prefix", type=str)
    args = parser.parse_args()
    generate_plot(args.models, args.trial, args.prefix)
