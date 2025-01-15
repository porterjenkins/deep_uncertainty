from argparse import ArgumentParser

import pandas as pd
import torch
from sklearn.metrics import auc
from tqdm import tqdm


ID_UNCERTAINTIES_PATH = "results/reviews/id-ood/{head}_ensemble/reviews_uncertainties.pt"
OOD_UNCERTAINTIES_PATH = "results/reviews/id-ood/{head}_ensemble/bible_uncertainties.pt"
N = 158793  # Length of test split for Amazon Reviews


def eval_ensemble_ood(
    head: str, holdout_indices: torch.LongTensor, num_thresholds: int
) -> dict[str, torch.Tensor]:
    all_id_uncertainties: dict[str, torch.Tensor] = torch.load(
        ID_UNCERTAINTIES_PATH.format(head=head),
        map_location="cpu",
        weights_only=True,
    )
    all_ood_uncertainties: dict[str, torch.Tensor] = torch.load(
        OOD_UNCERTAINTIES_PATH.format(head=head),
        map_location="cpu",
        weights_only=True,
    )

    id_uncertainties = all_id_uncertainties["aleatoric"] + all_id_uncertainties["epistemic"]
    ood_uncertainties = all_ood_uncertainties["aleatoric"] + all_ood_uncertainties["epistemic"]

    id_holdout = id_uncertainties[holdout_indices]

    alpha_vals = torch.linspace(1e-3, 1, num_thresholds)
    precisions = []
    recalls = []
    fprs = []
    for alpha in alpha_vals:
        threshold = torch.quantile(id_holdout, 1 - alpha)

        tp = (ood_uncertainties > threshold).float().sum()
        fp = (id_uncertainties[~holdout_indices] > threshold).float().sum()
        fn = (ood_uncertainties <= threshold).float().sum()
        tn = (id_uncertainties[~holdout_indices] <= threshold).float().sum()

        tpr = tp / (tp + fn)
        fpr = fp / (tn + fp)

        if tp + fp == 0:
            precision = 1
        else:
            precision = tp / (tp + fp)

        precisions.append(precision)
        recalls.append(tpr)
        fprs.append(fpr)

    precisions = torch.tensor(precisions)
    recalls = torch.tensor(recalls)
    fprs = torch.tensor(fprs)
    return_dict = {
        "fprs": fprs,
        "precisions": precisions,
        "recalls": recalls,
        "auroc": auc(fprs, recalls),
        "aupr": auc(recalls, precisions),
    }
    return return_dict


def run_experiment(num_trials: int = 5, num_thresholds: int = 100, holdout_pct: float = 0.2):
    num_holdout = int(holdout_pct * N)
    results_df = {"head": [], "auroc": [], "aupr": [], "fpr80": []}
    for i in tqdm(range(num_trials), desc="Running permutation trials..."):
        holdout_indices = torch.randperm(N)[:num_holdout]

        for head in (
            "beta_ddpn_0.5",
            "beta_ddpn_1.0",
            "ddpn",
            "gaussian",
            "immer",
            "seitzer_0.5",
            "seitzer_1.0",
            "stirn",
            "poisson",
            "nbinom",
        ):
            metrics = eval_ensemble_ood(head, holdout_indices, num_thresholds)
            torch.save(metrics, f"results/reviews/id-ood/{head}_ensemble/ood_results_{i}.pt")
            results_df["head"].append(head)
            results_df["auroc"].append(metrics["auroc"])
            results_df["aupr"].append(metrics["aupr"])

            idx_of_80_tpr = torch.argmin((metrics["recalls"] - torch.tensor(0.8)).abs())
            results_df["fpr80"].append(metrics["fprs"][idx_of_80_tpr])

    results_df = (
        pd.DataFrame(results_df)
        .groupby("head")
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(by=("auroc", "mean"), ascending=False)
        .reset_index(drop=True)
    )
    results_df.to_csv("results/reviews/id-ood/aggregated_results.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--num-thresholds", type=int, default=100)
    parser.add_argument("--holdout-pct", type=float, default=0.2)
    args = parser.parse_args()
    run_experiment(args.num_trials, args.num_thresholds, args.holdout_pct)
