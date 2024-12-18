import argparse
import os

import pandas as pd
import yaml


def main(dataset: str, results_dir: str):
    dataset_results_dir = os.path.join(results_dir, dataset)
    heads = os.listdir(dataset_results_dir)
    df = {"model": [], "mae": [], "nll": []}
    for head in heads:
        for i in range(5):
            metrics_path = os.path.join(
                dataset_results_dir, head, f"version_{i}/test_metrics.yaml"
            )
            if not os.path.exists(metrics_path):
                continue
            with open(metrics_path) as f:
                metrics = yaml.safe_load(f)
            nll = metrics["nll"]
            mae = metrics["test_mae"]
            df["model"].append(head)
            df["nll"].append(nll)
            df["mae"].append(mae)
    result = pd.DataFrame(df).groupby("model").agg(["mean", "std", "count"]).reset_index()
    result.to_csv(
        os.path.join(dataset_results_dir, "aggregated_results.csv"),
        float_format="%.3f",
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--results-dir")
    args = parser.parse_args()
    main(args.dataset, args.results_dir)
