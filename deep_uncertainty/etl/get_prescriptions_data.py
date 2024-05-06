import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import wget

from deep_uncertainty.data_generator import DataGenerator


def save_train_val_test_split(output_dir: str = "."):
    """Method to download and preprocess data from the Prescriptions dataset for regression tasks.

    When this method has finished running, a `"prescriptions.npz"` file will be in `output_dir` for later use, containing
    pre-computed train/val/test splits.

    Args:
        output_dir (str, optional): Directory to download files to. Defaults to ".".
    """
    output_dir = Path(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    csv_filepath = output_dir / "prescriptions.csv"
    if not os.path.exists(csv_filepath):
        FILE_ID = "1XbyJDk3YjBRZxjIOoGdfhhKhhqcbIcYd"  # Publicly hosted on Spencer's Google Drive.
        URL = f"https://docs.google.com/uc?export=download&id={FILE_ID}"
        csv_filepath = wget.download(URL, str(output_dir))

    prescriptions_df = pd.read_csv(csv_filepath)
    standardize = ["num_sales_calls", "num_ordered_last_month", "mean_samples_given"]
    prescriptions_df[standardize] = (
        prescriptions_df[standardize] - prescriptions_df[standardize].mean()
    ) / prescriptions_df[standardize].std()
    upper = prescriptions_df["num_ordered"].quantile(0.99)
    prescriptions_df = prescriptions_df[prescriptions_df["num_ordered"] <= upper]

    X, y = (
        prescriptions_df.loc[:, prescriptions_df.columns != "num_ordered"].to_numpy(),
        prescriptions_df["num_ordered"].to_numpy(),
    )
    train_val_test_dict = DataGenerator.generate_train_val_test_split(
        lambda: (X, y), {}, random_seed=1998
    )

    np.savez(output_dir / "prescriptions.npz", **train_val_test_dict)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Output dir to save data file to."
    )
    args = parser.parse_args()
    save_train_val_test_split(args.output_dir)
