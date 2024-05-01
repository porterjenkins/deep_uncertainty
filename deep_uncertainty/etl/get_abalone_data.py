import os
from argparse import ArgumentParser
from pathlib import Path
from shutil import unpack_archive

import numpy as np
import pandas as pd
import wget

from deep_uncertainty.data_generator import DataGenerator


def save_train_val_test_split(output_dir: str = "."):
    """Method to download and preprocess the Abalone dataset for regression tasks.

    When this method has finished running, an `"abalone.npz"` file will be in `output_dir` for later use, containing
    pre-computed train/test/val splits.

    Args:
        output_dir (str, optional): Directory to download files to. Defaults to ".".
    """
    output_dir = Path(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    URL = "https://archive.ics.uci.edu/static/public/1/abalone.zip"

    csv_filepath = output_dir / "abalone.csv"
    if not os.path.exists(csv_filepath):
        print(f"\nDownloading abalone data from {URL}")
        zip_filepath = Path(wget.download(URL, str(output_dir)))
        unpack_archive(zip_filepath, output_dir)
        os.rename(csv_filepath.with_suffix(".data"), csv_filepath)

    column_names = [
        "sex",
        "length",
        "diameter",
        "height",
        "whole_weight",
        "shucked_weight",
        "viscera_weight",
        "shell_weight",
        "rings",
    ]
    standardize = column_names[1:-1]
    df = pd.read_csv(csv_filepath, names=column_names)
    df = pd.get_dummies(df, columns=["sex"], dtype=float)
    df[standardize] = (df[standardize] - df[standardize].mean()) / df[standardize].std()

    X, y = df.loc[:, df.columns != "rings"].to_numpy(), df.loc[:, df.columns == "rings"].to_numpy()
    train_val_test_dict = DataGenerator.generate_train_val_test_split(
        lambda: (X, y), {}, random_seed=1998
    )
    np.savez(output_dir / "abalone.npz", **train_val_test_dict)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Output dir to save data file to."
    )
    args = parser.parse_args()
    save_train_val_test_split(args.output_dir)
