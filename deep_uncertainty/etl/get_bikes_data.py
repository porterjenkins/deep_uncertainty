import os
from argparse import ArgumentParser
from pathlib import Path
from shutil import unpack_archive

import numpy as np
import pandas as pd
import wget

from deep_uncertainty.data_generator import DataGenerator


def save_train_val_test_split(output_dir: str = "."):
    """Method to download and preprocess rental bike counts from the Bike Sharing dataset for regression tasks.

    When this method has finished running, a `"bikes.npz"` file will be in `output_dir` for later use, containing
    pre-computed train/test/val splits.

    Args:
        output_dir (str, optional): Directory to download files to. Defaults to ".".
    """
    output_dir = Path(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"

    csv_filepath = output_dir / "hour.csv"
    if not os.path.exists(csv_filepath):
        print(f"\nDownloading bikes data from {URL}")
        zip_filepath = Path(wget.download(URL, str(output_dir)))
        unpack_archive(zip_filepath, output_dir)

    keep = [
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "cnt",
    ]
    standardize = ["weathersit", "weekday", "temp", "atemp", "hum", "windspeed"]
    periodize = ["season", "mnth", "hr"]
    bikes_df = pd.read_csv(csv_filepath)[keep]
    bikes_df[standardize] = (bikes_df[standardize] - bikes_df[standardize].mean()) / bikes_df[
        standardize
    ].std()
    bikes_df[periodize] = (
        2
        * np.pi
        * (bikes_df[periodize] - bikes_df[periodize].min())
        / (bikes_df[periodize].max() - bikes_df[periodize].min())
    )
    for col in periodize:
        bikes_df[f"{col}_sin"] = np.sin(bikes_df[col])
        bikes_df[f"{col}_cos"] = np.cos(bikes_df[col])
        bikes_df.drop(col, axis="columns", inplace=True)

    X, y = bikes_df.loc[:, bikes_df.columns != "cnt"].to_numpy(), bikes_df["cnt"].to_numpy()
    train_val_test_dict = DataGenerator.generate_train_test_val_split(
        lambda: (X, y), {}, random_seed=1998
    )
    np.savez(output_dir / "bikes.npz", **train_val_test_dict)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Output dir to save data file to."
    )
    args = parser.parse_args()
    save_train_val_test_split(args.output_dir)
