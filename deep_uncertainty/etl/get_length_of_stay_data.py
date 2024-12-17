import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import wget

from deep_uncertainty.data_generator import DataGenerator


def save_train_val_test_split(output_dir: str = "."):
    """Method to download and preprocess length-of-stay data from https://microsoft.github.io/r-server-hospital-length-of-stay/input_data.html.

    When this method has finished running, a `"length_of_stay.npz"` file will be in `output_dir` for later use, containing
    pre-computed train/test/val splits.

    Args:
        output_dir (str, optional): Directory to download files to. Defaults to ".".
    """
    breakpoint()
    output_dir = Path(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    URL = "https://raw.githubusercontent.com/microsoft/r-server-hospital-length-of-stay/refs/heads/master/Data/LengthOfStay.csv"

    csv_filepath = output_dir / "LengthOfStay.csv"
    if not os.path.exists(csv_filepath):
        print(f"\nDownloading length-of-stay data from {URL}")
        wget.download(URL, str(output_dir))

    numerical = [
        "hematocrit",
        "neutrophils",
        "sodium",
        "glucose",
        "bloodureanitro",
        "creatinine",
        "bmi",
        "pulse",
        "respiration",
    ]
    categorical = [
        "gender",
        "dialysisrenalendstage",
        "asthma",
        "irondef",
        "substancedependence",
        "psychologicaldisordermajor",
        "depress",
        "psychother",
        "fibrosisandother",
        "malnutrition",
        "hemo",
        "secondarydiagnosisnonicd9",
        "facid",
    ]
    target = ["lengthofstay"]

    df = pd.read_csv(csv_filepath)[["rcount"] + numerical + categorical + target]
    df["rcount"][df["rcount"] == "5+"] = 5
    df["rcount"] = df["rcount"].astype(float)
    df[numerical] = df[numerical].astype(float)
    df[numerical] = (df[numerical] - df[numerical].mean()) / df[numerical].std()
    df[categorical] = df[categorical].astype(str)
    one_hot = pd.get_dummies(df[categorical], drop_first=True).astype(float)
    df = df.drop(categorical, axis="columns")
    df = pd.concat([df, one_hot], axis="columns")

    X, y = df.loc[:, df.columns != target[0]].to_numpy(), df[target[0]].to_numpy()
    train_val_test_dict = DataGenerator.generate_train_val_test_split(
        lambda: (X, y), {}, random_seed=1998
    )
    np.savez(output_dir / "length_of_stay.npz", **train_val_test_dict)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Output dir to save data file to."
    )
    args = parser.parse_args()
    save_train_val_test_split(args.output_dir)
