import os
from argparse import ArgumentParser
from pathlib import Path
from shutil import unpack_archive

import numpy as np
import pandas as pd
import wget

from deep_uncertainty.data_generator import DataGenerator


def save_train_val_test_split(output_dir: str = "."):
    """Method to download and preprocess the Online Retail II dataset for regression tasks.

    When this method has finished running, a `"retail.npz"` file will be in `output_dir` for later use, containing
    pre-computed train/test/val splits.

    Args:
        output_dir (str, optional): Directory to download files to. Defaults to ".".
    """
    output_dir = Path(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"

    excel_filepath = output_dir / "online_retail_II.xlsx"
    if not os.path.exists(excel_filepath):
        print(f"\nDownloading retail data from {URL}")
        zip_filepath = Path(wget.download(URL, str(output_dir)))
        unpack_archive(zip_filepath, output_dir)

    print("Reading raw data...")
    df = pd.read_excel(
        excel_filepath, usecols=["StockCode", "InvoiceDate", "Price", "Country", "Quantity"]
    )
    df = df.rename(
        {
            "StockCode": "stock_code",
            "InvoiceDate": "invoice_date",
            "Price": "price",
            "Country": "country",
            "Quantity": "quantity",
        },
        axis="columns",
    )
    df["year"] = df["invoice_date"].dt.year
    df["month"] = df["invoice_date"].dt.month
    df["day"] = df["invoice_date"].dt.day
    df["hour"] = df["invoice_date"].dt.hour
    df = df.drop("invoice_date", axis="columns")

    print("Transforming features for easier ingestion...")
    standardize = ["price"]
    one_hot = ["stock_code", "country", "year"]
    periodize = ["month", "day", "hour"]
    df[standardize] = (df[standardize] - df[standardize].mean()) / df[standardize].std()
    df[periodize] = (
        2
        * np.pi
        * (df[periodize] - df[periodize].min())
        / (df[periodize].max() - df[periodize].min())
    )
    for col in periodize:
        df[f"{col}_sin"] = np.sin(df[col])
        df[f"{col}_cos"] = np.cos(df[col])
        df.drop(col, axis="columns", inplace=True)
    df = pd.get_dummies(df, columns=one_hot)

    print("Splitting into train/val/test...")
    features = [col for col in df.columns if col != "quantity"]
    X = df[features].values
    y = df["quantity"].values
    train_val_test_dict = DataGenerator.generate_train_val_test_split(
        lambda: (X, y), {}, random_seed=1998
    )
    print(f"Saving to {output_dir / 'retail.npz'}")
    np.savez(output_dir / "retail.npz", **train_val_test_dict)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Output dir to save data file to."
    )
    args = parser.parse_args()
    save_train_val_test_split(args.output_dir)
