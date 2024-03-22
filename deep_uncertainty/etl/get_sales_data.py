import os
from argparse import ArgumentParser
from pathlib import Path
from shutil import unpack_archive

import numpy as np
import polars as pl
import wget
from pandas import read_stata

from deep_uncertainty.data_generator import DataGenerator


def _get_detergent_sales_df(output_dir: str = ".") -> pl.DataFrame:
    """Download and clean detergent sales data from the Dominick's dataset.

    Only rows from the top 3 most common detergent products are kept (for shorter training feedback loops), and we eliminate outlier counts.
    Additionally, only the columns "store", "upc", "week", and "move" are retained.

    Args:
        output_dir (str, optional): Directory to download data files to. Defaults to ".".

    Returns:
        pl.DataFrame: Detergent sales dataframe.
    """
    URL = "https://www.chicagobooth.edu/-/media/enterprise/centers/kilts/datasets/dominicks-dataset/movement_csv-files/wlnd.zip"
    csv_filepath = Path(output_dir) / "wlnd.csv"
    if not os.path.exists(csv_filepath):
        print(f"\nDownloading detergent sales data from {URL}")
        zip_filepath = Path(wget.download(URL, output_dir))
        unpack_archive(zip_filepath, output_dir)
        csv_filepath = zip_filepath.with_suffix(".csv")
    schema = {
        "STORE": pl.Int64,
        "UPC": pl.String,
        "WEEK": pl.Int64,
        "MOVE": pl.Int64,
        "OK": pl.Int64,  # Indicates if data is fishy or not.
    }
    detergent_df = pl.read_csv(csv_filepath, columns=list(schema.keys()), dtypes=schema)
    top_3_upcs = detergent_df["UPC"].value_counts().sort(by="count", descending=True)["UPC"][:3]
    detergent_df = (
        detergent_df.filter(pl.col("OK") == 1, pl.col("UPC").is_in(top_3_upcs)).drop("OK")
    ).select(pl.all().name.to_lowercase())
    mean_demand = detergent_df["move"].mean()
    std_demand = detergent_df["move"].std()
    lower, upper = mean_demand - 3 * std_demand, mean_demand + 3 * std_demand
    detergent_df.filter(pl.col("move").is_between(lower, upper))
    return detergent_df


def _get_store_demographics_df(output_dir: str = ".") -> pl.DataFrame:
    """Download and clean store demographics data.

    Rows containing null values are dropped, and only the columns mentioned in "Part 2: Store-Specific Demographics" of the dataset manual are kept.

    See https://www.chicagobooth.edu/-/media/enterprise/centers/kilts/datasets/dominicks-dataset/dominicks-manual-and-codebook_kiltscenter for the manual.

    Args:
        output_dir (str, optional): Directory to download data files to. Defaults to ".".

    Returns:
        pl.DataFrame: Store demographics dataframe.
    """
    URL = "https://www.chicagobooth.edu/boothsitecore/docs/dff/store-demos-customer-count/demo_stata.zip"
    stata_filepath = Path(output_dir) / "demo.dta"
    if not os.path.exists(stata_filepath):
        print(f"\nDownloading store demographics data from {URL}")
        zip_filepath = Path(wget.download(URL, output_dir))
        unpack_archive(zip_filepath, output_dir)
        stata_filepath = zip_filepath.parent / "demo.dta"
    demo_columns = [
        "store",
        "age9",
        "age60",
        "ethnic",
        "educ",
        "nocar",
        "income",
        "incsigma",
        "hsizeavg",
        "hsize1",
        "hsize2",
        "hsize34",
        "hsize567",
        "hh3plus",
        "hh4plus",
        "hhsingle",
        "hhlarge",
        "workwom",
        "sinhouse",
        "density",
        "hval150",
        "hval200",
        "hvalmean",
        "single",
        "retired",
        "unemp",
        "wrkch5",
        "wrkch17",
        "nwrkch5",
        "nwrkch17",
        "wrkch",
        "nwrkch",
        "wrkwch",
        "wrkwnch",
        "telephn",
        "mortgage",
        "nwhite",
        "poverty",
        "shopindx",
    ]
    demo_df = read_stata(stata_filepath)[demo_columns].dropna()
    demo_df["store"] = demo_df["store"].astype(int)
    return pl.DataFrame(demo_df)


def _combine_demographics_and_detergent_df(
    demographics_df: pl.DataFrame, detergent_df: pl.DataFrame
) -> pl.DataFrame:
    """Join the demographics and detergent dataframes where a store link exists, then preprocess for count regression.

    Preprocessing involves one-hot encoding categorical features and standardizing continuous ones.

    Args:
        demographics_df (pl.DataFrame): DataFrame containing store demographics information.
        detergent_df (pl.DataFrame): DataFrame containing detergent sales information.

    Returns:
        pl.DataFrame: Combined dataframe (with preprocessed predictive features).
    """
    combined_df = (
        demographics_df.join(detergent_df, on="store", how="inner").drop("store").to_dummies("upc")
    )
    columns_to_standardize = pl.all().exclude("^upc.*$", "move")
    combined_df = combined_df.select(
        (columns_to_standardize - columns_to_standardize.mean()) / columns_to_standardize.std(),
        pl.col("^upc.*$", "move"),
    )
    return combined_df


def save_train_val_test_split(output_dir: str = "."):
    """Method to download and preprocess detergent sales data from the Dominick's dataset for count regression tasks.

    When this method has finished running, a `"sales.npz"` file will be in `output_dir` for later use, containing
    pre-computed train/test/val splits.

    Args:
        output_dir (str, optional): Directory to download files to. Defaults to ".".
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    demo_df = _get_store_demographics_df(output_dir)
    detergent_df = _get_detergent_sales_df(output_dir)
    combined_df = _combine_demographics_and_detergent_df(demo_df, detergent_df)

    def df_to_X_y_array(df: pl.DataFrame):
        arr = df.to_numpy()
        return arr[:, :-1], arr[:, -1]

    train_val_test_dict = DataGenerator.generate_train_val_test_split(
        df_to_X_y_array, {"df": combined_df}, random_seed=1998
    )

    output_file = Path(output_dir) / "sales.npz"
    np.savez(output_file, **train_val_test_dict)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Output dir to save data file to."
    )
    args = parser.parse_args()
    save_train_val_test_split(args.output_dir)
