import os
from argparse import ArgumentParser
from zipfile import ZipFile

import gdown


def download_file_with_wget(file_id, output_filename):
    """Method to download the data from the UK collision.

    The data has been pre-processed including the feature selection and one-hot encoding.
    The original data source is from: https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data (2022)

    Args:
        output_dir (str, optional): Directory to download files to. Defaults to ".".
    """
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    command = f"wget --no-check-certificate '{download_url}' -O {output_filename}"
    os.system(command)


def unzip_file(zip_path, extract_to):
    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Unzipped files in {extract_to}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./UK_collision/",
        help="Output dir to save data file to.",
    )
    args = parser.parse_args()

    url = "https://drive.google.com/uc?id=1f3wJHTC-o2jFhZgQzxF6v0Oj85xAlenA"
    output = "UK_collision.zip"
    gdown.download(url, output, quiet=False)

    unzip_file(output, args.output_dir)
