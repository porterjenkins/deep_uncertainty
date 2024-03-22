import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from deep_uncertainty.data_generator import DataGenerator
from deep_uncertainty.utils.generic_utils import get_yaml


GENERATORS = {
    DataGenerator.generate_gaussian_sine_wave.__name__: DataGenerator.generate_gaussian_sine_wave,
    DataGenerator.generate_linear_binom_data.__name__: DataGenerator.generate_linear_binom_data,
    DataGenerator.generate_nonlinear_binom_data.__name__: DataGenerator.generate_nonlinear_binom_data,
    DataGenerator.generate_nonlinear_count_data.__name__: DataGenerator.generate_nonlinear_count_data,
    DataGenerator.generate_linear_count_data.__name__: DataGenerator.generate_linear_count_data,
}


def main(config: dict):

    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"])

    generation_function = GENERATORS[config["task"]]
    train_val_test_pcts = [config["train_pct"], config["val_pct"], config["test_pct"]]
    datasets = DataGenerator.generate_train_val_test_split(
        generation_function,
        config["generation_params"],
        split_pcts=train_val_test_pcts,
    )
    fpath = os.path.join(config["output_dir"], config["name"])
    np.savez(
        fpath + ".npz",
        X_train=datasets["X_train"],
        X_val=datasets["X_val"],
        X_test=datasets["X_test"],
        y_train=datasets["y_train"],
        y_val=datasets["y_val"],
        y_test=datasets["y_test"],
    )

    plt.scatter(datasets["X_train"], datasets["y_train"], label="TRAIN")
    plt.scatter(datasets["X_test"], datasets["y_test"], label="TEST")
    plt.legend()
    plt.savefig(fpath + "_scatter.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-cfg",
        type=str,
        default="./deep_uncertainty/toy/toy_data_config.yaml",
        help="Path to data config file",
    )
    args = parser.parse_args()

    cfg = get_yaml(args.data_cfg)
    main(config=cfg["dataset"])
