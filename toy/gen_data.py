import numpy as np
import argparse
from scipy.stats import binom
import pandas as pd
from sklearn.model_selection import train_test_split
import os

from models.model_utils import get_binom_n, get_binom_p
from utils import get_yaml

def generate_gaussian_data(num_points, x_min, x_max):
    #np.random.seed(42)  # Set a seed for reproducibility

    x_values = np.random.uniform(x_min, x_max, num_points)
    mu_values = np.sin(x_values)
    sigma_values = 0.5 * (1 + np.exp(-x_values)) ** -1
    y_values = np.random.normal(mu_values, sigma_values ** 2)

    return x_values, y_values


def generate_linear_binom_data(num_points, x_min, x_max, beta_mu=1.5, beta_sig=1.1, bias_sig=-1.0):
    x_values = np.random.uniform(x_min, x_max, num_points)
    mu = beta_mu * x_values
    sig2 = beta_sig * x_values + bias_sig
    n = np.round(get_binom_n(mu=mu, sig2=sig2)).astype(int)
    p = get_binom_p(mu=mu, n=n)
    y = binom.rvs(n=n, p=p)
    return x_values, y


def generate_nonlinear_binom_data(num_points, x_min, x_max, beta_sig=1.1, bias_sig=-1.0):
    x_values = np.random.uniform(x_min, x_max, num_points)
    mu = np.sin(x_values)
    sig2 = beta_sig * x_values + bias_sig
    n = np.round(get_binom_n(mu=mu, sig2=sig2)).astype(int)
    p = get_binom_p(mu=mu, n=n)
    y = binom.rvs(n=n, p=p)
    return x_values, y


GENERATORS = {
    generate_gaussian_data.__name__: generate_gaussian_data,
    generate_linear_binom_data.__name__: generate_linear_binom_data,
    generate_nonlinear_binom_data.__name__: generate_nonlinear_binom_data,
}

def main(config: dict):

    if not os.path.exists(config["output_dir"]):
        os.mkdir(config["output_dir"])

    generator = GENERATORS[config["task"]]

    X_trn, y_trn = generator(
        num_points=config['n_train'],
        x_min=config['x_min'],
        x_max=config['x_max']
    )

    X_val, y_val = generator(
        num_points=config['n_val'],
        x_min=config['x_min'],
        x_max=config['x_max']
    )

    X_test, y_test = generator(
        num_points=config['n_test'],
        x_min=config['x_min'],
        x_max=config['x_max']
    )

    fpath = os.path.join(config['output_dir'], config["name"])

    np.savetxt(fname=fpath + '_x_train.txt', X=X_trn)
    np.savetxt(fname=fpath + '_y_train.txt', X=y_trn)

    np.savetxt(fname=fpath + '_x_val.txt', X=X_val)
    np.savetxt(fname=fpath + '_y_val.txt', X=y_val)

    np.savetxt(fname=fpath + '_x_test.txt', X=X_test)
    np.savetxt(fname=fpath + '_y_test.txt', X=y_test)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-cfg', type=str, default="./toy_data_config.yaml", help='Path to data config file')
    args = parser.parse_args()

    cfg = get_yaml(args.data_cfg)

    main(config=cfg["dataset"])



