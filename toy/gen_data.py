import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

from models.model_utils import get_binom_n, get_binom_p

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