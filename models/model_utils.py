import numpy as np

def get_binom_p(mu: np.ndarray, n: np.ndarray):
    """
    Derive the binomial p parameter from the mean, and the n parameter
    :param mu: array of mu's
    :param n: array of n's
    :return: p: np.array
    """
    return mu / n

def get_binom_n(mu: np.array, sig2: np.array):
    """
    Get the binomial n parameter from mu and sig2
    :param mu: array of means
    :param sig2: array of variances
    :return: n: np.array
    """

    return mu/(1-(sig2/mu))