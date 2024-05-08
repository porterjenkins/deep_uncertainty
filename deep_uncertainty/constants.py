from collections import OrderedDict

HEADS_TO_NAMES = OrderedDict(
    {
        "nbinom": "NB DNN",
        "poisson": "Poisson DNN",
        "faithful_gaussian": "Stirn et al. ('23)",
        "beta_gaussian": "Seitzer et al. ('23)",
        "gaussian": "Gaussian DNN",
        "natural_gaussian": "Immer et al. ('23)",
        "beta_ddpn": r"$\beta$-DDPN (Ours)",
        "ddpn": "DDPN (Ours)",
    }
)

ENSEMBLE_HEADS_TO_NAMES = OrderedDict(
    {
        "nbinom": "NB Mixture",
        "poisson": "Poisson Mixture",
        "gaussian": "Laksh. et al. ('17)",
        "beta_ddpn": r"$\beta$-DDPN Mixture (Ours)",
        "ddpn": "DDPN Mixture (Ours)",
    }
)
