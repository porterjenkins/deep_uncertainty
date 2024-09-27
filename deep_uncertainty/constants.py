from collections import OrderedDict

HEADS_TO_NAMES = OrderedDict(
    {
        "nbinom": "NB DNN",
        "poisson": "Poisson DNN",
        "faithful_gaussian": "Stirn et al. ('23)",
        "gaussian": "Gaussian DNN",
        "seitzer": "Seitzer et al. ('23)",
        "natural_gaussian": "Immer et al. ('23)",
        "beta_ddpn_0.5": r"$\beta_{0.5}$-DDPN (Ours)",
        "beta_ddpn_1.0": r"$\beta_{1.0}$-DDPN (Ours)",
        "ddpn": "DDPN (Ours)",
    }
)

ENSEMBLE_HEADS_TO_NAMES = OrderedDict(
    {
        "nbinom": "NB Mixture",
        "poisson": "Poisson Mixture",
        "gaussian": "Laksh. et al. ('17)",
        "beta_ddpn_0.5": r"$\beta_{0.5}$-DDPN Mixture (Ours)",
        "beta_ddpn_1.0": r"$\beta_{1.0}$-DDPN Mixture (Ours)",
        "ddpn": "DDPN Mixture (Ours)",
    }
)
