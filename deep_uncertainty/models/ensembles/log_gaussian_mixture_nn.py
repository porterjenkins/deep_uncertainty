from deep_uncertainty.models import LogGaussianNN
from deep_uncertainty.models.ensembles import GaussianMixtureNN


class LogGaussianMixtureNN(GaussianMixtureNN):
    """A GaussianMixtureNN where individual members were trained to output logmu, not mu (but they exponentiate this when their _predict_impl is called)."""

    member_type = LogGaussianNN
