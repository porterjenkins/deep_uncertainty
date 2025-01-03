from deep_uncertainty.models import LogFaithfulGaussianNN
from deep_uncertainty.models.ensembles import FaithfulGaussianMixtureNN


class LogFaithfulGaussianMixtureNN(FaithfulGaussianMixtureNN):
    """A FaithfulGaussianMixtureNN where individual members were trained to output logmu, not mu (but they exponentiate this when their _predict_impl is called)."""

    member_type = LogFaithfulGaussianNN
