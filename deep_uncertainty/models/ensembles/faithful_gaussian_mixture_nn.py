from deep_uncertainty.models import FaithfulGaussianNN
from deep_uncertainty.models.ensembles.gaussian_mixture_nn import GaussianMixtureNN


class FaithfulGaussianMixtureNN(GaussianMixtureNN):
    """An ensemble of Faithful Gaussian NNs that outputs the predictive mean and variance of the implied uniform mixture.

    See https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html for details.
    """

    member_type = FaithfulGaussianNN
