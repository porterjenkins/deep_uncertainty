import numpy as np
import torch
from scipy.stats import entropy
from torch.distributions import Poisson

from deep_uncertainty.evaluation.utils import calculate_entropy


def test_torch_entropy_equals_scipy():
    dist = Poisson(rate=torch.tensor([7, 3, 5, 99]))
    support = torch.arange(0, 2000).reshape(-1, 1)
    probs = torch.exp(dist.log_prob(support))
    my_entropy = calculate_entropy(probs).detach().cpu().numpy()
    scipy_entropy = entropy(probs.detach().cpu().numpy())

    assert np.allclose(my_entropy, scipy_entropy)
