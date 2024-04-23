import numpy as np
import torch
from torch.distributions import Distribution

from deep_uncertainty.random_variables.discrete_random_variable import DiscreteRandomVariable


class DiscreteConflation(DiscreteRandomVariable):
    """A conflation of discrete random variables, as described in https://www.researchgate.net/publication/1747728_Conflations_of_Probability_Distributions.

    Given discrete random variables X_1, X_2, ..., X_n, with pmfs g_{X_1}, g_{X_2}, ..., g_{X_n}, the conflation Q = X_1 & X_2 & ... & X_n has pmf given by

        g_Q(y) = (1 / Z) * g_{X_1}(y) * g_{X_2}(y) * ... * g_{X_n}(y)

    where Z is a normalizing constant so the masses sum to 1. This is a proper random variable with certain desirable properties for combining distributions.

    Args:
        random_variables (list[Distribution | DiscreteRandomVariable]): A list of the discrete random variables to conflate (i.e. X_1, X_2, ..., X_n)
    """

    def __init__(
        self,
        random_variables: list[Distribution | DiscreteRandomVariable],
    ) -> None:
        if isinstance(random_variables[0], Distribution):
            dimension = random_variables[0].mean.numel()
            use_torch = True
            device = random_variables[0].mean.device
        else:
            dimension = random_variables[0].dimension
            use_torch = random_variables[0].use_torch
            device = random_variables[0].device
        super().__init__(dimension=dimension, use_torch=use_torch, device=device)
        self.random_variables = random_variables

    def _pmf(self, x: int | np.ndarray | torch.Tensor) -> float | np.ndarray | torch.Tensor:
        pmf_vals = []
        for rv in self.random_variables:
            if isinstance(rv, DiscreteRandomVariable):
                pmf_vals.append(rv.pmf(x))
            else:
                pmf_vals.append(torch.exp(rv.log_prob(x)))
        if self.use_torch:
            return torch.cat(pmf_vals, dim=1).prod(dim=1, keepdim=True)
        else:
            return np.concatenate(pmf_vals, axis=1).prod(axis=1, keepdims=True)


if __name__ == "__main__":
    from torch.distributions import Poisson

    rv_list = [Poisson(rate=4), Poisson(rate=7)]
    conflation = DiscreteConflation(rv_list)
    support = torch.arange(15)
    conflation.pmf(support)
