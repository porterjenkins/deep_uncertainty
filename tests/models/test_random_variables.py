import math

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from deep_uncertainty.models.random_variables import DiscreteConflation
from deep_uncertainty.models.random_variables import DiscreteRandomVariable
from deep_uncertainty.models.random_variables import DiscreteTruncatedNormal
from deep_uncertainty.models.random_variables import DoublePoisson
from deep_uncertainty.models.random_variables import GammaCount


@pytest.fixture
def upper_bound() -> int:
    return 2000


@pytest.fixture
def discrete_support(upper_bound: int) -> np.ndarray:
    return np.arange(upper_bound)


@pytest.fixture
def univariate_discrete_truncated_normal(upper_bound: int) -> DiscreteTruncatedNormal:
    return DiscreteTruncatedNormal(lower_bound=0, upper_bound=upper_bound, mu=3, sigma=0.5)


@pytest.fixture
def univariate_gamma_count() -> GammaCount:
    return GammaCount(alpha=1, beta=1)


@pytest.fixture
def univariate_double_poisson() -> DoublePoisson:
    return DoublePoisson(mu=3, phi=2)


@pytest.fixture
def univariate_discrete_conflation(
    univariate_gamma_count, univariate_double_poisson, discrete_support
) -> DiscreteConflation:
    return DiscreteConflation(
        [univariate_gamma_count, univariate_double_poisson], discrete_support
    )


@pytest.fixture
def multivariate_discrete_truncated_normal(upper_bound: int) -> DiscreteTruncatedNormal:
    return DiscreteTruncatedNormal(
        lower_bound=0, upper_bound=upper_bound, mu=np.array([3, 3]), sigma=np.array([1, 0.5])
    )


@pytest.fixture
def multivariate_gamma_count() -> GammaCount:
    return GammaCount(alpha=np.array([1, 1]), beta=np.array([1, 2]))


@pytest.fixture
def multivariate_double_poisson() -> DoublePoisson:
    return DoublePoisson(mu=np.array([3, 3]), phi=np.array([2, 10]))


@pytest.fixture
def multivariate_discrete_conflation(
    multivariate_gamma_count, multivariate_double_poisson, discrete_support
) -> DiscreteConflation:
    return DiscreteConflation(
        [multivariate_gamma_count, multivariate_double_poisson], discrete_support
    )


@pytest.mark.parametrize(
    argnames="rv",
    argvalues=[
        lazy_fixture("univariate_discrete_truncated_normal"),
        lazy_fixture("univariate_gamma_count"),
        lazy_fixture("univariate_double_poisson"),
        lazy_fixture("univariate_discrete_conflation"),
    ],
)
def test_univariate_discrete_rv_pmf_sums_to_one(
    rv: DiscreteRandomVariable, discrete_support: np.ndarray
):
    assert math.isclose(rv.pmf(discrete_support).sum(), 1, abs_tol=2e-3)


@pytest.mark.parametrize(
    argnames="rv",
    argvalues=[
        lazy_fixture("multivariate_discrete_truncated_normal"),
        lazy_fixture("multivariate_gamma_count"),
        lazy_fixture("multivariate_double_poisson"),
        # lazy_fixture("multivariate_discrete_conflation"), TODO: Not working right now.
    ],
)
def test_multivariate_discrete_rv_pmf_sums_to_one(
    rv: DiscreteRandomVariable, discrete_support: np.ndarray
):
    masses = rv.pmf(discrete_support.reshape(-1, 1))
    assert np.allclose(masses.sum(axis=0), np.ones(masses.shape[1]), atol=3e-3)


@pytest.mark.parametrize(
    argnames="rv",
    argvalues=[
        # lazy_fixture("univariate_discrete_truncated_normal"), TODO: Not working right now.
        # lazy_fixture("univariate_gamma_count"),
        lazy_fixture("univariate_double_poisson"),
        lazy_fixture("univariate_discrete_conflation"),
    ],
)
def test_univariate_discrete_rv_ppf_matches_expectations(
    rv: DiscreteRandomVariable, discrete_support: np.ndarray
):
    quantile = rv.ppf(0.5)
    assert isinstance(quantile, int)
    assert rv.pmf(discrete_support[discrete_support <= quantile]).sum() <= 0.5
    assert rv.pmf(discrete_support[discrete_support <= (quantile + 1)]).sum() > 0.5


@pytest.mark.parametrize(
    argnames="rv",
    argvalues=[
        # lazy_fixture("multivariate_discrete_truncated_normal"), TODO: Not working right now.
        # lazy_fixture("multivariate_gamma_count"), TODO: Not working right now.
        lazy_fixture("multivariate_double_poisson"),
        # lazy_fixture("multivariate_discrete_conflation"), TODO: Not working right now.
    ],
)
def test_multivariate_discrete_rv_ppf_matches_expectations(
    rv: DiscreteRandomVariable, discrete_support: np.ndarray
):
    quantiles = rv.ppf(0.5)
    assert isinstance(quantiles, np.ndarray)
    for j, q in enumerate(quantiles):
        assert rv.pmf(discrete_support[discrete_support <= q].reshape(-1, 1))[j].sum() <= 0.5
        assert rv.pmf(discrete_support[discrete_support <= (q + 1)].reshape(-1, 1)).sum() > 0.5
