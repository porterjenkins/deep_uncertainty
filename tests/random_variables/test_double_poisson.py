import numpy as np

from deep_uncertainty.random_variables import DoublePoisson


def test_double_poisson_cdf_vectorizes_as_expected():

    dp1 = DoublePoisson(mu=7, phi=4)
    dp2 = DoublePoisson(mu=3, phi=10)
    dp_vec = DoublePoisson(mu=[7, 3], phi=[4, 10])
    assert np.allclose(dp_vec.cdf(5), np.array([dp1.cdf(5), dp2.cdf(5)]))
    assert np.allclose(
        dp_vec.cdf([5, 1]),
        np.array([dp1.cdf(5), dp2.cdf(1)]),
    )
    assert np.allclose(
        dp_vec.cdf(np.array([[5, 1], [4, 3]])),
        np.array(
            [
                [dp1.cdf(5), dp2.cdf(1)],
                [dp1.cdf(4), dp2.cdf(3)],
            ]
        ),
    )
