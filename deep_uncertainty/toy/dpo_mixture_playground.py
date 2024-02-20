"""A Panel dashboard to play around with the dynamics of the proposed Mixture of Double Poissons technique. Launch via `panel dpo_mixture_playground.py --show`."""
import numpy as np
import panel as pn
from matplotlib import pyplot as plt

from deep_uncertainty.models.random_variables import DoublePoisson


def get_mixture_plot(
    mu_0,
    mu_1,
    mu_2,
    phi_0,
    phi_1,
    phi_2,
    mse_0,
    mse_1,
    mse_2,
):
    domain = np.arange(10)

    X_0 = DoublePoisson(mu_0, phi_0)
    X_1 = DoublePoisson(mu_1, phi_1)
    X_2 = DoublePoisson(mu_2, phi_2)
    mse_vals = np.array([mse_0, mse_1, mse_2])
    weights = 1 / mse_vals
    weights = weights / weights.sum()

    fig, ax = plt.subplots(1, 1)

    ax.plot(domain, X_0.pmf(domain), "r.-", alpha=0.2, label=f"$X_0$ (MSE = ${mse_0:.3f}$)")
    ax.plot(domain, X_1.pmf(domain), "b.-", alpha=0.2, label=f"$X_1$ (MSE = ${mse_1:.3f}$)")
    ax.plot(domain, X_2.pmf(domain), "g.-", alpha=0.2, label=f"$X_2$ (MSE = ${mse_2:.3f}$)")

    stacked_densities = np.row_stack([X_0.pmf(domain), X_1.pmf(domain), X_2.pmf(domain)])
    ax.plot(domain, np.dot(weights, stacked_densities), "k.-", label="Mixture")

    expected_value = np.dot(weights, np.array([X_0.mu, X_1.mu, X_2.mu]))
    ax.scatter(
        expected_value,
        np.dot(
            weights,
            np.array([X_0.pmf(expected_value), X_1.pmf(expected_value), X_2.pmf(expected_value)]),
        ),
        marker="*",
        s=100,
        c="black",
        zorder=10,
        label="Expected Value",
    )
    ax.legend()
    return fig


mu_0 = pn.widgets.FloatSlider(start=0, end=7, value=2, step=0.1, name="mu")
mu_1 = pn.widgets.FloatSlider(start=0, end=7, value=3, step=0.1, name="mu")
mu_2 = pn.widgets.FloatSlider(start=0, end=7, value=7, step=0.1, name="mu")

phi_0 = pn.widgets.FloatSlider(start=1, end=10, value=4, step=0.1, name="phi")
phi_1 = pn.widgets.FloatSlider(start=1, end=10, value=4, step=0.1, name="phi")
phi_2 = pn.widgets.FloatSlider(start=1, end=10, value=8, step=0.1, name="phi")

mse_0 = pn.widgets.FloatSlider(start=0.001, end=3, value=0.5, step=0.001, name="mse")
mse_1 = pn.widgets.FloatSlider(start=0.001, end=3, value=0.5, step=0.001, name="mse")
mse_2 = pn.widgets.FloatSlider(start=0.001, end=3, value=0.5, step=0.001, name="mse")

mixture_plot = pn.bind(
    get_mixture_plot,
    mu_0,
    mu_1,
    mu_2,
    phi_0,
    phi_1,
    phi_2,
    mse_0,
    mse_1,
    mse_2,
)

plot_pane = pn.pane.Matplotlib(mixture_plot)
widgets = pn.Column(
    pn.pane.Markdown("## Parameters"),
    pn.Row(pn.pane.Markdown("### X_0"), mu_0, phi_0, mse_0),
    pn.Row(pn.pane.Markdown("### X_1"), mu_1, phi_1, mse_1),
    pn.Row(pn.pane.Markdown("### X_2"), mu_2, phi_2, mse_2),
)

app = pn.Column(widgets, plot_pane)
app.servable()
