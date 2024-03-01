import numpy as np
from matplotlib import pyplot as plt


def plot_posterior_predictive(
    x_test: np.ndarray,
    y_test: np.ndarray,
    preds: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    c: str = "r",
    alpha: float = 0.2,
    show: bool = True,
    title: str = "",
    ax: plt.Axes | None = None,
):
    order = x_test.argsort()

    ax = plt.subplots(1, 1, figsize=(10, 6))[1] if ax is None else ax

    ax.scatter(x_test[order], y_test[order], alpha=0.1, label="Test Data")
    ax.plot(x_test[order], preds[order])
    ax.fill_between(
        x_test[order], lower[order], upper[order], color=c, alpha=alpha, label="95% CI"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title(title)
    ax.set_ylim(y_test.min() - 5, y_test.max() + 5)
    if show:
        plt.show()
