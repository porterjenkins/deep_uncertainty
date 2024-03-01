import numpy as np
import pyro
import torch
from pyro.infer import SVI
from pyro.infer import Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from tqdm.auto import tqdm

from deep_uncertainty.models.old.regressors import OldPyroGaussianDNN

TRN_MIN = -5
TRN_MAX = 5
TEST_MIN = -6
TEST_MAX = 6

N_TRN = 2000
N_TEST = 1000


def generate_data(num_points=N_TRN, x_min=TRN_MIN, x_max=TRN_MAX):
    np.random.seed(42)  # Set a seed for reproducibility

    x_values = np.random.uniform(x_min, x_max, num_points)
    mu_values = np.sin(x_values)
    sigma_values = 0.15 * (1 + np.exp(-x_values)) ** -1
    y_values = np.random.normal(mu_values, sigma_values**2)

    return x_values, y_values


x, y = generate_data(num_points=10000)
x_train = torch.from_numpy(x).float()
y_train = torch.from_numpy(y).float()

# Check for CUDA availability and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = OldPyroGaussianDNN().to(device)
guide = AutoDiagonalNormal(model)
adam = pyro.optim.Adam({"lr": 1e-3})
svi = SVI(model, guide, adam, loss=Trace_ELBO())
pyro.clear_param_store()

x_train.to(device)
y_train.to(device)

num_epochs = 20000
progress_bar = tqdm(range(num_epochs), desc="Training Gaussian DNN", unit="epoch")
for epoch in progress_bar:
    loss = svi.step(x_train, y_train)

    progress_bar.set_postfix({"Train Loss": f"{loss / x_train.shape[0]:.4f}"})
