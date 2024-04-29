import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
from tueplots import figsizes
plt.rcParams['figure.dpi']= 300
import torch
import numpy as np

from laplace.curvature.asdl import AsdlGGN
from laplace import KronLaplace
from tueplots import bundles
from datetime import datetime

from hetreg.uci_datasets import Skafte
from hetreg.utils import TensorDataLoader
from hetreg.models import MLP
from hetreg.marglik import marglik_optimization
from Baseline.evaluate_utils import get_RMAE, get_RMSE, save_variable_to_file


torch.manual_seed(711)
device = 'cpu'
n_samples = 1000
lr = 1e-2
lr_min = 1e-5
lr_hyp = 1e-1
lr_hyp_min = 1e-1
marglik_early_stopping = True
n_epochs = 10000 # 10000
n_hypersteps = 50
marglik_frequency = 50
laplace = KronLaplace 
optimizer = 'Adam'
backend = AsdlGGN
n_epochs_burnin = 100
prior_prec_init = 1e-3
use_wandb = False

ds_train = Skafte(n_samples=n_samples, double=True)
train_loader = TensorDataLoader(ds_train.data.to(device), ds_train.targets.to(device), batch_size=-1)
xl, xr = ds_train.x_bounds
offset = 3
# x = torch.linspace(xl-offset, xr+offset, 1000).to(device).double().unsqueeze(-1)


# Sales data:
sales_data = np.load("./Baseline/sales.npz")
X_train = sales_data['X_train']
y_train = sales_data['y_train']
train_loader = TensorDataLoader(torch.tensor(X_train.astype('float64')), torch.tensor(y_train.astype('float64')).view(-1, 1), batch_size=-1)



# print(train_loader)



# Homoscedastic
#  input_size, width, depth
# model = MLP(input_size=42, width=128, depth=1, output_size=1, activation='tanh', head=None).to(device).double()

# la, model, margliks, _, _ = marglik_optimization(
#     model, train_loader, likelihood='regression', lr=lr, lr_min=lr_min, lr_hyp=lr_hyp, early_stopping=marglik_early_stopping,
#     lr_hyp_min=lr_hyp_min, n_epochs=n_epochs, n_hypersteps=n_hypersteps, marglik_frequency=marglik_frequency,
#     laplace=laplace, prior_structure='layerwise', backend=backend, n_epochs_burnin=n_epochs_burnin,
#     scheduler='cos', optimizer=optimizer, prior_prec_init=prior_prec_init, use_wandb=use_wandb
# )
# f_mu, f_var = la(x)
# f_mu, f_var = f_mu.squeeze(), f_var.squeeze()
# m_map, s_map = f_mu.numpy(), 2 * torch.ones_like(f_mu).numpy() * la.sigma_noise.item()
# m_bayes, s_bayes = f_mu.numpy(), 2 * np.sqrt(f_var.numpy() + la.sigma_noise.item()**2)
# s_emp = 2 * np.sqrt(f_var.numpy())

# Heteroscedastic
model = MLP(input_size=42, width=128, depth=1, output_size=2, activation='relu', head=None).to(device).double()

# model = MLP(input_size=1, width=128, depth=1, output_size=2, activation='tanh', head=None).to(device).double()
la, model, margliksh, _, _ = marglik_optimization(
    model, train_loader, likelihood='heteroscedastic_regression', lr=lr, lr_min=lr_min, lr_hyp=lr_hyp, early_stopping=marglik_early_stopping,
    lr_hyp_min=lr_hyp_min, n_epochs=n_epochs, n_hypersteps=n_hypersteps, marglik_frequency=marglik_frequency,
    laplace=laplace, prior_structure='layerwise', backend=backend, n_epochs_burnin=n_epochs_burnin,
    scheduler='cos', optimizer=optimizer, prior_prec_init=prior_prec_init, use_wandb=use_wandb
)
# plt.plot(margliks, label='homoscedastic')
plt.plot(margliksh, label='heteroscedastic')
plt.ylabel('log marginal likelihood')
plt.legend()
plt.show()



X_test = sales_data['X_test']
y_test = sales_data['y_test']

f_mu, f_var, y_var = la(torch.tensor(X_test.astype('float64')))

f_mu, f_var, y_var = f_mu.squeeze(), f_var.squeeze(), y_var.squeeze()
mh_map, sh_map = f_mu.numpy(), 2 * np.sqrt(y_var.numpy())
mh_bayes, sh_bayes = f_mu.numpy(), 2 * np.sqrt(f_var.numpy() + y_var.numpy())
sh_emp = 2 * np.sqrt(f_var.numpy())



# groundtruth_y
f = y_test

std = np.std(y_test)

s_var = f - std
abs_s_var = np.abs(s_var)
s_var_sqr = s_var ** 2


# f, s = ds_train.ground_truth(x)
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

save_path = "./Baseline/save_output/" + current_time + "/"


data_save = {}

data_save['RMSE_mean'] = get_RMSE(f, f_mu)
data_save['RMSE_var_f'] = get_RMSE(abs_s_var, f_var)
data_save['RMSE_var_y'] = get_RMSE(abs_s_var, y_var)

data_save['RMAE_mean'] = get_RMAE(f, f_mu)
data_save['RMAE_var_f'] = get_RMAE(abs_s_var, f_var)
data_save['RMAE_var_y'] = get_RMAE(abs_s_var, y_var)

data_save['RMSE_var_m_map'] = np.mean(mh_map)
data_save['RMSE_var_s_map'] = np.mean(sh_map)


save_variable_to_file(variable=data_save, path=save_path)


# # here we define the groundtruth x
# f, s = ds_train.ground_truth(x)


# with plt.rc_context(bundles.jmlr2001(ncols=4, rel_width=1.777)):
#     size = figsizes.jmlr2001(ncols=4, rel_width=1.2, height_to_width_ratio=0.8)['figure.figsize']
#     fig, axs = plt.subplots(ncols=4, nrows=1, sharex=True, sharey=True, constrained_layout=True,
#                             figsize=size)
#     for ax in axs:
#         ax.plot(x, f, color='darkgrey', linestyle='--', label='ground truth')
#         ax.plot(x, f-2*s, color='darkgrey', linestyle='--')
#         ax.plot(x, f+2*s, color='darkgrey', linestyle='--')
#         ax.scatter(ds_train.data.numpy().flatten(), ds_train.targets.numpy().flatten(), alpha=0.7, color='darkgrey',
#                    facecolor='none', edgecolor='darkgrey', linewidth=0.5, s=10)
#         ax.set_ylim(-4, 4)
#         ax.set_xlim(-6, 6)
#         ax.set_xticks([-5, 0, 5])
#         ax.set_xlabel('$x$')
#     axs[0].set_ylabel('$y$')

#     # MAP homoscedastic
#     axs[0].set_title('Homoscedastic MAP')
#     # axs[0].plot(x, m_map, color='black')
#     # axs[0].fill_between(x.squeeze(), m_map - s_map, m_map + s_map, color='tab:orange', alpha=0.3)
#     # Bayes homoscedastic
#     # axs[1].plot(x, m_bayes, color='black')
#     axs[1].set_title('Homoscedastic Laplace')
#     # axs[1].fill_between(x.squeeze(), m_bayes - s_emp, m_bayes + s_emp, color='tab:blue', alpha=0.3)
#     # axs[1].fill_between(x.squeeze(), m_bayes - s_emp, m_bayes - s_bayes, color='tab:orange', alpha=0.3)
#     # axs[1].fill_between(x.squeeze(), m_bayes + s_emp, m_bayes + s_bayes, color='tab:orange', alpha=0.3)
#     # MAP heteroscedastic
#     axs[2].plot(x, mh_map, color='black')
#     axs[2].set_title('Heteroscedastic MAP')
#     axs[2].fill_between(x.squeeze(), mh_map - sh_map, mh_map + sh_map, color='tab:orange', alpha=0.3)
#     # Bayes heteroscedastic
#     axs[3].plot(x, mh_bayes, color='black', label='mean')
#     axs[3].set_title('Heteroscedastic Laplace')
#     axs[3].fill_between(x.squeeze(), mh_map - sh_emp, mh_map + sh_emp, color='tab:blue', alpha=0.3, label='epistemic')
#     axs[3].fill_between(x.squeeze(), mh_bayes - sh_emp, mh_bayes - sh_bayes, color='tab:orange', alpha=0.3, label='aleatoric')
#     axs[3].fill_between(x.squeeze(), mh_bayes + sh_emp, mh_bayes + sh_bayes, color='tab:orange', alpha=0.3)
#     handles, labels = ax.get_legend_handles_labels()
#     fig.legend(handles, labels, bbox_to_anchor =(0.5, -0.2), loc='lower center', ncol=4)
#     plt.savefig('figures/illustration.pdf')
#     plt.show()
