import numpy as np
import argparse
import os
import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt


from utils import get_yaml
from torch.utils.data import DataLoader, TensorDataset
from models.regressors import RegressionNN
from models.model_utils import get_mean_preds_and_targets, train_regression_nn, get_gaussian_bounds
from evaluation.plots import get_1d_mean_plot
from evaluation.evals import evaluate_model_mse
from evaluation.metrics import get_mse, get_calibration

NUM_MEMBERS = 5

def main(config: dict):

    X_train = np.loadtxt(
        fname=os.path.join(config['dataset']["dir"], config['dataset']["name"] + "_x_train.txt")
    )
    X_val = np.loadtxt(
        fname=os.path.join(config['dataset']["dir"], config['dataset']["name"] + "_x_val.txt")
    )
    X_test = np.loadtxt(
        fname=os.path.join(config['dataset']["dir"], config['dataset']["name"] + "_x_test.txt")
    )
    y_train = np.loadtxt(
        fname=os.path.join(config['dataset']["dir"], config['dataset']["name"] + "_y_train.txt")
    )
    y_val = np.loadtxt(
        fname=os.path.join(config['dataset']["dir"], config['dataset']["name"] + "_y_val.txt")
    )
    y_test = np.loadtxt(
        fname=os.path.join(config['dataset']["dir"], config['dataset']["name"] + "_y_test.txt")
    )


    train_dataset = TensorDataset(
        torch.Tensor(X_train.reshape(-1, 1)),
        torch.Tensor(y_train.reshape(-1, 1))
    )
    val_dataset = TensorDataset(
        torch.Tensor(X_val.reshape(-1, 1)),
        torch.Tensor(y_val.reshape(-1, 1))
    )

    test_dataset = TensorDataset(
        torch.Tensor(X_test.reshape(-1, 1)),
        torch.Tensor(y_test.reshape(-1, 1))
    )


    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Check for CUDA availability and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()

    ensemble_preds = np.zeros((len(test_loader), NUM_MEMBERS))

    for i in range(NUM_MEMBERS):
        # Instantiate and train the network
        model_i = RegressionNN().to(device)

        optimizer = optim.Adam(model_i.parameters(), lr=3e-4)

        # Train the network
        num_epochs = config['optim']['epochs']
        progress_bar = tqdm(range(num_epochs), desc="Training", unit="epoch")
        trn_losses = []
        val_losses = []
        for epoch in progress_bar:
            train_loss = train_regression_nn(train_loader, model_i, criterion, optimizer, device)
            val_loss = evaluate_model_mse(val_loader, model_i, device)

            progress_bar.set_postfix({"Train Loss": f"{train_loss:.4f}", "Val Loss": f"{val_loss:.4f}"})
            trn_losses.append(train_loss)
            val_losses.append(val_loss)


        test_preds, test_targets = get_mean_preds_and_targets(test_loader, model_i, device)
        ensemble_preds[:, i] = test_preds.data.numpy().flatten()

    preds = ensemble_preds.mean(1)
    sigma = ensemble_preds.std(1)

    test_mse = get_mse(test_targets, preds)
    print("Test MSE: {:.4f}".format(test_mse))
    upper, lower = get_gaussian_bounds(test_preds, sigma)
    test_calib = get_calibration(test_targets, upper, lower)
    print("Test Calib: {:.4f}".format(test_calib))













if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trn-cfg', type=str, default="./toy_exp_train_config.yaml", help='Path to data config file')
    args = parser.parse_args()

    cfg = get_yaml(args.trn_cfg)

    main(config=cfg)
