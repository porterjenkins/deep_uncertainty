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
from models.model_utils import get_1d_plot
from toy.gen_data import generate_gaussian_data

def train_regression_nn(train_loader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)

# Function for evaluation
def evaluate_regression_nn(test_loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(test_loader.dataset)

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

    # Instantiate and train the network
    model = RegressionNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the network
    num_epochs = 500
    progress_bar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    trn_losses = []
    val_losses = []
    for epoch in progress_bar:
        train_loss = train_regression_nn(train_loader, model, criterion, optimizer, device)
        val_loss = evaluate_regression_nn(val_loader, model, criterion, device)

        progress_bar.set_postfix({"Train Loss": f"{train_loss:.4f}", "Val Loss": f"{val_loss:.4f}"})
        trn_losses.append(train_loss)
        val_losses.append(val_loss)



    test_loss = evaluate_regression_nn(test_loader, model, criterion, device)
    print("Test MSE: {:.4f}".format(test_loss))

    plt.plot(np.arange(num_epochs), trn_losses, label="TRAIN")
    plt.plot(np.arange(num_epochs), val_losses, label="VAL")
    plt.legend()
    plt.show()


    get_1d_plot(X_test, y_test, model)












if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trn-cfg', type=str, default="./toy_exp_train_config.yaml", help='Path to data config file')
    args = parser.parse_args()

    cfg = get_yaml(args.trn_cfg)

    main(config=cfg)
