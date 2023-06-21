import torch

from evaluation.metrics import get_mse

def evaluate_model_mse(val_loader, model, device):
    model.eval()
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            pred = model(inputs.to(device))
            if isinstance(pred, tuple):
                mean, logvar = pred
            else:
                mean = pred

            preds_all.append(mean.cpu())
            targets_all.append(targets)

    preds_all = torch.cat(preds_all)
    targets_all = torch.cat(targets_all)

    mse = get_mse(y_true=targets_all, y_hat=preds_all)

    return mse

def evaluate_model_criterion(val_loader, model, criterion, device):
    model.eval()
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            pred = model(inputs.to(device))
            if isinstance(pred, tuple):
                mean, logvar = pred
            else:
                mean = pred

            preds_all.append(mean.cpu())
            targets_all.append(targets)

    preds_all = torch.cat(preds_all)
    targets_all = torch.cat(targets_all)

    loss = criterion(preds_all, targets_all)

    return loss


