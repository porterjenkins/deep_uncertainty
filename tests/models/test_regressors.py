import torch

from deep_uncertainty.models.old_regressors import RegressionNN


def test_regression_nn_handles_log_output_correctly():
    output_dim = 2
    log_dims = [False, True]
    model = RegressionNN(output_dim=output_dim, log_dims=log_dims)
    x = torch.rand(2, 1)

    with torch.no_grad():
        model.train()
        train_output = model(x)
        model.eval()
        eval_output = model(x)

    for j in range(output_dim):
        dim_in_log_space_while_training = log_dims[j]
        if dim_in_log_space_while_training:
            assert torch.equal(torch.exp(train_output[:, j]), eval_output[:, j])
