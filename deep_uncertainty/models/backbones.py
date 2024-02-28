import torch
from torch import nn


class MLPBackbone(nn.Module):
    """An MLP feature extractor.

    Attributes:
        input_dim (int, optional): Dimension of input data. Defaults to 1 (scalars).
        output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
    """

    def __init__(self, input_dim: int = 1, output_dim: int = 64):
        super(MLPBackbone, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
