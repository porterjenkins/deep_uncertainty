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


class CNNBackbone(nn.Module):
    """A CNN feature extractor.

    Attributes:
        in_channels (int, optional): Number of assumed channels in input tensor (N, **C**, ...). Defaults to 1.
        output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
    """

    def __init__(self, in_channels: int = 1, output_dim: int = 64):
        super(CNNBackbone, self).__init__()

        self.in_channels = 1
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(5 * 5 * 64, output_dim * 2)
        self.fc2 = nn.Linear(output_dim * 2, output_dim)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.max_pool(self.relu(self.conv1(x)))
        x = self.max_pool(self.relu(self.conv2(x)))
        x = self.dropout(self.flat(x))
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.relu(self.fc2(x))
        return x
