import torch
from torch import nn


class Backbone(nn.Module):
    """Base class to ensure a common interface for all backbones.

    Attributes:
        output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
    """

    def __init__(self, output_dim: int = 64):
        super(Backbone, self).__init__()
        self.output_dim = output_dim


class ScalarMLP(Backbone):
    """An MLP feature extractor for scalar input data.

    Attributes:
        output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
    """

    def __init__(self, output_dim: int = 64):
        super(ScalarMLP, self).__init__(output_dim=output_dim)

        self.layers = nn.Sequential(
            nn.Linear(1, 128),
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


class MNISTCNN(Backbone):
    """A CNN feature extractor for the MNIST dataset (1x28x28 image tensors).

    Attributes:
        output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
    """

    def __init__(self, output_dim: int = 64):
        super(MNISTCNN, self).__init__(output_dim=output_dim)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
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


class CNN(Backbone):
    """A CNN feature extractor for 3x128x128 image tensors.

    Attributes:
        output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
    """

    def __init__(self, output_dim: int = 64):
        super(CNN, self).__init__(output_dim=output_dim)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(14 * 14 * 128, output_dim * 2)
        self.fc2 = nn.Linear(output_dim * 2, output_dim)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.max_pool(self.relu(self.conv1(x)))
        x = self.max_pool(self.relu(self.conv2(x)))
        x = self.max_pool(self.relu(self.conv3(x)))
        x = self.dropout(self.flat(x))
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.relu(self.fc2(x))
        return x
