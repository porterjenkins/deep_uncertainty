import torch
from torch import nn
from torchvision.models import mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights
from transformers import BatchEncoding
from transformers import DistilBertConfig
from transformers import DistilBertModel


class Backbone(nn.Module):
    """Base class to ensure a common interface for all backbones.

    Attributes:
        output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
    """

    def __init__(self, output_dim: int = 64):
        super(Backbone, self).__init__()
        self.output_dim = output_dim


class MLP(Backbone):
    """An MLP feature extractor for (N, d) input data.

    Attributes:
        layers (nn.Sequential): The layers of this MLP.
    """

    def __init__(self, input_dim: int = 1, output_dim: int = 64):
        """Instantiate a ScalarMLP.

        Args:
            input_dim (int, optional): Dimension of input feature vectors. Defaults to 1.
            output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
        """
        self.input_dim = input_dim
        super(MLP, self).__init__(output_dim=output_dim)

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


class MNISTCNN(Backbone):
    """A CNN feature extractor for the MNIST dataset (1x28x28 image tensors).

    Attributes:
        output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
    """

    def __init__(self, output_dim: int = 64):
        super(MNISTCNN, self).__init__(output_dim=output_dim)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(5 * 5 * 64, output_dim * 2)
        self.fc2 = nn.Linear(output_dim * 2, output_dim)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.max_pool(self.relu(self.conv1(x))))
        x = self.bn2(self.max_pool(self.relu(self.conv2(x))))
        x = self.dropout(self.flat(x))
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.relu(self.fc2(x))
        return x


class SmallCNN(Backbone):
    """A small CNN feature extractor for 3x128x128 image tensors.

    Attributes:
        output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
    """

    def __init__(self, output_dim: int = 64):
        super(SmallCNN, self).__init__(output_dim=output_dim)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(128)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(14 * 14 * 128, output_dim * 2)
        self.fc2 = nn.Linear(output_dim * 2, output_dim)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.max_pool(self.relu(self.conv1(x))))
        x = self.bn2(self.max_pool(self.relu(self.conv2(x))))
        x = self.bn3(self.max_pool(self.relu(self.conv3(x))))
        x = self.dropout(self.flat(x))
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.relu(self.fc2(x))
        return x


class MobileNetV3(Backbone):
    """A MobileNetV3 feature extractor for 3x224x224 image tensors.

    Attributes:
        output_dim (int): Dimension of output feature vectors.
    """

    def __init__(self, output_dim: int = 64):
        """Initialize a MobileNetV3 feature extractor.

        Args:
            output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
        """
        super(MobileNetV3, self).__init__(output_dim=output_dim)

        self.backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).features
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten(start_dim=2)
        self.conv1d = nn.Conv1d(in_channels=960, out_channels=self.output_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.flatten(self.avg_pool(self.backbone(x)))
        h = self.conv1d(h).squeeze(-1)
        return h


class DistilBert(Backbone):
    """A DistilBert feature extractor for text sequences.

    Attributes:
        output_dim (int): Dimension of output feature vectors.
    """

    def __init__(self, output_dim: int = 64):
        """Initialize a DistilBert text feature extractor.

        Args:
            output_dim (int, optional): Dimension of output feature vectors. Defaults to 64.
        """
        super(DistilBert, self).__init__(output_dim=output_dim)

        config = DistilBertConfig(
            output_attentions=False, output_hidden_states=False, return_dict=False
        )
        self.backbone = DistilBertModel(config)
        self.projection_1 = nn.Linear(config.dim, config.dim // 2)
        self.projection_2 = nn.Linear(config.dim // 2, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: BatchEncoding) -> torch.Tensor:
        h: torch.Tensor = self.backbone(**x)[0][:, 0]
        h = self.relu(self.projection_1(h))
        h = self.relu(self.projection_2(h))
        return h
