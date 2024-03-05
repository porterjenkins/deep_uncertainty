from enum import Enum


class AcceleratorType(Enum):
    cpu = "cpu"
    gpu = "gpu"
    mps = "mps"
    auto = "auto"


class HeadType(Enum):
    MEAN = "mean"
    GAUSSIAN = "gaussian"
    POISSON = "poisson"
    DOUBLE_POISSON = "double_poisson"


class BackboneType(Enum):
    MLP = "mlp"
    RESIDUAL_MLP = "residual_mlp"
    CNN = "cnn"
    RESIDUAL_CNN = "residual_cnn"
    TRANSFORMER = "transformer"


class OptimizerType(Enum):
    ADAM = "adam"
    SGD = "sgd"
    ADAM_W = "adam_w"


class LRSchedulerType(Enum):
    COSINE_ANNEALING = "cosine_annealing"


class BetaSchedulerType(Enum):
    COSINE_ANNEALING = "cosine_annealing"
    LINEAR = "linear"


class DatasetType(Enum):
    SCALAR = "scalar"
    TABULAR = "tabular"
    IMAGE = "image"


class DatasetName(Enum):
    ROTATED_MNIST = "rotated_mnist"
