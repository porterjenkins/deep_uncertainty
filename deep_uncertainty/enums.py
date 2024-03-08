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
    NEGATIVE_BINOMIAL = "negative_binomial"


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


class ImageDatasetName(Enum):
    MNIST = "mnist"
    COIN_COUNTING = "coin_counting"
