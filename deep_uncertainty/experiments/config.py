from __future__ import annotations

from pathlib import Path

import yaml

from deep_uncertainty.experiments.enums import AcceleratorType
from deep_uncertainty.experiments.enums import BackboneType
from deep_uncertainty.experiments.enums import HeadType
from deep_uncertainty.experiments.enums import LRSchedulerType
from deep_uncertainty.experiments.enums import OptimizerType
from deep_uncertainty.utils.generic_utils import get_yaml
from deep_uncertainty.utils.generic_utils import to_snake_case


class ExperimentConfig:
    """Class with configuration options for an experiment.

    Attributes:
        experiment_name (str): The name of the experiment (used for identifying chkp weights / eval logs), automatically cast to snake case.
        backbone_type (BackboneType): The backbone type to use in the neural network, e.g. "mlp", "cnn", etc.
        head_type (HeadType): The output head to use in the neural network, e.g. "gaussian", "mean", "poisson", etc.
        head_kwargs (dict | None): Key-value argument specifications for the chosen regression head (if applicable).
        chkp_dir (Path): Directory to checkpoint model weights in.
        batch_size (int): The batch size to train with.
        num_epochs (int): The number of epochs through the data to complete during training.
        optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
        optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
        lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing".
        lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}.
        dataset_path (Path): Path to the dataset .npz file to use.
        num_trials (int): Number of trials to run for this experiment.
        log_dir (Path): Directory to log results to.
        source_dict (dict): Dictionary from which config was constructed.
    """

    def __init__(
        self,
        experiment_name: str,
        accelerator_type: AcceleratorType,
        backbone_type: BackboneType,
        head_type: HeadType,
        head_kwargs: dict | None,
        chkp_dir: Path,
        batch_size: int,
        num_epochs: int,
        optim_type: OptimizerType,
        optim_kwargs: dict,
        lr_scheduler_type: LRSchedulerType | None,
        lr_scheduler_kwargs: dict | None,
        dataset_path: Path,
        num_trials: int,
        log_dir: Path,
        source_dict: dict,
    ):
        self.experiment_name = experiment_name
        self.accelerator_type = accelerator_type
        self.backbone_type = backbone_type
        self.head_type = head_type
        self.head_kwargs = head_kwargs
        self.chkp_dir = chkp_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.optim_type = optim_type
        self.optim_kwargs = optim_kwargs
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.dataset_path = dataset_path
        self.num_trials = num_trials
        self.log_dir = log_dir
        self.source_dict = source_dict

    @staticmethod
    def from_yaml(config_path: str | Path) -> ExperimentConfig:
        """Factory method to construct an ExperimentConfig from a .yaml file.

        Args:
            config_path (str | Path): Path to the .yaml file with config options.

        Returns:
            ExperimentConfig: The specified config.
        """
        config_dict = get_yaml(config_path)
        architecture_dict = config_dict["architecture"]
        training_dict = config_dict["training"]
        eval_dict = config_dict["evaluation"]

        experiment_name = to_snake_case(config_dict["experiment_name"])
        accelerator_type = AcceleratorType(training_dict["accelerator"])
        backbone_type = BackboneType(architecture_dict["backbone_type"])
        head_type = HeadType(architecture_dict["head"]["type"])
        head_kwargs = architecture_dict["head"].get("kwargs", None)
        chkp_dir = Path(training_dict["chkp_dir"])
        batch_size = training_dict["batch_size"]
        num_epochs = training_dict["num_epochs"]
        optim_type = OptimizerType(training_dict["optimizer"]["type"])
        optim_kwargs = training_dict["optimizer"]["kwargs"]

        if "lr_scheduler" in training_dict:
            lr_scheduler_type = LRSchedulerType(training_dict["lr_scheduler"]["type"])
            lr_scheduler_kwargs = training_dict["lr_scheduler"]["kwargs"]
        else:
            lr_scheduler_type = None
            lr_scheduler_kwargs = None

        dataset_path = Path(config_dict["dataset"]["path"])
        num_trials = eval_dict["num_trials"]
        log_dir = Path(eval_dict["log_dir"])

        return ExperimentConfig(
            experiment_name=experiment_name,
            accelerator_type=accelerator_type,
            backbone_type=backbone_type,
            head_type=head_type,
            head_kwargs=head_kwargs,
            chkp_dir=chkp_dir,
            batch_size=batch_size,
            num_epochs=num_epochs,
            optim_type=optim_type,
            optim_kwargs=optim_kwargs,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            dataset_path=dataset_path,
            num_trials=num_trials,
            log_dir=log_dir,
            source_dict=config_dict,
        )

    def to_yaml(self, filepath: str | Path):
        """Save this config as a .yaml file at the given filepath.

        Args:
            filepath (str | Path): The filepath to save this config at.
        """
        with open(filepath, "w") as f:
            yaml.dump(self.source_dict, f)
