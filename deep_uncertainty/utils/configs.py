from __future__ import annotations

from pathlib import Path

import yaml

from deep_uncertainty.enums import AcceleratorType
from deep_uncertainty.enums import BetaSchedulerType
from deep_uncertainty.enums import DatasetType
from deep_uncertainty.enums import HeadType
from deep_uncertainty.enums import ImageDatasetName
from deep_uncertainty.enums import LRSchedulerType
from deep_uncertainty.enums import OptimizerType
from deep_uncertainty.enums import TextDatasetName
from deep_uncertainty.utils.generic_utils import get_yaml
from deep_uncertainty.utils.generic_utils import to_snake_case


class TrainingConfig:
    """Class with configuration options for training a model.

    Attributes:
        experiment_name (str): The name of the training run (used for identifying chkp weights / eval logs), automatically cast to snake case.
        head_type (HeadType): The output head to use in the neural network, e.g. "gaussian", "mean", "poisson", etc.
        chkp_dir (Path): Directory to checkpoint model weights in.
        chkp_freq (int): Number of epochs to wait in between checkpointing model weights.
        batch_size (int): The batch size to train with.
        accumulate_grad_batches (int): How many batches to accumulate gradients for before updating.
        num_epochs (int): The number of epochs through the data to complete during training.
        optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
        optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
        lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing".
        lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}.
        beta_scheduler_type (BetaSchedulerType | None): If specified, the type of beta scheduler to use for training loss (if applicable).
        beta_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen beta scheduler, e.g. {"beta_0": 1.0, "beta_1": 0.5}.
        dataset_type (DatasetType): Type of dataset to use in this experiment (tabular or image).
        dataset_spec (Path | ImageDataset): If dataset is tabular, path to the dataset .npz file to use. Otherwise, name of the image dataset to load.
        num_trials (int): Number of trials to run for this experiment.
        log_dir (Path): Directory to log results to.
        source_dict (dict): Dictionary from which config was constructed.
        precision (str | None, optional): String specifying desired floating point precision for training. Defaults to None.
        input_dim (int | None, optional): If dataset is tabular, the input dim of the data (used to construct the MLP). Defaults to None.
        hidden_dim (int, optional): Feature dimension used in the model (before feeding the representation to the output head). Defaults to 64.
        random_seed (int | None, optional): If specified, the random seed to use for reproducibility. Defaults to None.
    """

    def __init__(
        self,
        experiment_name: str,
        accelerator_type: AcceleratorType,
        head_type: HeadType,
        chkp_dir: Path,
        chkp_freq: int,
        batch_size: int,
        accumulate_grad_batches: int,
        num_epochs: int,
        optim_type: OptimizerType,
        optim_kwargs: dict,
        lr_scheduler_type: LRSchedulerType | None,
        lr_scheduler_kwargs: dict | None,
        beta_scheduler_type: BetaSchedulerType | None,
        beta_scheduler_kwargs: dict | None,
        dataset_type: DatasetType,
        dataset_spec: Path | ImageDatasetName,
        num_trials: int,
        log_dir: Path,
        source_dict: dict,
        input_dim: int | None = None,
        hidden_dim: int = 64,
        precision: str | None = None,
        random_seed: int | None = None,
    ):
        self.experiment_name = experiment_name
        self.accelerator_type = accelerator_type
        self.head_type = head_type
        self.chkp_dir = chkp_dir
        self.chkp_freq = chkp_freq
        self.batch_size = batch_size
        self.accumulate_grad_batches = accumulate_grad_batches
        self.num_epochs = num_epochs
        self.optim_type = optim_type
        self.optim_kwargs = optim_kwargs
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.beta_scheduler_type = beta_scheduler_type
        self.beta_scheduler_kwargs = beta_scheduler_kwargs
        self.dataset_type = dataset_type
        self.dataset_spec = dataset_spec
        self.num_trials = num_trials
        self.log_dir = log_dir
        self.source_dict = source_dict
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.precision = precision
        self.random_seed = random_seed

    @staticmethod
    def from_yaml(config_path: str | Path) -> TrainingConfig:
        """Factory method to construct an TrainingConfig from a .yaml file.

        Args:
            config_path (str | Path): Path to the .yaml file with config options.

        Returns:
            TrainingConfig: The specified config.
        """
        config_dict = get_yaml(config_path)
        training_dict: dict = config_dict["training"]
        eval_dict: dict = config_dict["evaluation"]

        experiment_name = to_snake_case(config_dict["experiment_name"])
        accelerator_type = AcceleratorType(training_dict["accelerator"])
        head_type = HeadType(config_dict["head_type"])
        chkp_dir = Path(training_dict["chkp_dir"])
        chkp_freq = training_dict["chkp_freq"]
        batch_size = training_dict["batch_size"]
        accumulate_grad_batches = training_dict.get("accumulate_grad_batches", 1)
        num_epochs = training_dict["num_epochs"]
        precision = training_dict.get("precision")
        optim_type = OptimizerType(training_dict["optimizer"]["type"])
        optim_kwargs = training_dict["optimizer"]["kwargs"]

        if "lr_scheduler" in training_dict:
            lr_scheduler_type = LRSchedulerType(training_dict["lr_scheduler"]["type"])
            lr_scheduler_kwargs = training_dict["lr_scheduler"]["kwargs"]
        else:
            lr_scheduler_type = None
            lr_scheduler_kwargs = None

        if "beta_scheduler" in training_dict:
            beta_scheduler_type = BetaSchedulerType(training_dict["beta_scheduler"]["type"])
            beta_scheduler_kwargs = training_dict["beta_scheduler"]["kwargs"]
            if beta_scheduler_kwargs.get("last_epoch", None) == -1:
                beta_scheduler_kwargs["last_epoch"] = num_epochs
        else:
            beta_scheduler_type = None
            beta_scheduler_kwargs = None

        dataset_type = DatasetType(config_dict["dataset"]["type"])
        if dataset_type == DatasetType.TABULAR:
            dataset_spec = Path(config_dict["dataset"]["spec"])
        elif dataset_type == DatasetType.IMAGE:
            dataset_spec = ImageDatasetName(config_dict["dataset"]["spec"])
        elif dataset_type == DatasetType.TEXT:
            dataset_spec = TextDatasetName(config_dict["dataset"]["spec"])

        num_trials = eval_dict["num_trials"]
        log_dir = Path(eval_dict["log_dir"])
        input_dim = config_dict["dataset"].get("input_dim")
        hidden_dim = config_dict.get("hidden_dim", 64)
        random_seed = config_dict.get("random_seed")

        return TrainingConfig(
            experiment_name=experiment_name,
            accelerator_type=accelerator_type,
            head_type=head_type,
            chkp_dir=chkp_dir,
            chkp_freq=chkp_freq,
            batch_size=batch_size,
            accumulate_grad_batches=accumulate_grad_batches,
            num_epochs=num_epochs,
            optim_type=optim_type,
            optim_kwargs=optim_kwargs,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            beta_scheduler_type=beta_scheduler_type,
            beta_scheduler_kwargs=beta_scheduler_kwargs,
            dataset_type=dataset_type,
            dataset_spec=dataset_spec,
            num_trials=num_trials,
            log_dir=log_dir,
            source_dict=config_dict,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            precision=precision,
            random_seed=random_seed,
        )

    def to_yaml(self, filepath: str | Path):
        """Save this config as a .yaml file at the given filepath.

        Args:
            filepath (str | Path): The filepath to save this config at.
        """
        with open(filepath, "w") as f:
            yaml.dump(self.source_dict, f)


class EnsembleConfig:
    """Class with configuration options for specifying / evaluating an ensemble.

    Attributes:
        experiment_name (str): The name of the experiment (used for identifying chkp weights / eval logs), automatically cast to snake case.
        accelerator_type (AcceleratorType): The type of hardware accelerator to use for model inference.
        member_head_type (HeadType): The HeadType shared by each member of the ensemble.
        members (list[Path]): List with checkpoint paths for each desired ensemble member.
        batch_size (int): The batch size to perform inference with.
        dataset_type (DatasetType): Type of dataset to use in this experiment (tabular or image).
        dataset_spec (Path | ImageDataset): If dataset is tabular, path to the dataset .npz file to use. Otherwise, name of the image dataset to load.
        log_dir (Path): Directory to log results to.
    """

    def __init__(
        self,
        experiment_name: str,
        accelerator_type: AcceleratorType,
        member_head_type: HeadType,
        members: list[Path],
        batch_size: int,
        dataset_type: DatasetType,
        dataset_spec: Path | ImageDatasetName,
        log_dir: Path,
        source_dict: dict,
    ):
        self.experiment_name = experiment_name
        self.accelerator_type = accelerator_type
        self.member_head_type = member_head_type
        self.members = members
        self.batch_size = batch_size
        self.dataset_type = dataset_type
        self.dataset_spec = dataset_spec
        self.log_dir = log_dir
        self.source_dict = source_dict

    @staticmethod
    def from_yaml(config_path: str | Path) -> EnsembleConfig:
        """Factory method to construct an EnsembleConfig from a .yaml file.

        Args:
            config_path (str | Path): Path to the .yaml file with config options.

        Returns:
            EnsembleConfig: The specified config.
        """
        config_dict = get_yaml(config_path)

        experiment_name = to_snake_case(config_dict["experiment_name"])
        accelerator_type = AcceleratorType(config_dict["accelerator"])
        member_head_type = HeadType(config_dict["member_head_type"])
        members = [Path(path) for path in config_dict["members"]]
        batch_size = config_dict["batch_size"]
        dataset_type = DatasetType(config_dict["dataset"]["type"])
        if dataset_type == DatasetType.TABULAR:
            dataset_spec = Path(config_dict["dataset"]["spec"])
        elif dataset_type == DatasetType.IMAGE:
            dataset_spec = ImageDatasetName(config_dict["dataset"]["spec"])
        elif dataset_type == DatasetType.TEXT:
            dataset_spec = TextDatasetName(config_dict["dataset"]["spec"])
        log_dir = Path(config_dict["log_dir"])

        return EnsembleConfig(
            experiment_name=experiment_name,
            accelerator_type=accelerator_type,
            member_head_type=member_head_type,
            members=members,
            batch_size=batch_size,
            dataset_type=dataset_type,
            dataset_spec=dataset_spec,
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
