# Predictive Uncertainty with Deep Learning and Count Data

## Setup

### Install Project Dependencies

```bash
conda create --name deep-uncertainty python=3.10
conda activate deep-uncertainty
pip install -r requirements.txt
```

### Install Pre-Commit Hook

To install this repo's pre-commit hook with automatic linting and code quality checks, simply execute the following command:

```bash
pre-commit install
```

When you commit new code, the pre-commit hook will run a series of scripts to standardize formatting. There will also be a flake8 check that provides warnings about various Python styling violations. These must be resolved for the commit to go through. If you need to bypass the linters for a specific commit, add the `--no-verify` flag to your git commit command.

### Downloading Data

Some of the datasets we are using to run experiments are in a non-standard format online. ETL code for this data has been pre-defined in the `etl` module and can be invoked from the command line.

For example, to get a `.npz` file for the `Bikes` dataset, run the following:

```bash
python deep_uncertainty/etl/get_bikes_data.py --output-dir path/to/your/data/dir
```

### Training models

To train a model, first fill out a config (using [this config](deep_uncertainty/training/sample_train_config.yaml) as a template). Then, from the terminal, run

```bash
python deep_uncertainty/training/train_model.py --config path/to/your/config.yaml
```

Logs / saved model weights will be found at the locations specified in your config.

#### Training on Tabular Datasets

If fitting a model on tabular data, the training script assumes the dataset will be stored locally in `.npz` files with `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, and `y_test` splits (these files are automatically produced by our [data generating code](deep_uncertainty/data_generator.py)). Pass a path to this `.npz` file in the `dataset` `spec` key in the config (also ensure that the `dataset` `type` is set to `tabular` and the `dataset` `input_dim` key is properly specified).

#### Training on Image Datasets

The currently-supported image datasets for training models are:

- `Vehicles` (VEDAI, labeled with the count of vehicle annotations)
- `COCO-People` (All images in COCO containing people, labeled with the count of "person" annotations)

To train a model on any of these datasets, simply specify `"image"` for the `dataset` `type` key in the config, then set `dataset` `spec` to the requisite dataset name (see the options in the `ImageDatasetName` class [here](deep_uncertainty/enums.py))

### Evaluating Models

#### Individual Models

To obtain evaluation metrics for a given model (and have them save to its log directory), use the following command:

```bash
python deep_uncertainty/evaluation/eval_model.py \
--log-dir path/to/training/log/dir \
--chkp-path path/to/model.ckpt
```

#### Ensembles

Sometimes, we may wish to evaluate an ensemble of models. To do this, first fill out a config using [this file](deep_uncertainty/evaluation/sample_ensemble_config.yaml) as a template. Then run:

```bash
python deep_uncertainty/evaluation/eval_ensemble.py --config path/to/config.yaml
```

### Adding New Models

All regression models should inherit from the `DiscreteRegressionNN` class (found [here](deep_uncertainty/models/discrete_regression_nn.py)). This base class is a `lightning` module, which allows for a lot of typical NN boilerplate code to be abstracted away. Beyond setting a few class attributes like `loss_fn` while calling the super-initializer, the only methods you need to actually write to make a new module are:

- `_forward_impl` (defines a forward pass through the network)
- `_predict_impl` (defines how to make predictions with the network, including any transformations on the output of the forward pass)
- `_point_prediction` (defines how to interpret network output as a single point prediction for a regression target)
- `_addl_test_metrics_dict` (defines any metrics beyond rmse/mae that are computed during model evaluation)
- `_update_addl_test_metrics_batch` (defines how to update additional metrics beyond rmse/mae for each test batch).

See existing model classes like `GaussianNN` (found [here](deep_uncertainty/models/gaussian_nn.py)) for an example of these steps.
