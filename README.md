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

For example, to get a `.npz` file for the Sales dataset, run the following:

```bash
python deep_uncertainty/etl/get_sales_data.py --output-dir path/to/your/data/dir
```

### Running Experiments

To run an experiment, first fill out a config (using [this config](deep_uncertainty/experiments/sample_config.yaml) as a template). Then, from the terminal, run

```bash
python deep_uncertainty/experiments/run_experiment.py --config path/to/your/config.yaml
```

Results / saved model weights will log to the locations specified in your config.

#### Experiments with Tabular Datasets

If running an experiment with tabular data, the experiment script assumes the dataset will be stored locally in `.npz` files with `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, and `y_test` splits (these files are automatically produced by our [data generating code](deep_uncertainty/data_generator.py)). Pass a path to this `.npz` file in the `dataset` `spec` key in the config (also ensure that the `dataset` `type` is set to `tabular`).

#### Experiments with Image Datasets

Currently, the only supported image datasets are MNIST and Coin Counting. To run an experiment with MNIST, simply specify `image` for the `dataset` `type` key in the experiment config, then set `dataset` `spec` to `mnist`.

For Coin Counting, things are a bit hard-coded at the moment. Talk to Spencer if you want to run experiments with this dataset.

### Adding New Models

All regression models should inherit from the `BaseRegressionNN` class (found [here](deep_uncertainty/models/base_regression_nn.py)). This base class is a `lightning` module, which allows for a lot of typical NN boilerplate code to be abstracted away. Beyond setting a few class attributes like `loss_fn` while calling the super-initializer, the only methods you need to actually write to make a new model are:

- `_forward_impl` (defines a forward pass through the network)
- `_predict_impl` (defines how to make a forward pass through the network, along with how to transform model output to represent the quantity being predicted)
- `_test_metrics_dict` (defines the metrics that are computed during model evaluation)
- `_update_test_metrics_batch` (defines how to update those metrics for each test batch). See existing model classes like `GaussianNN` (found [here](deep_uncertainty/models/gaussian_nn.py)) for an example of these steps.
