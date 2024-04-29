# Faithful Heteroscedasticity

This repository is baseline implementation of
[''Faithful Heteroscedastic Regression with Neural Networks.''](https://arxiv.org/abs/2212.09184v1)

## Requirements
We use conda to obtain the tensorflow2.8.0, after that, please use `pip install -r xxx` to install requirements listed in `requirements.txt`.

1. conda create -n tf-gpu python=3.8
2. conda activate tf-gpu
3. pip install tensorflow-gpu==2.8.0
4. pip install protobuf==3.20.*
5. pip install matplotlib
6. pip install -r requirements.txt

Original file requires GPU on tensorflow archi, this requirement has modified to automatically judge and decide on the current running system.

## New dataset integration
pending...


## Downloading the Datasets
To download the UCI and VAE datasets, run:
```
python3 datasets.py
```
To ease reproducibility for those without computational biology backgrounds, we provide the three CRISPR-Cas13 efficacy datasets from our manuscript as pickle files:
```
data/crispr/flow-cytometry-HEK293.pkl
data/crispr/survival-screen-A375.pkl
data/crispr/survival-screen-HEK293.pkl
```

## Reproducing Results
Executing the following commands will run our experiments and perform our analyses.
Running the complete set of commands from any of the subsequent subsections will create the `experiments` and  `results` directories.
The former will contain model weights.
The latter will contain the plots and tables from our manuscript.

### Convergence Experiments
```
python3 experiments_convergence.py
python3 analysis.py --experiment convergence --model_class "Normal"
python3 analysis.py --experiment convergence --model_class "Deep Ensemble"
python3 analysis.py --experiment convergence --model_class "Monte Carlo Dropout"
python3 analysis.py --experiment convergence --model_class "Student"
```

### UCI Regression Experiments
```
python3 experiments_uci.py --dataset boston
python3 experiments_uci.py --dataset carbon
python3 experiments_uci.py --dataset concrete
python3 experiments_uci.py --dataset energy
python3 experiments_uci.py --dataset naval
python3 experiments_uci.py --dataset "power plant"
python3 experiments_uci.py --dataset protein
python3 experiments_uci.py --dataset superconductivity
python3 experiments_uci.py --dataset wine-red
python3 experiments_uci.py --dataset wine-white
python3 experiments_uci.py --dataset yacht
python3 analysis.py --experiment uci --model_class "Normal"
python3 analysis.py --experiment uci --model_class "Deep Ensemble"
python3 analysis.py --experiment uci --model_class "Monte Carlo Dropout"
python3 analysis.py --experiment uci --model_class "Student"
```

### VAE Experiments
```
python3 experiments_vae.py --dataset mnist
python3 experiments_vae.py --dataset fashion_mnist
python3 analysis.py --experiment vae
```

### CRISPR-Cas13 Experiments
```
python3 experiments_crispr.py --dataset flow-cytometry-HEK293
python3 experiments_crispr.py --dataset survival-screen-A375
python3 experiments_crispr.py --dataset survival-screen-HEK293
python3 analysis.py --experiment crispr
```
