# AsymBench: Benchmarking Framework for Asymmetric Reaction Modelling

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-blueviolet)
![Research](https://img.shields.io/badge/purpose-research-critical)
![Benchmarking](https://img.shields.io/badge/type-benchmarking-informational)
![Reproducible](https://img.shields.io/badge/experiments-reproducible-brightgreen)
![Optuna](https://img.shields.io/badge/HPO-Optuna-ff69b4)
![Scikit-Learn](https://img.shields.io/badge/ML-scikit--learn-f7931e)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-ec6f00)
![RDKit](https://img.shields.io/badge/chemistry-RDKit-darkgreen)

**AsymBench** is a modular, reproducible benchmarking framework for evaluating molecular representations and machine learning models in the prediction of asymmetric reaction outcomes.

It is designed for **research-grade experiments**, enabling systematic comparison of:

- Molecular representations (fingerprints, descriptors, bespoke features)
- Machine learning models (RF, SVR, XGBoost, MLP, GNNs)
- Data splitting strategies
- Training set sizes
- Random seeds

The framework automatically performs:

- Data loading
- Representation generation
- Feature preprocessing
- Target scaling
- Hyperparameter optimization
- Model training
- Evaluation
- Statistical analysis
- Plot generation
- Reproducible run caching

---

## Key Features

### Reproducible Benchmarking
- Fully configuration-driven experiments via YAML
- Deterministic splits using seeds
- Automatic caching of completed runs
- Resume interrupted experiments

### Multiple Molecular Representations
- Morgan fingerprints
- RDKit descriptors
- Circus descriptors
- Precomputed/bespoke features from dataframes
- Easily extensible representation interface

### Multiple Machine Learning Models
- Random Forest
- Support Vector Regression
- XGBoost
- MLPRegressor
- Graph Neural Networks (optional)

### Automated Hyperparameter Optimization
- Optuna-based optimization
- Cross-validation on training set only
- Model-specific search spaces
- Fully configurable from YAML

### Advanced Data Splitting
- Random split
- Scaffold split
- Target-property-based split
- Train size sweeps
- Multi-seed evaluation

### Built-in Analysis Tools
- Parity plots
- Distribution plots
- Statistical tests:
  - Friedman test
  - Pairwise Wilcoxon comparisons

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/asymbench.git
cd asymbench
```

### 2. Create environment

```bash
conda create -n asymbench python=3.11
conda activate asymbench
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Prepare your dataset

Example CSV:
```
substrate_smiles,ligand_smiles,solvent_smiles,ddG
C=CC,O=P(...),CCO,1.23
...
```

### 2. Configure experiment (YAML)
Example:
``` yaml
dataset:
  path: data/DAAA.csv
  smiles_columns: [substrate_smiles, ligand_smiles, solvent_smiles]
  target: ddG
  id_col: Example

representations:
  - type: morgan
    params:
      radius: 2
      n_bits: 2048

models:
  - type: random_forest
    hpo:
      enabled: true
      n_trials: 50
      cv: 3
      scoring: rmse
      search_space:
        n_estimators: {type: int, low: 100, high: 1200}
```

### 3. Run the benchmark

``` bash
python benchmarks/run_benchmark.py
```

Results will be saved in:
```
experiments/
├── runs/
├── plots/
└── results/
```

---
## Run Caching and Reproducibility
Each experiment is uniquely identified by:
- Dataset
- Representation
- Model
- Split strategy
- Train size
- Seed
If a run already exists, it is loaded instead of recomputed.

This enables:
- Interrupted experiment recovery
- Large-scale benchmarking
- Parallel execution across machines

---

## Adding New Representations
Create a new class inheriting from:

``` python
BaseSmilesFeaturizer
```

or

``` python
BaseRepresentation
```

Then register in

``` python
asymbench/representations/base.py
```

## Adding New Models
Add your model in:

``` python
asymbench/models/base.py
```

Example:

``` python
elif model_type == "my_model":
    return MyModel(**params)
```

---
## Citation
Coming soon :)

## License
MIT License

## Contact
Eduardo Aguilar: eduardo.aguilar-bejarano@gmail.com
