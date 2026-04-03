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
![PyG](https://img.shields.io/badge/GNN-PyTorch_Geometric-orange)
![RDKit](https://img.shields.io/badge/chemistry-RDKit-darkgreen)

<p align="center">
  <img src="static/asymbench_logo.png" alt="AsymBench logo" width="220"/>
</p>

**AsymBench** is a modular, reproducible benchmarking framework for evaluating molecular representations and machine learning models in the prediction of asymmetric reaction outcomes.

It is designed for **research-grade experiments**, enabling systematic comparison of:

- Molecular representations (fingerprints, descriptors, deep learning embeddings, graph representations, bespoke features)
- Machine learning models (RF, SVR, XGBoost, MLP, and Graph Neural Networks)
- Data splitting strategies
- Training set sizes
- Random seeds

The framework automatically performs:

- Data loading and validation
- Representation generation (with caching)
- Feature preprocessing and scaling
- Hyperparameter optimisation (Optuna)
- Model training with full reproducibility
- Evaluation (RMSE, MAE, R²)
- Explainability (SHAP for traditional ML; Integrated Gradients for GNNs)
- Publication-ready plot generation
- Reproducible run caching and resumption

---

## Table of Contents

1. [Key Features](#key-features)
2. [Installation](#installation)
3. [Input Data Format](#input-data-format)
4. [Quick Start](#quick-start)
5. [YAML Configuration Reference](#yaml-configuration-reference)
   - [dataset](#dataset)
   - [representations](#representations)
   - [models — traditional ML](#models--traditional-ml)
   - [models — GNNs](#models--graph-neural-networks)
   - [preprocessing](#preprocessing)
   - [target_scaling](#target_scaling)
   - [splits](#splits)
   - [explainability](#explainability)
   - [log_dirs](#log_dirs)
6. [Graph Neural Networks](#graph-neural-networks)
7. [Explainability](#explainability-1)
8. [GPU Setup](#gpu-setup)
9. [Output Files](#output-files)
10. [Run Caching and Reproducibility](#run-caching-and-reproducibility)
11. [Extending the Framework](#extending-the-framework)
12. [Citation / License / Contact](#citation)

---

## Key Features

### Reproducible Benchmarking
- Fully configuration-driven experiments via a single YAML file
- All RNGs seeded (Python, NumPy, PyTorch, CUDA) for identical results across machines
- Automatic run caching — completed runs are loaded from disk, not recomputed
- Interrupted experiments can be resumed without losing work

### Molecular Representations
| Key | Description |
|-----|-------------|
| `morgan` | Morgan circular fingerprints |
| `rdkit` | RDKit 2D molecular descriptors |
| `circus` | CIRCuS corpus-fit descriptors (training-set-aware) |
| `hf_transformer` | HuggingFace transformer embeddings (ChemBERTa, MolT5, …) |
| `unimol` | UniMol v1/v2 3D embeddings |
| `bespoke` / `precomputed` / `df_lookup` | Pre-computed features loaded from CSV/Parquet |
| `graph` | Reaction graph (merged disconnected molecular graph for GNNs) |

### Traditional ML Models
| Key | Description |
|-----|-------------|
| `random_forest` | Random Forest Regressor |
| `svr` | Support Vector Regression |
| `xgb` | XGBoost Regressor |
| `mlp` | Multi-Layer Perceptron |

### Graph Neural Network Models
| Architecture | Key | Edge features | Reference |
|---|---|---|---|
| Graph Convolutional Network | `gcn` | ✗ | Kipf & Welling, 2017 |
| Graph Attention Network v2 | `gat` | ✓ | Brody et al., 2022 |
| Graph Isomorphism Network + E | `gin` | ✓ | Hu et al., 2020 |

### Hyperparameter Optimisation
- Optuna-based, cross-validated on the training set only
- Per-model search spaces defined directly in YAML
- Configurable number of trials and folds

### Data Splitting
- Powered by [astartes](https://github.com/JacksonBurns/astartes)
- `random`, `scaffold`, `target_property`, `external` samplers
- Train-size sweeps and multi-seed evaluation

### Explainability
- **Traditional ML** — SHAP (TreeExplainer or generic Explainer): beeswarm summary plots + CSV values
- **GNNs** — Integrated Gradients (signed, no external dependencies): fragment-level beeswarm plots + CSV values

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/EdAguilarB/asymbench.git
cd DAAA_ML_Benchmarking
```

### 2. Create the conda environment
```bash
conda create -n asymmetric_benchmark python=3.11
conda activate asymmetric_benchmark
```

### 3. Install dependencies
```bash
pip install poetry
poetry install
```

### 4. (Optional) GPU support
The GNN pipeline auto-detects CUDA. If you are running on a GPU server, reinstall PyTorch with the correct CUDA wheel **before** running:

```bash
# Check your CUDA version first
nvidia-smi   # look at "CUDA Version:" in the top-right corner

# For CUDA 12.1
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121 \
    --force-reinstall

# Then reinstall torch-geometric against the new build
TORCH=$(python -c "import torch; print(torch.__version__)")
pip install torch-geometric \
    -f https://data.pyg.org/whl/torch-${TORCH}+cu121.html

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

Replace `cu121` with `cu118` for CUDA 11.8.

---

## Input Data Format

### Required CSV structure

The dataset must be a **CSV file** where:
- Each row is one reaction
- Molecular components are represented as **SMILES strings** in dedicated columns
- The regression **target** (e.g. ΔΔG‡ in kJ/mol) is a numeric column
- An optional **ID column** uniquely identifies each row

```
id,substrate_smiles,ligand_smiles,solvent_smiles,ddG
1,O=C1C(C(OCC=C)=O)(c2ccccc2)CCC1,O=P1(O)Oc2ccccc2-c2ccccc21,ClCCl,-6.11
2,O=C1C(C(OCC=C)=O)(c2ccc(OC)cc2)CCC1,O=P1(O)Oc2ccccc2-c2ccccc21,ClCCl,-6.05
...
```

Rules:
- There is **no limit on the number of SMILES columns** — list all reaction components (substrate, ligand, additive, solvent, …)
- The SMILES columns are listed in `smiles_columns` in the YAML config
- All SMILES columns listed there are used to build the graph representation; other columns are ignored
- For **bespoke / precomputed representations**, a separate CSV or Parquet file may be provided containing pre-computed numerical features indexed by the same ID column

### Bespoke features CSV

When using `type: bespoke` or `type: precomputed`, provide a second file:

```
id,steric_param,hammett_sigma,solvent_dielectric,...
1,19.96,0.00,8.93,...
2,17.66,-0.27,8.93,...
```

The `index_col` key tells AsymBench which column to use as the join key (must match `id_col` in the main dataset).

---

## Quick Start

### 1. Place your data
```
data/
└── my_reaction/
    ├── reactions.csv          # main dataset
    └── bespoke_features.csv   # optional pre-computed features
```

### 2. Create a config file
```
benchmarks/
└── my_reaction/
    └── benchmark_config.yaml
```

### 3. Run
```bash
python -m benchmarks.run_benchmark --config benchmarks/my_reaction/benchmark_config.yaml
```

Results are written to the directories specified in `log_dirs`.

---

## YAML Configuration Reference

Below is a fully annotated template covering every supported key. Copy it, remove the sections you don't need, and adjust values.

```yaml
# =============================================================
#  AsymBench — benchmark_config.yaml  (full reference template)
# =============================================================

# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
dataset:
  path: data/my_reaction/reactions.csv   # path to the main CSV
  smiles_columns:                        # all molecular component columns
    - substrate_smiles
    - ligand_smiles
    - solvent_smiles
  target: ddG                            # numeric regression target column
  id_col: Example                        # unique row identifier column

# Optional external hold-out set — used with sampler: external
external_test_set:
  path: data/my_reaction/test_set.csv
  smiles_columns:
    - substrate_smiles
    - ligand_smiles
    - solvent_smiles
  target: ddG
  id_col: Example


# ─────────────────────────────────────────────────────────────
# REPRESENTATIONS
# One entry per representation; the benchmark crosses all
# representations × all models automatically.
# GNN models (type: gnn) only run with type: graph.
# All other models only run with non-graph representations.
# ─────────────────────────────────────────────────────────────
representations:

  # ── Graph (GNN input) ──────────────────────────────────────
  - type: graph
    params:
      include_hydrogens: false   # include explicit H atoms in the graph

  # ── Morgan fingerprints ────────────────────────────────────
  - type: morgan
    params:
      radius: 2
      n_bits: 2048

  # ── RDKit 2D descriptors ───────────────────────────────────
  - type: rdkit

  # ── CIRCuS (corpus-fit, training-set-aware) ────────────────
  - type: circus
    params:
      radius: 2

  # ── HuggingFace transformer embeddings ────────────────────
  - type: hf_transformer
    params:
      model_type: chemberta                         # chemberta | molt5
      model_name: DeepChem/ChemBERTa-77M-MLM        # HF model identifier
      pooling: mean                                 # mean | cls
      device: cpu                                   # cpu | cuda

  - type: hf_transformer
    params:
      model_type: molt5

  # ── UniMol 3D embeddings ───────────────────────────────────
  - type: unimol
    params:
      model_name: unimolv1          # unimolv1 | unimolv2
      data_type: molecule
      remove_hs: false

  # ── Bespoke / precomputed features ────────────────────────
  - type: bespoke                   # also accepted: precomputed | df_lookup
    params:
      features_path: data/my_reaction/bespoke_features.csv
      feature_name: v1              # label used in output filenames
      index_col: Example            # column in features CSV matching id_col
      feature_columns:              # which columns to use (omit = use all)
        - steric_param
        - hammett_sigma
        - solvent_dielectric
      prefix: Bespoke               # optional prefix for column names
      strict: true                  # raise error if index mismatch


# ─────────────────────────────────────────────────────────────
# MODELS — Traditional ML
# Each model is crossed with every non-graph representation.
# ─────────────────────────────────────────────────────────────
models:

  # ── Random Forest ─────────────────────────────────────────
  - type: random_forest
    hpo:
      enabled: true
      n_trials: 50
      cv: 3
      scoring: rmse
      search_space:
        n_estimators: {type: int,   low: 100,  high: 1200}
        max_depth:    {type: int,   low: 3,    high: 30}
        min_samples_leaf: {type: int, low: 1,  high: 10}

  # ── SVR ───────────────────────────────────────────────────
  - type: svr
    hpo:
      enabled: true
      n_trials: 50
      cv: 3
      scoring: rmse
      search_space:
        C:       {type: float, low: 1e-2, high: 1e3,  log: true}
        gamma:   {type: float, low: 1e-6, high: 1e0,  log: true}
        epsilon: {type: float, low: 1e-3, high: 1.0,  log: true}

  # ── XGBoost ───────────────────────────────────────────────
  - type: xgb
    hpo:
      enabled: true
      n_trials: 50
      cv: 3
      scoring: rmse
      search_space:
        n_estimators:      {type: int,   low: 200,  high: 1500}
        max_depth:         {type: int,   low: 3,    high: 10}
        learning_rate:     {type: float, low: 0.01, high: 0.3, log: true}
        subsample:         {type: float, low: 0.5,  high: 1.0}
        colsample_bytree:  {type: float, low: 0.5,  high: 1.0}
        min_child_weight:  {type: float, low: 1.0,  high: 10.0}
        reg_alpha:         {type: float, low: 1e-8, high: 10.0, log: true}
        reg_lambda:        {type: float, low: 1e-3, high: 10.0, log: true}

  # ── MLP ───────────────────────────────────────────────────
  - type: mlp
    hpo:
      enabled: true
      n_trials: 50
      cv: 3
      scoring: rmse
      search_space:
        hidden_layer_sizes:
          type: categorical
          choices: [[16], [32], [64], [16,16], [32,16], [32,32], [64,32]]
        activation:
          type: categorical
          choices: [relu]
        alpha:
          type: float
          low: 1e-6
          high: 1e-1
          log: true
        learning_rate_init:
          type: float
          low: 1e-4
          high: 1e-2
          log: true
        batch_size:
          type: categorical
          choices: [32, 64, 128]


# ─────────────────────────────────────────────────────────────
# MODELS — Graph Neural Networks
# All GNN entries must use type: gnn.
# They are automatically paired with type: graph representations.
# Multiple GNN architectures can be listed and will be run
# independently, each identified by its architecture name.
# ─────────────────────────────────────────────────────────────

  # ── GCN (no edge features) ────────────────────────────────
  - type: gnn
    params:
      architecture: gcn       # gcn | gat | gin  [default: gcn]

      # Shared parameters
      hidden_dim: 64          # conv and readout layer width     [default: 64]
      num_layers: 3           # number of conv layers            [default: 3]
      pooling: mean           # mean | add | max | mean_max      [default: mean]
                              # mean_max concatenates mean+max,
                              # doubling the embedding dimension
      readout_layers: 2       # MLP depth including output       [default: 2]
                              # 2 → hidden → hidden//2 → 1
      dropout: 0.0            # dropout probability              [default: 0.0]

      # GCN-specific
      improved: false         # self-loop weight = 2             [default: false]

      # Training parameters (not passed to the model constructor)
      epochs: 100
      lr: 0.001
      batch_size: 32
      num_workers: 0          # parallel data-loading workers    [default: 0]
                              # set to 4–8 on GPU servers

  # ── GAT (Graph Attention v2, uses edge features) ──────────
  - type: gnn
    params:
      architecture: gat

      hidden_dim: 64
      num_layers: 3
      pooling: mean
      readout_layers: 2
      dropout: 0.1            # applied inside attention layers too

      # GAT-specific
      num_heads: 4            # attention heads per layer        [default: 4]
                              # heads averaged (concat=False) so
                              # output width stays hidden_dim
      edge_in_dim: 8          # bond feature dimension           [default: 8]

      epochs: 100
      lr: 0.001
      batch_size: 32
      num_workers: 0

  # ── GIN-E (Graph Isomorphism + edge features) ─────────────
  - type: gnn
    params:
      architecture: gin

      hidden_dim: 64
      num_layers: 4           # GIN benefits from more layers
      pooling: mean_max
      readout_layers: 2
      dropout: 0.0

      # GIN-specific
      train_eps: false        # learn epsilon per layer          [default: false]
      edge_in_dim: 8          # bond feature dimension           [default: 8]

      epochs: 150
      lr: 0.001
      batch_size: 32
      num_workers: 0


# ─────────────────────────────────────────────────────────────
# PREPROCESSING  (traditional ML only — ignored for GNNs)
# ─────────────────────────────────────────────────────────────
preprocessing:
  scaling: minmax             # minmax | standard | none

  feature_selection:
    variance_filter:
      enabled: true
      threshold: 0.0          # remove features with zero variance

    correlation_filter:
      enabled: true
      threshold: 0.95         # remove one of each correlated pair
      method: pearson         # pearson | spearman


# ─────────────────────────────────────────────────────────────
# TARGET SCALING
# Applied to the regression target for all model types.
# ─────────────────────────────────────────────────────────────
target_scaling:
  scaling: minmax             # minmax | standard | none


# ─────────────────────────────────────────────────────────────
# SPLITS
# ─────────────────────────────────────────────────────────────
splits:
  - sampler: random           # random train/test split
  - sampler: scaffold         # scaffold-based split (astartes)
  - sampler: target_property  # split on the target value distribution

# For external hold-out only (requires external_test_set above):
# splits:
#   - sampler: external

split_by_mol_col:
  - substrate_smiles          # column(s) used as the molecular identity
                              # for scaffold / target_property splits

train_set_sizes:
  - 0.8                       # fraction of data used for training
  - 0.6
  - 0.4

seeds: [0, 1, 2, 3, 4]       # each seed = one independent run


# ─────────────────────────────────────────────────────────────
# EXPLAINABILITY
# Disabled by default (computationally expensive).
# Traditional ML uses SHAP; GNNs use Integrated Gradients.
# ─────────────────────────────────────────────────────────────
explainability:
  enabled: false              # set to true to activate

  # Traditional ML (SHAP)
  max_background: 200         # background samples for KernelExplainer
  max_explain: 500            # max samples to compute SHAP values for

  # GNN (Integrated Gradients)
  n_steps: 50                 # Riemann-sum steps; use ≥ 100 for publication
  fragmentation: brics        # brics | murcko_scaffold


# ─────────────────────────────────────────────────────────────
# OUTPUT DIRECTORIES
# ─────────────────────────────────────────────────────────────
log_dirs:
  runs: experiments/my_reaction/runs           # individual run artefacts
  benchmark: experiments/my_reaction/benchmark # aggregated results
```

---

## Graph Neural Networks

### How the graph representation works

Each reaction is encoded as a **single disconnected molecular graph**: all participant molecules (substrate, ligand, solvent, …) are individually converted to graphs and concatenated into one PyG `Data` object with a correctly offset `edge_index`. A single GNN operates on this merged graph, and global pooling produces a fixed-size reaction embedding.

**Node features (33-dimensional):**
Element (one-hot), degree, hybridisation, formal charge, chirality, aromaticity, in-ring flag.

**Edge features (8-dimensional):**
Bond type, stereo configuration, in-ring, conjugated.

### Architecture overview

```
Reaction SMILES
    ↓  (one graph per molecule)
Merged reaction graph  [N nodes, E edges]
    ↓
Graph conv layers × num_layers  (GCN / GAT / GIN)
+ BatchNorm + ReLU + Dropout
    ↓
Global pooling  (mean / add / max / mean_max)
    ↓
MLP readout  (hidden → hidden//2 → … → 1)
    ↓
Predicted ΔΔG‡
```

### GNN parameter quick-reference

| Parameter | Applies to | Choices / type | Default |
|---|---|---|---|
| `architecture` | all | `gcn` / `gat` / `gin` | `gcn` |
| `hidden_dim` | all | int | `64` |
| `num_layers` | all | int ≥ 1 | `3` |
| `pooling` | all | `mean` / `add` / `max` / `mean_max` | `mean` |
| `readout_layers` | all | int ≥ 1 | `2` |
| `dropout` | all | 0.0 – 1.0 | `0.0` |
| `improved` | GCN | bool | `false` |
| `num_heads` | GAT | int | `4` |
| `train_eps` | GIN | bool | `false` |
| `edge_in_dim` | GAT / GIN | int | `8` |
| `epochs` | all | int | `100` |
| `lr` | all | float | `0.001` |
| `batch_size` | all | int | `32` |
| `num_workers` | all | int | `0` |

`mean_max` pooling concatenates global mean and max pools, doubling the embedding fed to the readout MLP (e.g. `hidden_dim=64` → 128-dim input to the MLP).

---

## Explainability

### Traditional ML — SHAP

SHAP values are computed on both the train and test sets.

- Tree-based models (RF, XGBoost): `shap.TreeExplainer` (exact, fast)
- Other models (SVR, MLP): generic `shap.Explainer` with a sampled background

**Outputs per split (train / test):**
| File | Contents |
|---|---|
| `*_shap_importance.csv` | Mean absolute SHAP per feature, ranked |
| `*_shap_summary_beeswarm.png` | SHAP beeswarm summary plot (top features) |

### GNNs — Integrated Gradients

Integrated Gradients (Sundararajan et al., 2017) attribute the model prediction to input node features by integrating the gradient along a straight-line path from an all-zeros baseline. Crucially, the scores are **signed**:

- **Positive** — the feature/fragment pushes the prediction above baseline
- **Negative** — the feature/fragment pulls the prediction below baseline

Attribution is then aggregated to **molecular fragment level** using BRICS decomposition or Murcko scaffolds. Each fragment is tracked per reaction component (substrate, ligand, solvent, …) so fragments appearing in multiple components are never merged.

**Outputs per split (train / test):**
| File | Contents |
|---|---|
| `node_masks.npz` | Per-graph signed IG scores for every node feature |
| `fragment_importances.csv` | Long-format CSV: `fragment`, `source`, `importance` — one row per score |
| `fragment_beeswarm.png` | SHAP-style beeswarm: top-k positive and top-k negative fragments |

The `fragment_importances.csv` long format (one row per individual IG score, no aggregation) is ready to use directly with seaborn `stripplot` / `swarmplot` or `shap.plots.beeswarm`.

---

## GPU Setup

The entire GNN pipeline automatically uses CUDA when `torch.cuda.is_available()` returns `True`. No code changes are needed — only the PyTorch installation must match the server's CUDA version (see [Installation → GPU support](#installation)).

**GPU-recommended YAML settings:**
```yaml
params:
  num_workers: 4    # parallel data loading; set to CPU core count
  batch_size: 64    # larger batches fit on GPU memory
```

**Reproducibility note:** `torch.backends.cudnn.benchmark` is set to `False` and `deterministic` to `True` before every run. This trades ~5–15 % GPU speed for bit-exact reproducibility across seeds and machines. This is the right default for research.

---

## Output Files

### Per-run directory (`log_dirs.runs/…`)

```
runs/
└── rep=graph__model=gat__sampler=scaffold__train=0.8__mol=substrate_smiles__seed=2/
    ├── predictions.csv          # train + test rows with true and predicted target
    ├── parity_test.png          # parity plot for the test set
    ├── metrics.json             # RMSE, MAE, R², run metadata
    └── explainability/
        ├── train/
        │   ├── node_masks.npz
        │   ├── fragment_importances.csv
        │   └── fragment_beeswarm.png
        └── test/
            ├── node_masks.npz
            ├── fragment_importances.csv
            └── fragment_beeswarm.png
```

For traditional ML the `explainability/` folder contains SHAP files instead:
```
        └── explainability/
            ├── train_shap_importance.csv
            ├── train_shap_summary_beeswarm.png
            ├── test_shap_importance.csv
            └── test_shap_summary_beeswarm.png
```

### Benchmark directory (`log_dirs.benchmark/…`)

```
benchmark/
└── raw_results.json    # aggregated metrics for every run in the experiment
```

Each entry in `raw_results.json`:
```json
{
  "representation": "graph",
  "model": "gat",
  "split": "scaffold",
  "seed": 2,
  "rmse": 0.84,
  "mae": 0.61,
  "r2": 0.91,
  "model_type": "gat",
  "rep_type": "graph",
  "split_sampler": "scaffold",
  "train_size": 0.8,
  "cache_hit": false
}
```

---

## Run Caching and Reproducibility

Each run is uniquely identified by a **signature** built from:
- Representation type + params
- Model architecture + type
- Split sampler + train size + split column
- Random seed

If a run with an identical signature has already completed, its metrics are loaded from disk and the run is skipped. This enables:

- Resuming interrupted benchmarks without data loss
- Adding new representations or models to an existing benchmark without rerunning everything
- Parallel execution across machines (shared network filesystem)

---

## Extending the Framework

### Adding a new molecular representation

```python
# 1. Create asymbench/representations/my_rep.py
from asymbench.representations.base import BaseSmilesFeaturizer

class MyFeaturizer(BaseSmilesFeaturizer):
    @property
    def feature_dim_per_mol(self) -> int: ...
    def featurize_mol(self, mol) -> np.ndarray: ...
    def feature_names_per_mol(self) -> list[str]: ...

# 2. Register in asymbench/representations/__init__.py
if rep_type == "my_rep":
    return MyFeaturizer(config)
```

### Adding a new traditional ML model

```python
# asymbench/models/base.py (or equivalent)
elif model_type == "my_model":
    _set_if_missing(params, "random_state", seed)
    return MyModel(**params)
```

### Adding a new GNN architecture

```python
# 1. Create asymbench/gnn/architectures/my_arch.py
from asymbench.gnn.base import BaseReactionGNN

class ReactionMyArch(BaseReactionGNN):
    ARCH_NAME = "my_arch"

    def __init__(self, node_in_dim, hidden_dim=64, num_layers=3,
                 pooling="mean", readout_layers=2, dropout=0.0,
                 my_param=..., **kwargs):
        super().__init__(node_in_dim, hidden_dim, num_layers,
                         pooling, readout_layers, dropout)
        # define self.conv_layers and self.norm_layers here
        self.make_readout_layers()

    # Override get_graph_embedding() if your conv signature differs
    # from conv(x, edge_index, edge_attr)

# 2. Register in asymbench/gnn/architectures/__init__.py
from asymbench.gnn.architectures.my_arch import ReactionMyArch
_REGISTRY["my_arch"] = ReactionMyArch
```

Then use `architecture: my_arch` in the YAML `params` block.

---

## Citation

Coming soon :)

## License

MIT

## Contact

Eduardo Aguilar: ed.aguilar.bejarano@gmail.com
