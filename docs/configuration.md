# YAML Configuration Reference

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
  reaction_features:                     # optional: numeric columns to append to
    - temperature                        # molecular features before normalisation
    - reaction_time                      # (traditional ML only; omit if unused)

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
#
# NAMING: every representation entry accepts an optional "name:"
# field (see Representation Names below for details).
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
    name: morgan_2048            # optional label (required when using
                                 # multiple configs of the same type)

  - type: morgan
    params:
      radius: 2
      n_bits: 1024
    name: morgan_1024

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
    name: ChemBERTa-77M-MLM

  - type: hf_transformer
    params:
      model_type: molt5
    name: MolT5

  # ── UniMol 3D embeddings ───────────────────────────────────
  - type: unimol
    params:
      model_name: unimolv1          # unimolv1 | unimolv2
      data_type: molecule
      remove_hs: false

  # ── Bespoke / precomputed features ────────────────────────
  # Option A: explicit column list
  - type: bespoke                   # also accepted: precomputed | df_lookup
    params:
      features_path: data/my_reaction/bespoke_features.csv
      feature_name: v1              # label used in output filenames
      index_col: Example            # column in features CSV matching id_col
      feature_columns:              # explicit list of columns to use
        - steric_param
        - hammett_sigma
        - solvent_dielectric
      prefix: bespoke               # prepended to column names as "bespoke__<col>"
                                    # omit or set to ~ for no prefix
      strict: true                  # raise error if index mismatch
    name: bespoke_v1

  # Option B: all-columns mode — feature_columns omitted
  - type: bespoke
    params:
      features_path: data/my_reaction/all_features.csv
      feature_name: all
      index_col: Example            # identifier column; all other cols = features
      strict: true
    name: bespoke_all


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
# ─────────────────────────────────────────────────────────────

  # ── GCN (no edge features) ────────────────────────────────
  - type: gnn
    params:
      architecture: gcn       # gcn | gat | gin  [default: gcn]
      hidden_dim: 64
      num_layers: 3
      pooling: mean           # mean | add | max | mean_max
      readout_layers: 2
      dropout: 0.0
      activation: relu        # relu | leaky_relu | elu | silu | gelu | tanh
      improved: false         # self-loop weight = 2  [default: false]
      epochs: 100
      lr: 0.001
      batch_size: 32
      num_workers: 0

  # ── GAT (Graph Attention v2) ──────────────────────────────
  - type: gnn
    params:
      architecture: gat
      hidden_dim: 64
      num_layers: 3
      pooling: mean
      readout_layers: 2
      dropout: 0.1
      activation: relu
      num_heads: 4            # attention heads per layer  [default: 4]
      edge_in_dim: 8          # bond feature dimension     [default: 8]
      epochs: 100
      lr: 0.001
      batch_size: 32
      num_workers: 0

  # ── GIN-E (Graph Isomorphism + edge features) ─────────────
  - type: gnn
    params:
      architecture: gin
      hidden_dim: 64
      num_layers: 4
      pooling: mean_max
      readout_layers: 2
      dropout: 0.0
      activation: relu
      train_eps: false        # learn epsilon per layer  [default: false]
      edge_in_dim: 8
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

train_set_sizes:
  - 0.8
  - 0.6
  - 0.4

seeds: [0, 1, 2, 3, 4]


# ─────────────────────────────────────────────────────────────
# EXPLAINABILITY
# ─────────────────────────────────────────────────────────────
explainability:
  enabled: false              # set to true to activate

  # Traditional ML (SHAP)
  max_background: 200         # background samples for KernelExplainer
  max_explain: 500            # max samples to compute SHAP values for

  # GNN (Integrated Gradients)
  n_steps: 50                 # Riemann-sum steps; use ≥ 100 for publication
  fragmentation: brics        # brics | murcko_scaffold
  top_k: 20                   # fragments shown in beeswarm plot


# ─────────────────────────────────────────────────────────────
# OUTPUT DIRECTORIES
# ─────────────────────────────────────────────────────────────
log_dirs:
  runs: experiments/my_reaction/runs
  benchmark: experiments/my_reaction/benchmark
```

---

## Representation Names

Every representation entry supports an optional **`name:`** field. It is used consistently in:
- The `"representation"` key in `raw_results.json`
- On-disk feature cache filenames (`benchmark/representations/<name>_<hash>.csv`)
- Run directory paths and log messages

**Required when two representations share the same `type`** — without a name, the second result would silently overwrite the first. AsymBench raises a `ValueError` at startup if two entries resolve to the same label.

### Auto-label fallback

| Representation type | Auto-label |
|---------------------|------------|
| `morgan`, `rdkit`, `circus`, `unimol`, `graph` | `<type>` |
| `hf_transformer` | `hf_transformer_<model_type>` |
| `bespoke`, `df_lookup`, `precomputed` | `<type>_<feature_name>` |

Changing a `name:` does **not** invalidate the on-disk feature cache — only `type`, `params`, and the dataset path affect the cache hash.

---

## Bespoke / Pre-computed Features

These types (`bespoke`, `precomputed`, `df_lookup`) load numerical features from an external CSV or Parquet file joined by a shared index column.

### Explicit column list

```yaml
- type: bespoke
  params:
    features_path: data/features.csv
    feature_name: v1
    index_col: Example
    feature_columns:
      - steric_param
      - hammett_sigma
    prefix: bespoke   # columns become "bespoke__steric_param", etc.
    strict: true
  name: bespoke_v1
```

### All-columns mode

Omit `feature_columns` to use every column in the file (after the index):

```yaml
- type: bespoke
  params:
    features_path: data/all_features.csv
    feature_name: all
    index_col: Example
    strict: true
  name: bespoke_all
```

> Non-numeric columns trigger a warning but do not abort the run.

### Parameter reference

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `features_path` | ✓ | — | Path to the `.csv` or `.parquet` file |
| `index_col` | — | `None` | Join key column |
| `feature_name` | — | — | Label suffix for auto-naming |
| `feature_columns` | — | `None` (all) | Explicit column list |
| `prefix` | — | `""` | Prepended to column names as `<prefix>__<col>` |
| `join_key` | — | `None` | Dataset column to use for lookup instead of `df.index` |
| `strict` | — | `True` | Error on missing index entries; `false` fills with zeros |
