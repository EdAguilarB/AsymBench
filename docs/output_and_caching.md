# Output Files & Run Caching

## Per-run directory (`log_dirs.runs/…`)

Run directories are named after the resolved representation label:

```
runs/
└── morgan_2048/
│   └── random_forest/random/train_0p80/seed_0/
│       ├── predictions.csv
│       ├── parity_test.png
│       └── metrics.json
└── graph/
    └── gcn/scaffold/train_0p80/seed_2/
        ├── predictions.csv
        ├── parity_test.png
        └── metrics.json
```

## Feature cache (`log_dirs.benchmark/representations/…`)

Fit-free representations are precomputed once on the full dataset:

```
benchmark/representations/
    ├── morgan_2048_a1b2c3d4.csv   # <name>_<hash>.csv
    ├── bespoke_v1_c9d0e1f2.csv
    └── ChemBERTa-77M-MLM_g3h4i5j6.csv
```

The hash covers `type` + `params` + dataset path. **Renaming a representation never invalidates its cache** — only changing `params` or the dataset path does.

## Benchmark results (`log_dirs.benchmark/…`)

```
benchmark/
└── raw_results.json
```

Each entry in `raw_results.json`:

```json
{
  "representation": "morgan_2048",
  "model": "random_forest",
  "split": "scaffold",
  "seed": 2,
  "rmse": 0.84,
  "mae": 0.61,
  "r2": 0.91,
  "rep_type": "morgan",
  "split_sampler": "scaffold",
  "train_size": 0.8,
  "cache_hit": false
}
```

## Run caching and reproducibility

Each run is uniquely identified by a **signature** built from:
- Representation label
- Model type
- Split sampler + train size + split column
- Random seed

Completed runs are loaded from disk and skipped. This enables:
- Resuming interrupted benchmarks without data loss
- Adding representations or models to an existing benchmark without rerunning everything
- Parallel execution across machines on a shared filesystem

> **Tip:** Adding a `name:` to a representation that previously had none changes its signature and treats those runs as new. Keep labels consistent when continuing an existing benchmark.

## Extending with new representations or models

### New molecular representation

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

### New traditional ML model

```python
# asymbench/models/base.py
elif model_type == "my_model":
    _set_if_missing(params, "random_state", seed)
    return MyModel(**params)
```
