# Explainability

## Traditional ML — SHAP

SHAP values are computed on both the train and test sets.

- Tree-based models (RF, XGBoost): `shap.TreeExplainer` (exact, fast)
- Other models (SVR, MLP): generic `shap.Explainer` with a sampled background

**Outputs per split (train / test):**

| File | Contents |
|---|---|
| `*_shap_importance.csv` | Mean absolute SHAP per feature, ranked |
| `*_shap_summary_beeswarm.png` | SHAP beeswarm summary plot (top features) |

## GNNs — Integrated Gradients

Integrated Gradients (Sundararajan et al., 2017) attribute the model prediction to input node features by integrating the gradient along a straight-line path from an all-zeros baseline. Scores are **signed**:

- **Positive** — the feature/fragment pushes the prediction above baseline
- **Negative** — the feature/fragment pulls the prediction below baseline

Attribution is aggregated to **molecular fragment level** using BRICS decomposition or Murcko scaffolds. Each fragment is tracked per reaction component (substrate, ligand, solvent, …) so the same substructure in different components is never merged.

**Outputs per split (train / test):**

| File | Contents |
|---|---|
| `node_masks.npz` | Per-graph signed IG scores for every node feature |
| `fragment_importances.csv` | `reaction_idx`, `fragment`, `source`, `importance` (mean IG), `count` (occurrences) — one row per (reaction, fragment) pair |
| `fragment_beeswarm.png` | SHAP beeswarm via `shap.summary_plot()` — top-k fragments by mean \|IG\|, colour-coded by occurrence count |

`reaction_idx` matches the original dataset row index, so you can join `fragment_importances.csv` back to your reaction CSV. `count` helps distinguish high-importance scaffolds from rare motifs.

## Output file layout

```
runs/
└── graph/
    └── gin/scaffold/train_0p80/seed_0/
        ├── predictions.csv
        ├── parity_test.png
        ├── metrics.json
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

For traditional ML the `explainability/` folder contains SHAP files:

```
        └── explainability/
            ├── train_shap_importance.csv
            ├── train_shap_summary_beeswarm.png
            ├── test_shap_importance.csv
            └── test_shap_summary_beeswarm.png
```
