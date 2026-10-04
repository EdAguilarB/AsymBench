# Graph Neural Networks

## How the graph representation works

Each reaction is encoded as a **single disconnected molecular graph**: all participant molecules (substrate, ligand, solvent, …) are individually converted to graphs and concatenated into one PyG `Data` object with a correctly offset `edge_index`. A single GNN operates on this merged graph, and global pooling produces a fixed-size reaction embedding.

**Node features (33-dimensional):**
Element (one-hot), degree, hybridisation, formal charge, chirality, aromaticity, in-ring flag.

**Edge features (8-dimensional):**
Bond type, stereo configuration, in-ring, conjugated.

## Architecture overview

```
Reaction SMILES
    ↓  (one graph per molecule)
Merged reaction graph  [N nodes, E edges]
    ↓
Graph conv layers × num_layers  (GCN / GAT / GIN)
+ BatchNorm + activation + Dropout
    ↓
Global pooling  (mean / add / max / mean_max)
    ↓
MLP readout  (hidden → hidden//2 → … → 1)
    ↓
Predicted ΔΔG‡
```

## Supported architectures

| Architecture | Key | Edge features | Reference |
|---|---|---|---|
| Graph Convolutional Network | `gcn` | ✗ | Kipf & Welling, 2017 |
| Graph Attention Network v2 | `gat` | ✓ | Brody et al., 2022 |
| Graph Isomorphism Network + E | `gin` | ✓ | Hu et al., 2020 |

## Parameter quick-reference

| Parameter | Applies to | Choices / type | Default |
|---|---|---|---|
| `architecture` | all | `gcn` / `gat` / `gin` | `gcn` |
| `hidden_dim` | all | int | `64` |
| `num_layers` | all | int ≥ 1 | `3` |
| `pooling` | all | `mean` / `add` / `max` / `mean_max` | `mean` |
| `readout_layers` | all | int ≥ 1 | `2` |
| `dropout` | all | 0.0 – 1.0 | `0.0` |
| `activation` | all | `relu` / `leaky_relu` / `elu` / `silu` / `gelu` / `tanh` | `relu` |
| `improved` | GCN | bool | `false` |
| `num_heads` | GAT | int | `4` |
| `train_eps` | GIN | bool | `false` |
| `edge_in_dim` | GAT / GIN | int | `8` |
| `epochs` | all | int | `100` |
| `lr` | all | float | `0.001` |
| `batch_size` | all | int | `32` |
| `num_workers` | all | int | `0` |

`mean_max` pooling concatenates global mean and max pools, doubling the embedding fed to the readout MLP (e.g. `hidden_dim=64` → 128-dim input to the MLP).

`activation` controls the non-linearity after every conv layer, between readout MLP layers, and — for GIN — inside each GINEConv aggregation MLP.

## GPU setup

The pipeline auto-detects CUDA. No code changes are needed — only the PyTorch installation must match the server's CUDA version.

```bash
# Check your CUDA version
nvidia-smi   # look at "CUDA Version:" in the top-right corner

# For CUDA 12.1
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121 \
    --force-reinstall

TORCH=$(python -c "import torch; print(torch.__version__)")
pip install torch-geometric \
    -f https://data.pyg.org/whl/torch-${TORCH}+cu121.html

python -c "import torch; print(torch.cuda.is_available())"
```

Replace `cu121` with `cu118` for CUDA 11.8.

**GPU-recommended YAML settings:**
```yaml
params:
  num_workers: 4    # parallel data loading
  batch_size: 64    # larger batches fit on GPU memory
```

**Reproducibility:** `torch.backends.cudnn.benchmark = False` and `deterministic = True` are set before every run, trading ~5–15 % GPU speed for bit-exact results across seeds and machines.

## Adding a new architecture

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

# 2. Register in asymbench/gnn/architectures/__init__.py
from asymbench.gnn.architectures.my_arch import ReactionMyArch
_REGISTRY["my_arch"] = ReactionMyArch
```

Then use `architecture: my_arch` in the YAML `params` block.
