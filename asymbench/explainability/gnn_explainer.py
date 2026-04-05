"""GNN explainability via Integrated Gradients + molecular fragment attribution.

Integrated Gradients (Sundararajan et al., 2017) attributes the model output
to input node features by integrating the gradient along a straight-line path
from a baseline (all-zeros) to the actual input.  Crucially, the resulting
attribution scores are **signed**:

* **Positive** → the feature pushes the prediction *above* the baseline
* **Negative** → the feature pushes the prediction *below* the baseline

This mirrors SHAP-value semantics for tabular models and requires nothing
beyond PyTorch — no captum, no numpy-version constraints.

Reference
---------
Sundararajan, M., Taly, A., & Yan, Q. (2017).
Axiomatic Attribution for Deep Networks. ICML 2017.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
import torch
import torch.nn as nn
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# Fragmentation helpers
# ---------------------------------------------------------------------------


class FragmentationMethod(Enum):
    BRICS = "brics"
    MurckoScaffold = "murcko_scaffold"


def _remove_dummy_atoms(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Strip RDKit dummy atoms (atomic number 0) from a molecule."""
    from rdkit.Chem.rdchem import RWMol

    rw = RWMol(mol)
    dummies = sorted(
        [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0],
        reverse=True,
    )
    for idx in dummies:
        rw.RemoveAtom(idx)
    try:
        clean = rw.GetMol()
        Chem.SanitizeMol(clean)
        return clean
    except Exception:
        return None


def _fragments_brics(mol: Chem.Mol) -> Dict[str, List[List[int]]]:
    """BRICS decomposition with atom-index mapping."""
    from rdkit.Chem import BRICS

    raw_frags = BRICS.BRICSDecompose(mol, returnMols=False)
    result: Dict[str, List[List[int]]] = {}

    for frag_smi in raw_frags:
        raw_mol = Chem.MolFromSmiles(frag_smi)
        if raw_mol is None:
            continue
        clean = _remove_dummy_atoms(raw_mol)
        if clean is None or clean.GetNumAtoms() == 0:
            continue
        clean_smi = Chem.MolToSmiles(clean)
        matches = mol.GetSubstructMatches(clean)
        if matches:
            result[clean_smi] = [list(m) for m in matches]

    return result


def _fragments_murcko(mol: Chem.Mol) -> Dict[str, List[List[int]]]:
    """Murcko scaffold with atom-index mapping."""
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffold_smi = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=False
    )
    if not scaffold_smi:
        return {}

    scaffold_mol = Chem.MolFromSmiles(scaffold_smi)
    if scaffold_mol is None:
        return {}

    matches = mol.GetSubstructMatches(scaffold_mol)
    if not matches:
        return {}

    return {scaffold_smi: [list(m) for m in matches]}


def fragment_and_match(
    smiles: str, fragmentation_method: FragmentationMethod
) -> Dict[str, List[List[int]]]:
    """Fragment *smiles* and return atom indices of every fragment occurrence.

    Returns
    -------
    dict
        ``{fragment_smiles: [[atom_idx, ...], ...]}``
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    if fragmentation_method is FragmentationMethod.BRICS:
        return _fragments_brics(mol)
    if fragmentation_method is FragmentationMethod.MurckoScaffold:
        return _fragments_murcko(mol)
    raise ValueError(f"Unknown fragmentation method: {fragmentation_method!r}")


# ---------------------------------------------------------------------------
# Fragment importance aggregation
# ---------------------------------------------------------------------------


def get_fragment_importance(
    data_list: list,
    node_masks: Dict,
    fragmentation_method: FragmentationMethod,
) -> Dict[Tuple[str, str], List[float]]:
    """Aggregate signed per-atom IG scores to fragment-level importances.

    Each reaction graph is a concatenation of molecular graphs.  The function
    iterates over the per-molecule SMILES stored in ``data.mol_smiles``,
    fragments each molecule, and sums the node-mask values over the atoms
    belonging to each fragment occurrence.

    Fragments are keyed by ``(fragment_smiles, source_column)`` so that the
    same substructure appearing in different reaction components (e.g. a
    phenyl ring in both the substrate and the ligand) is tracked separately.

    Fragment scores inherit the sign of the underlying IG attributions —
    a positive score means the fragment contributes towards a higher
    prediction; negative means it pulls the prediction down.

    Parameters
    ----------
    data_list:
        ``dataset._data`` — each item must have ``data.mol_smiles``,
        a list of ``(column_name, smiles)`` pairs.
    node_masks:
        ``{data.idx: np.ndarray shape [num_nodes, num_features]}``
    fragmentation_method:
        How to fragment each molecule.

    Returns
    -------
    dict
        ``{(fragment_smiles, source_col): [signed_score, ...]}``
    """
    frags_importances: Dict[Tuple[str, str], List[float]] = {}

    for data in data_list:
        mol_idx = data.idx
        if mol_idx not in node_masks:
            continue

        node_mask = node_masks[mol_idx]  # [num_nodes, num_features]
        num_atoms = 0

        for col, smiles in data.mol_smiles:
            mol_obj = Chem.MolFromSmiles(smiles)
            mol_num_atoms = mol_obj.GetNumAtoms() if mol_obj is not None else 0

            try:
                frags = fragment_and_match(smiles, fragmentation_method)
            except ValueError:
                num_atoms += mol_num_atoms
                continue

            for frag_smiles, atom_indices_list in frags.items():
                if len(frag_smiles) < 3:
                    continue
                key = (frag_smiles, col)
                frags_importances.setdefault(key, [])

                for atom_indices in atom_indices_list:
                    global_indices = [num_atoms + i for i in atom_indices]
                    # Sum over atoms and features — signed because IG can be negative
                    frag_score = float(np.sum(node_mask[global_indices]))
                    frags_importances[key].append(frag_score)

            num_atoms += mol_num_atoms

    return frags_importances


# ---------------------------------------------------------------------------
# Integrated Gradients core
# ---------------------------------------------------------------------------


def _integrated_gradients(
    model: nn.Module, data: Data, device: torch.device, n_steps: int = 50
) -> np.ndarray:
    """Compute Integrated Gradients for node features of a single graph.

    Uses a straight-line path from an all-zeros baseline to the actual
    node-feature matrix.  The integral is approximated by a Riemann sum
    with *n_steps* equally-spaced interpolation points.

    ``torch.autograd.grad`` is used rather than ``backward()`` to avoid
    accumulating spurious gradients in the model's parameter buffers.

    Parameters
    ----------
    model:
        Trained GNN in eval mode.
    data:
        Single (unbatched) reaction graph.
    device:
        Torch device to run inference on.
    n_steps:
        Number of Riemann sum steps (higher → more accurate, slower).

    Returns
    -------
    np.ndarray, shape ``[num_nodes, num_features]``
        Signed attributions.  Positive values indicate that the feature
        pushes the prediction above the baseline; negative values push it
        below.
    """
    model.eval()

    x = data.x.float().to(device)  # [N, F]
    edge_index = data.edge_index.to(device)
    batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)

    # Preserve edge features so GAT / GIN architectures receive them correctly
    edge_attr = getattr(data, "edge_attr", None)
    if edge_attr is not None:
        edge_attr = edge_attr.float().to(device)

    # Preserve graph-level reaction features — these are fixed (not interpolated)
    # because IG attributes predictions to *node* features, not global conditions.
    rxn_feats = getattr(data, "reaction_features", None)
    if rxn_feats is not None:
        rxn_feats = rxn_feats.float().to(device)

    baseline = torch.zeros_like(x)  # all-zeros = neutral reference
    delta = x - baseline  # [N, F]

    grads_sum = torch.zeros_like(x)

    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        x_interp = (baseline + alpha * delta).requires_grad_(True)

        interp_data = Data(
            x=x_interp, edge_index=edge_index, edge_attr=edge_attr, batch=batch
        )
        if rxn_feats is not None:
            interp_data.reaction_features = rxn_feats
        out = model(interp_data)

        # Gradient of scalar output w.r.t. x_interp only — model params untouched
        (grad,) = torch.autograd.grad(out.sum(), x_interp)
        grads_sum = grads_sum + grad.detach()

    avg_grads = grads_sum / n_steps
    ig = delta * avg_grads  # [N, F], signed

    return ig.cpu().numpy()


# ---------------------------------------------------------------------------
# Main explainability class
# ---------------------------------------------------------------------------


class GNNExplainer:
    """Signed node-attribute explainer for reaction GNNs.

    Computes **Integrated Gradients** (Sundararajan et al., 2017) for every
    reaction graph in a dataset, then aggregates the signed per-atom
    importances to molecular fragment level.

    Unlike mask-optimisation approaches (PyG's GNNExplainer), IG attributions
    can be negative, giving the same directional information as SHAP values
    for tabular models.

    Parameters
    ----------
    model:
        A trained :class:`~asymbench.gnn.model.ReactionGCN`.
    n_steps:
        Number of Riemann-sum steps for the IG approximation (default 50;
        use ≥ 100 for publication-quality results).
    fragmentation:
        ``"brics"`` or ``"murcko_scaffold"``.
    """

    def __init__(
        self, model: nn.Module, n_steps: int = 50, fragmentation: str = "brics"
    ) -> None:
        self.model = model
        self.n_steps = n_steps
        self.fragmentation = FragmentationMethod(fragmentation)

    def fit(self) -> "GNNExplainer":
        """No-op — IG is computed on-the-fly.  Kept for API consistency."""
        return self

    def explain_dataset(self, dataset, device) -> Dict:
        """Compute signed IG node masks for every graph in *dataset*.

        Returns
        -------
        dict
            ``{data.idx: np.ndarray shape [num_nodes, num_features]}``
        """
        self.model.eval()
        node_masks: Dict = {}

        for data in dataset._data:
            try:
                node_masks[data.idx] = _integrated_gradients(
                    self.model, data, device, self.n_steps
                )
            except Exception as exc:
                print(f"Warning: IG failed for graph idx={data.idx}: {exc}")

        return node_masks

    def save(
        self,
        node_masks: Dict,
        frag_importances: Dict[Tuple[str, str], List[float]],
        outdir: Path,
        top_k: int = 5,
    ) -> None:
        """Persist node masks and fragment importances to *outdir*.

        Writes
        ------
        ``node_masks.npz``
            Compressed numpy archive keyed by reaction index.
        ``fragment_importances.csv``
            Long-format CSV with one row per individual IG score.
            Columns: fragment, source, importance.
            Sorted by fragment then source for easy beeswarm plot construction.
        ``fragment_beeswarm.png``
            SHAP-style horizontal beeswarm plot of the top-*k*
            ``(fragment, source)`` pairs ranked by mean absolute IG importance.
            Red dots push the prediction up; blue dots push it down.
        """
        from asymbench.visualization.gnn_beeswarm import (
            plot_gnn_fragment_beeswarm,
        )

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        np.savez(
            outdir / "node_masks.npz",
            **{str(k): v for k, v in node_masks.items()},
        )

        if frag_importances:
            rows = [
                {
                    "fragment": frag_smi,
                    "source": source_col,
                    "importance": score,
                }
                for (frag_smi, source_col), scores in frag_importances.items()
                for score in scores
            ]
            df_frags = (
                pd.DataFrame(rows)
                .sort_values(["fragment", "source"])
                .reset_index(drop=True)
            )
            df_frags.to_csv(outdir / "fragment_importances.csv", index=False)

            plot_gnn_fragment_beeswarm(
                df_frags, outpath=outdir / "fragment_beeswarm.png", top_k=top_k
            )
