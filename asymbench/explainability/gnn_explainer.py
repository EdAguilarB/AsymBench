"""GNN explainability via CaptumExplainer + molecular fragment attribution.

Workflow
--------
1. A trained :class:`~asymbench.gnn.model.ReactionGCN` is wrapped so its
   forward signature matches what PyG's CaptumExplainer expects.
2. For every reaction graph, the explainer computes a *node mask* — a
   per-atom, per-feature importance tensor.
3. The node masks are aggregated to fragment-level scores by fragmenting
   each component molecule with BRICS or Murcko scaffold decomposition and
   summing the atom importances that fall inside each fragment.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
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
    """BRICS decomposition with atom-index mapping.

    Returns
    -------
    dict
        ``{canonical_fragment_smiles: [[atom_indices_occurrence_0], ...]}``
    """
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
    """Murcko scaffold with atom-index mapping.

    Returns
    -------
    dict
        ``{scaffold_smiles: [[atom_indices]]}``  (single occurrence)
    """
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
    smiles: str,
    fragmentation_method: FragmentationMethod,
) -> Dict[str, List[List[int]]]:
    """Fragment *smiles* and return the atom indices of every fragment occurrence.

    Parameters
    ----------
    smiles:
        SMILES of the molecule to fragment.
    fragmentation_method:
        Which fragmentation strategy to apply.

    Returns
    -------
    dict
        ``{fragment_smiles: [[atom_idx, ...], ...]}`` — each inner list is
        one occurrence of the fragment in the parent molecule.
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
) -> Dict[str, List[float]]:
    """Aggregate per-atom node masks to fragment-level importance scores.

    Each reaction graph is a concatenation of molecular graphs.  The function
    iterates over the per-molecule SMILES stored in ``data.mol_smiles``,
    fragments each molecule, and sums the node-mask values over the atoms
    belonging to each fragment occurrence.

    Parameters
    ----------
    data_list:
        List of PyG :class:`Data` objects (``dataset._data``).  Each must
        have a ``mol_smiles`` attribute — a list of ``(column_name, smiles)``
        pairs in the same order as the molecules were concatenated.
    node_masks:
        ``{data.idx: np.ndarray shape [num_nodes, num_features]}``
    fragmentation_method:
        How to fragment each molecule.

    Returns
    -------
    dict
        ``{fragment_smiles: [importance_score, ...]}`` — one score per
        fragment occurrence across the whole dataset.
    """
    frags_importances: Dict[str, List[float]] = {}

    for data in data_list:
        mol_idx = data.idx
        if mol_idx not in node_masks:
            continue

        node_mask = node_masks[mol_idx]  # [num_nodes, num_features]
        num_atoms = 0

        for _col, smiles in data.mol_smiles:
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
                frags_importances.setdefault(frag_smiles, [])

                for atom_indices in atom_indices_list:
                    global_indices = [num_atoms + i for i in atom_indices]
                    frag_score = float(np.sum(node_mask[global_indices]))
                    frags_importances[frag_smiles].append(frag_score)

            num_atoms += mol_num_atoms

    return frags_importances


# ---------------------------------------------------------------------------
# Model wrapper required by CaptumExplainer
# ---------------------------------------------------------------------------

class _ExplainerModelWrapper(nn.Module):
    """Adapts :class:`ReactionGCN` to the ``(x, edge_index, **kwargs)``
    signature expected by PyG's :class:`CaptumExplainer`.

    When called on a single (unbatched) graph, a batch tensor of all zeros
    is synthesised so that the global pooling layer works correctly.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs):
        batch = kwargs.get("batch")
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        data = Data(x=x, edge_index=edge_index, batch=batch)
        out = self.model(data)
        return out.view(-1)  # [batch_size]


# ---------------------------------------------------------------------------
# Main explainability class
# ---------------------------------------------------------------------------

class GNNExplainer:
    """Node-attribute explainer for reaction GNNs.

    Uses PyG's :class:`~torch_geometric.explain.Explainer` with a
    configurable Captum attribution method (default:
    ``ShapleyValueSampling``) to produce per-atom importance scores, then
    aggregates them to molecular fragment level.

    Parameters
    ----------
    model:
        A trained :class:`~asymbench.gnn.model.ReactionGCN`.
    attribution_method:
        Name of the ``captum.attr`` class to use, e.g.
        ``"ShapleyValueSampling"``, ``"IntegratedGradients"``,
        ``"Saliency"``.
    fragmentation:
        ``"brics"`` (BRICS decomposition) or ``"murcko_scaffold"``
        (Murcko scaffold extraction).
    """

    def __init__(
        self,
        model: nn.Module,
        attribution_method: str = "ShapleyValueSampling",
        fragmentation: str = "brics",
    ) -> None:
        self.model = model
        self.attribution_method = attribution_method
        self.fragmentation = FragmentationMethod(fragmentation)
        self._explainer = None

    def fit(self) -> "GNNExplainer":
        """Build the PyG Explainer.  Call once after model training."""
        import captum.attr
        from torch_geometric.explain import CaptumExplainer, Explainer, ModelConfig

        attr_cls = getattr(captum.attr, self.attribution_method, None)
        if attr_cls is None:
            raise ValueError(
                f"captum.attr has no attribute {self.attribution_method!r}. "
                f"Common choices: ShapleyValueSampling, IntegratedGradients, Saliency."
            )

        algorithm = CaptumExplainer(attribution_method=attr_cls)
        wrapped = _ExplainerModelWrapper(self.model)

        self._explainer = Explainer(
            model=wrapped,
            algorithm=algorithm,
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type=None,
            model_config=ModelConfig(
                mode="regression",
                task_level="graph",
                return_type="raw",
            ),
        )
        return self

    def explain_dataset(self, dataset, device) -> Dict:
        """Compute node masks for every graph in *dataset*.

        Parameters
        ----------
        dataset:
            A :class:`~asymbench.gnn.dataset.ReactionGraphDataset`.
        device:
            Torch device to run inference on.

        Returns
        -------
        dict
            ``{data.idx: np.ndarray shape [num_nodes, num_features]}``
        """
        if self._explainer is None:
            raise RuntimeError("Call GNNExplainer.fit() before explain_dataset().")

        self.model.eval()
        node_masks: Dict = {}

        for data in dataset._data:
            data = data.to(device)
            try:
                explanation = self._explainer(
                    x=data.x,
                    edge_index=data.edge_index,
                    batch=None,
                )
                node_masks[data.idx] = (
                    explanation.node_mask.detach().cpu().numpy()
                )
            except Exception as exc:
                print(
                    f"Warning: explanation failed for graph idx={data.idx}: {exc}"
                )

        return node_masks

    def save(
        self,
        node_masks: Dict,
        frag_importances: Dict[str, List[float]],
        outdir: Path,
    ) -> None:
        """Persist node masks and fragment importances to *outdir*.

        Writes
        ------
        ``node_masks.npz``
            Compressed numpy archive keyed by reaction index.
        ``fragment_importances.csv``
            Columns: fragment, mean_importance, std_importance, count.
        """
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        np.savez(
            outdir / "node_masks.npz",
            **{str(k): v for k, v in node_masks.items()},
        )

        if frag_importances:
            rows = [
                {
                    "fragment": frag,
                    "mean_importance": float(np.mean(scores)),
                    "std_importance": float(np.std(scores)),
                    "count": len(scores),
                }
                for frag, scores in frag_importances.items()
            ]
            (
                pd.DataFrame(rows)
                .sort_values("mean_importance", ascending=False)
                .to_csv(outdir / "fragment_importances.csv", index=False)
            )
