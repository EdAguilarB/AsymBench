from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from rdkit import Chem
import torch

from asymbench.representations.base_featurizer import BaseSmilesFeaturizer


@dataclass
class HFTransformerFeaturizer(BaseSmilesFeaturizer):
    """
    Hugging Face transformer featurizer for SMILES-based molecular embeddings.

    Expected rep_params
    -------------------
    model_name : str
        HF model checkpoint, e.g. "DeepChem/ChemBERTa-77M-MLM" or "laituan245/molt5-base".
    model_type : str
        One of {"chemberta", "molt5"}. Determines which model class to load.
    pooling : str
        One of {"mean", "cls"}. Default: "mean".
    device : str
        e.g. "cpu", "cuda". Default: auto-detected.
    max_length : int
        Tokenizer truncation length. Default: 512.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        self.model_type: str = (
            str(self.rep_params.get("model_type", "chemberta")).strip().lower()
        )
        if self.model_type not in {"chemberta", "molt5"}:
            raise ValueError(
                "model_type must be one of {'chemberta', 'molt5'}"
            )

        _defaults = {
            "chemberta": "DeepChem/ChemBERTa-77M-MLM",
            "molt5": "laituan245/molt5-base",
        }
        self.model_name: str = str(
            self.rep_params.get("model_name", _defaults[self.model_type])
        )

        self.pooling: str = (
            str(self.rep_params.get("pooling", "mean")).strip().lower()
        )
        if self.pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be one of {'mean', 'cls'}")

        self.max_length: int = int(self.rep_params.get("max_length", 512))

        # Device is resolved lazily so no CUDA context is created in the parent process
        self._requested_device: Optional[str] = self.rep_params.get(
            "device", None
        )

        # Model and tokenizer are loaded lazily (fork-safe)
        self._tokenizer = None
        self._model = None
        self._feature_dim: Optional[int] = None

    # ------------------------------------------------------------------ #
    #  Lazy loading                                                         #
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        device = self._requested_device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._device = device

        if self.model_type == "chemberta":
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._feature_dim = int(self._model.config.hidden_size)

        elif self.model_type == "molt5":
            from transformers import AutoTokenizer, T5EncoderModel

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = T5EncoderModel.from_pretrained(self.model_name)
            self._feature_dim = int(self._model.config.d_model)

        self._model.eval()
        self._model.to(self._device)
        torch.set_num_threads(1)

    # ------------------------------------------------------------------ #
    #  BaseSmilesFeaturizer interface                                       #
    # ------------------------------------------------------------------ #

    @property
    def feature_dim_per_mol(self) -> int:
        self._ensure_loaded()
        return self._feature_dim

    def featurize_mol(self, mol: Chem.Mol) -> np.ndarray:
        self._ensure_loaded()
        smiles = Chem.MolToSmiles(mol, canonical=True)

        inputs = self._tokenizer(
            smiles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
        last_hidden_state = outputs.last_hidden_state  # [1, seq_len, hidden]

        if self.pooling == "mean":
            emb = self._mean_pool(last_hidden_state, inputs["attention_mask"])
        else:  # cls
            emb = last_hidden_state[:, 0, :]

        return emb.squeeze(0).cpu().numpy().astype(float)

    def feature_names_per_mol(self) -> List[str]:
        dim = self.feature_dim_per_mol
        width = len(str(dim - 1))
        return [f"{self.model_type}_emb_{i:0{width}d}" for i in range(dim)]

    # ------------------------------------------------------------------ #
    #  Pickling — drop model so workers always reload in their own process  #
    # ------------------------------------------------------------------ #

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_model"] = None
        state["_tokenizer"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _mean_pool(
        last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = (
            attention_mask.unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts
