"""
asymbench.gnn.architectures
============================
Architecture-specific GNN subclasses and a model factory.

Available architectures
-----------------------
``"gcn"``   :class:`ReactionGCN`   — GCN (Kipf & Welling, 2017)
``"gat"``   :class:`ReactionGAT`   — GATv2 with edge features (Brody et al., 2022)
``"gin"``   :class:`ReactionGIN`   — GIN-E with edge features (Hu et al., 2020)

Factory usage
-------------
::

    from asymbench.gnn.architectures import build_reaction_gnn

    model = build_reaction_gnn(
        architecture="gat",
        node_in_dim=33,
        hidden_dim=64,
        num_layers=3,
        num_heads=4,
    )
"""

from __future__ import annotations

from asymbench.gnn.architectures.gcn import ReactionGCN
from asymbench.gnn.architectures.gat import ReactionGAT
from asymbench.gnn.architectures.gin import ReactionGIN
from asymbench.gnn.base import BaseReactionGNN

_REGISTRY: dict[str, type[BaseReactionGNN]] = {
    ReactionGCN.ARCH_NAME: ReactionGCN,
    ReactionGAT.ARCH_NAME: ReactionGAT,
    ReactionGIN.ARCH_NAME: ReactionGIN,
}


def build_reaction_gnn(architecture: str = "gcn", **kwargs) -> BaseReactionGNN:
    """Instantiate a reaction GNN by name.

    Parameters
    ----------
    architecture:
        One of ``"gcn"``, ``"gat"``, ``"gin"``.
    **kwargs:
        Forwarded to the chosen class constructor.  Architecture-specific
        parameters (e.g. ``num_heads`` for GAT, ``improved`` for GCN,
        ``train_eps`` for GIN) are passed through; unknown keys are silently
        absorbed by each class's ``**kwargs``.

    Returns
    -------
    BaseReactionGNN
        A ready-to-train (but not yet ``.to(device)``'d) model instance.

    Raises
    ------
    ValueError
        If *architecture* is not recognised.

    Examples
    --------
    >>> model = build_reaction_gnn("gat", node_in_dim=33, num_heads=8)
    >>> model = build_reaction_gnn("gcn", node_in_dim=33, improved=True)
    >>> model = build_reaction_gnn("gin", node_in_dim=33, train_eps=True)
    """
    key = architecture.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown GNN architecture {architecture!r}. "
            f"Available: {list(_REGISTRY)}."
        )
    return _REGISTRY[key](**kwargs)


__all__ = [
    "BaseReactionGNN",
    "ReactionGCN",
    "ReactionGAT",
    "ReactionGIN",
    "build_reaction_gnn",
]
