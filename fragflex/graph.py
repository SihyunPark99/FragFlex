from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch


@dataclass
class FragmentGraph:
    """A clean/generated fragment graph.

    ``x`` contains fragment IDs.

    ``e`` is a dense symmetric coarse adjacency matrix. In the legacy
    FragDiffusion-compatible representation, class 0 is no-edge and classes
    1..K are pair-dependent attachment modes. In the neural-site representation
    introduced for FragFlex, class 0 is no-edge and class 1 is a fragment bond;
    the atom-level endpoints are carried separately by ``sites``/``site_logits``.

    ``sites[i, j]`` is the *local atom index in fragment i* used by the directed
    endpoint i->j. Thus for a real coarse edge (i,j), sites[i,j] and sites[j,i]
    jointly specify the atom pair. -1 denotes no supervised/predicted endpoint.

    ``site_logits`` is optional and is normally present only on generated graphs.
    It has shape [N,N,M], where M is the maximum number of atoms in any fragment,
    and stores neural attachment-site scores used by the neural assembler. It is
    deliberately not persisted by :class:`FragmentGraphDataset`.
    """

    x: torch.Tensor
    e: torch.Tensor
    smiles: Optional[str] = None
    sites: Optional[torch.Tensor] = None
    site_logits: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        self.x = self.x.long()
        self.e = self.e.long()
        if self.sites is not None:
            self.sites = self.sites.long()
        self.validate()

    @property
    def num_nodes(self) -> int:
        return int(self.x.numel())

    def validate(self) -> None:
        if self.x.ndim != 1:
            raise ValueError(f"x must be [N], got {tuple(self.x.shape)}")
        n = self.x.shape[0]
        if self.e.shape != (n, n):
            raise ValueError(f"e must be [N,N], got {tuple(self.e.shape)} for N={n}")
        if not torch.equal(self.e, self.e.T):
            raise ValueError("e must be symmetric")
        if torch.any(torch.diag(self.e) != 0):
            raise ValueError("e diagonal must be no-edge (0)")
        if self.sites is not None:
            if self.sites.shape != (n, n):
                raise ValueError(
                    f"sites must be [N,N], got {tuple(self.sites.shape)} for N={n}"
                )
            if torch.any(torch.diag(self.sites) != -1):
                raise ValueError("sites diagonal must be -1")
        if self.site_logits is not None:
            if self.site_logits.ndim != 3 or self.site_logits.shape[:2] != (n, n):
                raise ValueError(
                    "site_logits must be [N,N,M], got "
                    f"{tuple(self.site_logits.shape)} for N={n}"
                )


@dataclass
class DenseGraphBatch:
    """Dense variable-size graph batch used by the diffusion/model code."""

    x: torch.Tensor                 # [B,N] integer state IDs
    e: torch.Tensor                 # [B,N,N] integer state IDs
    node_mask: torch.Tensor         # [B,N] bool
    root_mask: torch.Tensor         # [B,N] bool; exactly one valid root per graph
    t: torch.Tensor                 # [B] integer timestep
    delt_mask: Optional[torch.Tensor] = None  # [B,N] bool, current DEL* nodes

    def to(self, device: torch.device | str) -> "DenseGraphBatch":
        return DenseGraphBatch(
            x=self.x.to(device),
            e=self.e.to(device),
            node_mask=self.node_mask.to(device),
            root_mask=self.root_mask.to(device),
            t=self.t.to(device),
            delt_mask=None if self.delt_mask is None else self.delt_mask.to(device),
        )

    @property
    def batch_size(self) -> int:
        return self.x.shape[0]

    @property
    def max_nodes(self) -> int:
        return self.x.shape[1]

    def clone(self) -> "DenseGraphBatch":
        return DenseGraphBatch(
            x=self.x.clone(),
            e=self.e.clone(),
            node_mask=self.node_mask.clone(),
            root_mask=self.root_mask.clone(),
            t=self.t.clone(),
            delt_mask=None if self.delt_mask is None else self.delt_mask.clone(),
        )


def collate_fragment_graphs(graphs: Iterable[FragmentGraph]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad clean graphs to dense integer tensors.

    Returns x [B,N], e [B,N,N], node_mask [B,N]. Attachment-site targets are
    intentionally handled by the diffusion corruption code because structural
    deletion changes their node indexing together with x/e.
    """

    graphs = list(graphs)
    if not graphs:
        raise ValueError("cannot collate an empty graph list")
    b = len(graphs)
    nmax = max(g.num_nodes for g in graphs)
    x = torch.zeros((b, nmax), dtype=torch.long)
    e = torch.zeros((b, nmax, nmax), dtype=torch.long)
    mask = torch.zeros((b, nmax), dtype=torch.bool)
    for i, g in enumerate(graphs):
        n = g.num_nodes
        x[i, :n] = g.x
        e[i, :n, :n] = g.e
        mask[i, :n] = True
    return x, e, mask


def pack_variable_graphs(
    xs: List[torch.Tensor],
    es: List[torch.Tensor],
    roots: List[torch.Tensor],
    delts: List[torch.Tensor],
    t: torch.Tensor,
    x_pad: int = 0,
    e_pad: int = 0,
) -> DenseGraphBatch:
    """Pack already-filtered per-sample graph states into a padded batch."""

    if not xs:
        raise ValueError("empty state list")
    b = len(xs)
    nmax = max(int(x.shape[0]) for x in xs)
    device = xs[0].device
    x = torch.full((b, nmax), x_pad, dtype=torch.long, device=device)
    e = torch.full((b, nmax, nmax), e_pad, dtype=torch.long, device=device)
    node_mask = torch.zeros((b, nmax), dtype=torch.bool, device=device)
    root_mask = torch.zeros((b, nmax), dtype=torch.bool, device=device)
    delt_mask = torch.zeros((b, nmax), dtype=torch.bool, device=device)
    for i, (xi, ei, ri, di) in enumerate(zip(xs, es, roots, delts)):
        n = xi.shape[0]
        x[i, :n] = xi
        e[i, :n, :n] = ei
        node_mask[i, :n] = True
        root_mask[i, :n] = ri
        delt_mask[i, :n] = di
    return DenseGraphBatch(x=x, e=e, node_mask=node_mask, root_mask=root_mask, t=t, delt_mask=delt_mask)
