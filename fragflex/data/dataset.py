from __future__ import annotations

import gzip
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from fragflex.chemistry.library import AttachmentLibrary
from fragflex.graph import FragmentGraph


class FragmentGraphDataset(Dataset):
    def __init__(self, graphs: Sequence[FragmentGraph]):
        self.graphs = list(graphs)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> FragmentGraph:
        return self.graphs[idx]

    def save(self, path: str | Path) -> None:
        # site_logits are inference-time neural scores and can be very large;
        # only exact site targets are persisted.
        payload = [
            {
                "x": g.x.cpu(),
                "e": g.e.cpu(),
                "smiles": g.smiles,
                "sites": None if g.sites is None else g.sites.cpu(),
            }
            for g in self.graphs
        ]
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "FragmentGraphDataset":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            [
                FragmentGraph(
                    x=p["x"],
                    e=p["e"],
                    smiles=p.get("smiles"),
                    sites=p.get("sites"),
                )
                for p in payload
            ]
        )


def _ids_from_x(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.long()
    if x.ndim == 2:
        return x.argmax(dim=-1).long()
    raise ValueError(f"unsupported x shape {tuple(x.shape)}")


def _edge_mode_from_attr(attr: torch.Tensor, attr_has_no_edge: bool) -> torch.Tensor:
    if attr.ndim == 1:
        ids = attr.long()
    elif attr.ndim == 2:
        ids = attr.argmax(dim=-1).long()
    else:
        raise ValueError(f"unsupported edge_attr shape {tuple(attr.shape)}")
    return ids if attr_has_no_edge else ids + 1


def _torch_load_fragdiffusion(path: str | Path):
    """Load a FragDiffusion torch artifact, including the released .pt.gz file."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return torch.load(f, map_location="cpu", weights_only=False)
    return torch.load(path, map_location="cpu", weights_only=False)


def load_fragdiffusion_pt(
    path: str | Path,
    *,
    attr_has_no_edge: bool = False,
    smiles: Optional[Sequence[str]] = None,
) -> FragmentGraphDataset:
    """Load FragDiffusion's preprocessed PyG graph list into dense graphs.

    This is the legacy lookup-mode representation. FragDiffusion's raw stored
    fragment edges normally do not include the explicit no-edge class;
    ``FragDataset.__getitem__`` adds it at runtime. Therefore the default maps
    stored edge mode 0..K-1 -> dense class 1..K.
    """

    raw = _torch_load_fragdiffusion(path)
    graphs: List[FragmentGraph] = []
    for idx, data in enumerate(raw):
        x = _ids_from_x(data.x)
        n = int(x.shape[0])
        e = torch.zeros((n, n), dtype=torch.long)
        edge_ids = _edge_mode_from_attr(data.edge_attr, attr_has_no_edge)
        edge_index = data.edge_index.long()
        for k in range(edge_index.shape[1]):
            i = int(edge_index[0, k])
            j = int(edge_index[1, k])
            if i == j:
                continue
            cls = int(edge_ids[k])
            e[i, j] = cls
            e[j, i] = cls
        smi = None if smiles is None else smiles[idx]
        graphs.append(FragmentGraph(x=x, e=e, smiles=smi))
    return FragmentGraphDataset(graphs)


def load_fragdiffusion_pt_neural_sites(
    path: str | Path,
    *,
    attachments: AttachmentLibrary,
    attr_has_no_edge: bool = False,
    smiles: Optional[Sequence[str]] = None,
    strict: bool = True,
) -> FragmentGraphDataset:
    """Convert FragDiffusion data into lookup-free *training targets*.

    The original attachment table is consulted **only once during preprocessing**
    to recover the ground-truth atom endpoints encoded by FragDiffusion's edge
    mode. The saved dataset then contains:

      * ``e``: binary fragment topology (0=no edge, 1=linked), and
      * ``sites[i,j]``: local atom index in fragment ``i`` used to connect to j.

    Sampling/assembly from a model trained on this representation does not need
    ``fragment_edge_index.csv``. This separates the historical data encoding
    from the new learned attachment decoder.
    """

    raw = _torch_load_fragdiffusion(path)
    graphs: List[FragmentGraph] = []
    unresolved = 0

    for idx, data in enumerate(raw):
        x = _ids_from_x(data.x)
        n = int(x.shape[0])
        e = torch.zeros((n, n), dtype=torch.long)
        sites = torch.full((n, n), -1, dtype=torch.long)
        edge_ids = _edge_mode_from_attr(data.edge_attr, attr_has_no_edge)
        edge_index = data.edge_index.long()

        # Stored PyG graphs may contain both directions. Re-writing the same
        # symmetric coarse edge is harmless; endpoint orientation is resolved
        # with the current ordered fragment IDs through AttachmentLibrary.lookup.
        for k in range(edge_index.shape[1]):
            i = int(edge_index[0, k])
            j = int(edge_index[1, k])
            if i == j:
                continue
            dense_cls = int(edge_ids[k])
            if dense_cls == 0:
                continue
            mode = dense_cls - 1
            fi, fj = int(x[i]), int(x[j])
            pair = attachments.lookup(fi, fj, mode)
            if pair is None:
                unresolved += 1
                if strict:
                    raise ValueError(
                        "could not recover attachment target for "
                        f"graph={idx}, edge=({i},{j}), fragments=({fi},{fj}), mode={mode}"
                    )
                continue
            ai, aj = pair
            e[i, j] = e[j, i] = 1
            sites[i, j] = int(ai)
            sites[j, i] = int(aj)

        smi = None if smiles is None else smiles[idx]
        graphs.append(FragmentGraph(x=x, e=e, smiles=smi, sites=sites))

    if unresolved and not strict:
        print(f"warning: skipped {unresolved} unresolved attachment targets")
    return FragmentGraphDataset(graphs)
