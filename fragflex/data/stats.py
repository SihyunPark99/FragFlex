from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import torch
from rdkit import Chem

from fragflex.chemistry.library import FragmentLibrary
from fragflex.graph import FragmentGraph


def _atom_label(atom: Chem.Atom) -> str:
    symbol = atom.GetSymbol()
    charge = atom.GetFormalCharge()
    if charge == 0:
        return symbol
    return f"{symbol}{charge:+d}"


@dataclass
class FragFlexStats:
    num_fragments: int
    num_atom_latents: int
    num_edge_types: int
    fragment_marginal: torch.Tensor
    atom_marginal: torch.Tensor
    edge_marginal: torch.Tensor
    atom_labels: List[str]

    # New metadata for neural attachment-site prediction. Defaults keep old
    # datasets/checkpoints loadable.
    assembly_mode: str = "lookup"
    max_fragment_atoms: int = 0
    fragment_atom_counts: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.long)
    )

    @property
    def base_node_states(self) -> int:
        return self.num_fragments + self.num_atom_latents

    @property
    def node_del_id(self) -> int:
        return self.base_node_states

    @property
    def node_delt_id(self) -> int:
        return self.base_node_states + 1

    @property
    def edge_del_id(self) -> int:
        return self.num_edge_types

    @property
    def edge_delt_id(self) -> int:
        return self.num_edge_types + 1

    @property
    def uses_neural_sites(self) -> bool:
        return self.assembly_mode == "neural_sites"

    def state_dict(self) -> dict:
        return {
            "num_fragments": self.num_fragments,
            "num_atom_latents": self.num_atom_latents,
            "num_edge_types": self.num_edge_types,
            "fragment_marginal": self.fragment_marginal,
            "atom_marginal": self.atom_marginal,
            "edge_marginal": self.edge_marginal,
            "atom_labels": self.atom_labels,
            "assembly_mode": self.assembly_mode,
            "max_fragment_atoms": self.max_fragment_atoms,
            "fragment_atom_counts": self.fragment_atom_counts,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "FragFlexStats":
        # Backward compatibility with pre-neural-assembly checkpoints.
        d = dict(d)
        d.setdefault("assembly_mode", "lookup")
        d.setdefault("max_fragment_atoms", 0)
        d.setdefault("fragment_atom_counts", torch.zeros(0, dtype=torch.long))
        return cls(**d)


def compute_stats(
    graphs: Iterable[FragmentGraph],
    fragments: FragmentLibrary,
    num_edge_types: int | None = None,
    *,
    assembly_mode: str = "lookup",
) -> FragFlexStats:
    """Compute fragment/edge marginals and the one-atom terminal prior."""

    if assembly_mode not in {"lookup", "neural_sites"}:
        raise ValueError(f"unknown assembly_mode={assembly_mode!r}")

    graphs = list(graphs)
    if not graphs:
        raise ValueError("empty dataset")
    f = fragments.size
    frag_counts = torch.zeros(f, dtype=torch.float64)
    max_edge = 0
    edge_counts_dynamic: Dict[int, float] = {}

    for g in graphs:
        frag_counts.scatter_add_(0, g.x, torch.ones_like(g.x, dtype=torch.float64))
        if assembly_mode == "neural_sites" and g.sites is None:
            raise ValueError("neural_sites dataset contains a graph without site targets")
        n = g.num_nodes
        for i in range(n):
            for j in range(i + 1, n):
                cls = int(g.e[i, j])
                max_edge = max(max_edge, cls)
                edge_counts_dynamic[cls] = edge_counts_dynamic.get(cls, 0.0) + 1.0

    if num_edge_types is None:
        num_edge_types = max_edge + 1
    edge_counts = torch.zeros(num_edge_types, dtype=torch.float64)
    for k, v in edge_counts_dynamic.items():
        if k < num_edge_types:
            edge_counts[k] = v
    edge_counts += 1e-8
    fragment_marginal = (frag_counts + 1e-8) / (frag_counts.sum() + 1e-8 * f)
    edge_marginal = edge_counts / edge_counts.sum()

    atom_counts: Dict[str, float] = {}
    fragment_atom_counts = torch.zeros(f, dtype=torch.long)
    for frag_id in range(f):
        if frag_id not in fragments.id_to_smiles:
            continue
        mol = Chem.MolFromSmiles(fragments.id_to_smiles[frag_id])
        if mol is None:
            continue
        fragment_atom_counts[frag_id] = mol.GetNumAtoms()
        count = frag_counts[frag_id].item()
        if count <= 0:
            continue
        for atom in mol.GetAtoms():
            label = _atom_label(atom)
            atom_counts[label] = atom_counts.get(label, 0.0) + count

    atom_labels = sorted(atom_counts)
    if not atom_labels:
        raise ValueError("could not derive any atom latent types from fragment library")
    ac = torch.tensor([atom_counts[k] for k in atom_labels], dtype=torch.float64)
    atom_marginal = ac / ac.sum()

    return FragFlexStats(
        num_fragments=f,
        num_atom_latents=len(atom_labels),
        num_edge_types=num_edge_types,
        fragment_marginal=fragment_marginal.float(),
        atom_marginal=atom_marginal.float(),
        edge_marginal=edge_marginal.float(),
        atom_labels=atom_labels,
        assembly_mode=assembly_mode,
        max_fragment_atoms=int(fragment_atom_counts.max().item()) if f else 0,
        fragment_atom_counts=fragment_atom_counts,
    )
