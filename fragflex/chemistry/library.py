from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from rdkit import Chem


@dataclass
class FragmentLibrary:
    """FragDiffusion-compatible fragment vocabulary."""

    id_to_smiles: Dict[int, str]

    @classmethod
    def from_csv(cls, path: str | Path) -> "FragmentLibrary":
        df = pd.read_csv(path)
        required = {"fragment_index", "fragment_name"}
        if not required.issubset(df.columns):
            raise ValueError(f"{path} lacks columns {required}")
        return cls({int(r.fragment_index): str(r.fragment_name) for _, r in df.iterrows()})

    @property
    def size(self) -> int:
        return max(self.id_to_smiles) + 1 if self.id_to_smiles else 0

    def mol(self, frag_id: int) -> Chem.Mol:
        smi = self.id_to_smiles[int(frag_id)]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"cannot parse fragment {frag_id}: {smi}")
        return mol


class AttachmentLibrary:
    """Pair-dependent attachment lookup used by FragDiffusion.

    ``edge_mode`` is zero-based, matching ``fragment_edge_index.csv``'s
    ``edge_id``. The generative adjacency reserves 0 for no-edge and stores a
    real attachment as ``edge_mode + 1``.

    In addition to the original ``lookup`` API, this class exposes all valid
    modes for a fragment pair. The constrained assembler uses that information
    to try the predicted mode first and, optionally, repair only the attachment
    *mode* while keeping the generated fragment pair fixed.
    """

    def __init__(self, fragment_library: FragmentLibrary, table: pd.DataFrame):
        self.fragments = fragment_library
        self._table: Dict[Tuple[int, int], Dict[int, Tuple[int, int]]] = {}
        required = {"fragment_index_1", "fragment_index_2", "atom_idx_1", "atom_idx_2", "edge_id"}
        if not required.issubset(table.columns):
            raise ValueError(f"attachment table lacks columns {required}")
        for _, r in table.iterrows():
            key = (int(r.fragment_index_1), int(r.fragment_index_2))
            self._table.setdefault(key, {})[int(r.edge_id)] = (int(r.atom_idx_1), int(r.atom_idx_2))

    @classmethod
    def from_csv(
        cls,
        fragment_library: FragmentLibrary,
        path: str | Path,
    ) -> "AttachmentLibrary":
        return cls(fragment_library, pd.read_csv(path))

    def _canonical_pair(self, frag_a: int, frag_b: int) -> tuple[Tuple[int, int], bool]:
        names = [
            self.fragments.id_to_smiles.get(int(frag_a)),
            self.fragments.id_to_smiles.get(int(frag_b)),
        ]
        if any(x is None for x in names):
            return (int(frag_a), int(frag_b)), False
        swapped = names != sorted(names)
        key = (int(frag_b), int(frag_a)) if swapped else (int(frag_a), int(frag_b))
        return key, swapped

    def lookup(self, frag_a: int, frag_b: int, edge_mode: int) -> Optional[Tuple[int, int]]:
        key, swapped = self._canonical_pair(frag_a, frag_b)
        pair = self._table.get(key, {}).get(int(edge_mode))
        if pair is None:
            return None
        return (pair[1], pair[0]) if swapped else pair

    def available_modes(self, frag_a: int, frag_b: int) -> list[int]:
        """Return valid zero-based edge modes for the unordered fragment pair."""

        key, _ = self._canonical_pair(frag_a, frag_b)
        return sorted(self._table.get(key, {}).keys())

    def available_attachments(self, frag_a: int, frag_b: int) -> Dict[int, Tuple[int, int]]:
        """Return ``mode -> oriented (atom_a, atom_b)`` for a fragment pair."""

        return {
            mode: pair
            for mode in self.available_modes(frag_a, frag_b)
            if (pair := self.lookup(frag_a, frag_b, mode)) is not None
        }
