from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from rdkit import Chem
import random
from .library import AttachmentLibrary, FragmentLibrary


@dataclass
class AssemblyResult:
    mol: Optional[Chem.Mol]
    skipped_edges: int
    attempted_edges: int
    connected: bool = False
    num_components: int = 0
    repaired_mode_edges: int = 0
    lookup_missing_edges: int = 0
    valence_rejected_options: int = 0
    duplicate_rejected_options: int = 0
    sanitize_rejected_candidates: int = 0
    selected_edges: int = 0
    failure_reason: Optional[str] = None
    selected_modes: Dict[Tuple[int, int], int] = field(default_factory=dict)

    # Diagnostics for the improved constrained decoder.
    generated_topology_connected: bool = False
    generated_topology_tree: bool = False
    forward_pruned_states: int = 0
    exact_search_used: bool = False
    exact_states_visited: int = 0
    search_method: str = ""

    # Neural-site assembler diagnostics (lookup table is not used).
    neural_site_repaired_edges: int = 0
    neural_site_candidate_pairs: int = 0

    # Hydrogen-aware neural assembly diagnostics. Fragment SMILES are capped
    # molecules, so attachment at e.g. aromatic [nH] must replace the explicit
    # hydrogen rather than add a fifth-valence bond on top of it.
    attachment_h_replacements: int = 0
    static_site_masked_atoms: int = 0
    beam_exhausted_edge: int = -1
    valid_unique_products_considered: int = 0


class FragDiffusionAssembler:
    """Original FragDiffusion-style pairwise lookup assembly.

    This is intentionally left as the baseline decoder for A/B comparison.
    """

    def __init__(self, fragments: FragmentLibrary, attachments: AttachmentLibrary):
        self.fragments = fragments
        self.attachments = attachments

    def assemble(self, frag_ids: torch.Tensor, adjacency: torch.Tensor, sanitize: bool = True) -> AssemblyResult:
        frag_ids = frag_ids.detach().cpu().long()
        adjacency = adjacency.detach().cpu().long()
        if frag_ids.ndim != 1 or adjacency.shape != (len(frag_ids), len(frag_ids)):
            raise ValueError("expected frag_ids [N] and adjacency [N,N]")

        mols: List[Chem.Mol] = [self.fragments.mol(int(i)) for i in frag_ids.tolist()]
        if not mols:
            return AssemblyResult(None, 0, 0, failure_reason="empty graph")

        combined = mols[0]
        for m in mols[1:]:
            combined = Chem.CombineMols(combined, m)
        rw = Chem.RWMol(combined)

        starts: List[int] = []
        cur = 0
        for m in mols:
            starts.append(cur)
            cur += m.GetNumAtoms()

        skipped = 0
        attempted = 0
        lookup_missing = 0
        selected: list[tuple[int, int]] = []
        selected_modes: Dict[Tuple[int, int], int] = {}
        n = len(frag_ids)
        for i in range(n):
            for j in range(i + 1, n):
                edge_cls = int(adjacency[i, j])
                if edge_cls <= 0:
                    continue
                attempted += 1
                mode = edge_cls - 1
                atom_pair = self.attachments.lookup(int(frag_ids[i]), int(frag_ids[j]), mode)
                if atom_pair is None:
                    skipped += 1
                    lookup_missing += 1
                    continue
                ai = starts[i] + atom_pair[0]
                aj = starts[j] + atom_pair[1]
                if rw.GetBondBetweenAtoms(ai, aj) is not None:
                    skipped += 1
                    continue
                try:
                    rw.AddBond(ai, aj, Chem.BondType.SINGLE)
                    selected.append((i, j))
                    selected_modes[(i, j)] = mode
                except Exception:
                    skipped += 1

        mol = rw.GetMol()
        connected, components = _fragment_connectivity(n, selected)
        generated_edges = [
            (i, j)
            for i in range(n)
            for j in range(i + 1, n)
            if int(adjacency[i, j]) > 0
        ]
        generated_connected, _ = _fragment_connectivity(n, generated_edges)
        generated_tree = generated_connected and len(generated_edges) == max(n - 1, 0)

        if sanitize:
            mol, reason = _sanitize_copy(mol)
            if mol is None:
                return AssemblyResult(
                    None,
                    skipped,
                    attempted,
                    connected=connected,
                    num_components=components,
                    lookup_missing_edges=lookup_missing,
                    selected_edges=len(selected),
                    failure_reason=reason,
                    selected_modes=selected_modes,
                    generated_topology_connected=generated_connected,
                    generated_topology_tree=generated_tree,
                    search_method="fragdiffusion",
                )
        return AssemblyResult(
            mol,
            skipped,
            attempted,
            connected=connected,
            num_components=components,
            lookup_missing_edges=lookup_missing,
            selected_edges=len(selected),
            selected_modes=selected_modes,
            generated_topology_connected=generated_connected,
            generated_topology_tree=generated_tree,
            search_method="fragdiffusion",
        )

    def to_smiles(self, frag_ids: torch.Tensor, adjacency: torch.Tensor) -> tuple[Optional[str], AssemblyResult]:
        result = self.assemble(frag_ids, adjacency, sanitize=True)
        if result.mol is None:
            return None, result
        return Chem.MolToSmiles(result.mol, isomericSmiles=True), result


@dataclass
class _EdgeSpec:
    i: int
    j: int
    predicted_mode: int
    options: list[tuple[int, int, int]]  # (mode, local_atom_i, local_atom_j)


@dataclass
class _BeamState:
    rw: Chem.RWMol
    selected_pairs: tuple[tuple[int, int], ...]
    selected_modes: Dict[Tuple[int, int], int]
    realized: int
    repaired: int
    skipped: int


@dataclass
class _SearchCounters:
    valence_rejected: int = 0
    duplicate_rejected: int = 0
    sanitize_rejected: int = 0
    forward_pruned: int = 0
    exact_states_visited: int = 0


class ConstrainedFragDiffusionAssembler:
    """Globally constrained fragment assembly.

    The generated fragment identities and generated fragment-pair topology stay
    fixed. The decoder may only switch the attachment *mode* for an already
    generated non-zero fragment pair.

    Improvements implemented here:

    1. If the generated topology is a connected tree, every generated edge is a
       bridge, so the search never creates a skip branch for those edges.
    2. Forward checking prunes a partial state as soon as the remaining valid
       attachment options can no longer connect all fragments.
    3. Final candidates must be fully RDKit-sanitized and, by default, connected.

    Exact tree DFS is retained only as an opt-in diagnostic. It is disabled by
    default because the 1,000-sample benchmark recovered no additional valid
    molecules after beam failure while adding substantial CPU work.
    """

    def __init__(
        self,
        fragments: FragmentLibrary,
        attachments: AttachmentLibrary,
        *,
        beam_size: int = 256,
        allow_mode_repair: bool = True,
        require_connected: bool = True,
        exact_tree_fallback: bool = False,
        exact_tree_max_edges: int = 12,
    ) -> None:
        self.fragments = fragments
        self.attachments = attachments
        self.beam_size = max(1, int(beam_size))
        self.allow_mode_repair = bool(allow_mode_repair)
        self.require_connected = bool(require_connected)
        self.exact_tree_fallback = bool(exact_tree_fallback)
        self.exact_tree_max_edges = max(0, int(exact_tree_max_edges))

    def _make_base(self, frag_ids: torch.Tensor) -> tuple[list[Chem.Mol], Chem.RWMol, list[int]]:
        mols: List[Chem.Mol] = [self.fragments.mol(int(i)) for i in frag_ids.tolist()]
        combined = mols[0]
        for m in mols[1:]:
            combined = Chem.CombineMols(combined, m)
        starts: list[int] = []
        cur = 0
        for m in mols:
            starts.append(cur)
            cur += m.GetNumAtoms()
        return mols, Chem.RWMol(combined), starts

    def _edge_specs(self, frag_ids: torch.Tensor, adjacency: torch.Tensor) -> tuple[list[_EdgeSpec], int, int]:
        specs: list[_EdgeSpec] = []
        attempted = 0
        lookup_missing = 0
        n = len(frag_ids)
        for i in range(n):
            for j in range(i + 1, n):
                edge_cls = int(adjacency[i, j])
                if edge_cls <= 0:
                    continue
                attempted += 1
                predicted = edge_cls - 1
                fi, fj = int(frag_ids[i]), int(frag_ids[j])

                mode_order: list[int] = [predicted]
                if self.allow_mode_repair:
                    mode_order += [m for m in self.attachments.available_modes(fi, fj) if m != predicted]

                options: list[tuple[int, int, int]] = []
                seen_pairs: set[tuple[int, int]] = set()
                for mode in mode_order:
                    atom_pair = self.attachments.lookup(fi, fj, mode)
                    if atom_pair is None:
                        continue
                    # Multiple lookup modes can resolve to the same atom pair.
                    # Keep the first one so the predicted mode is preferred.
                    if atom_pair in seen_pairs:
                        continue
                    seen_pairs.add(atom_pair)
                    options.append((mode, atom_pair[0], atom_pair[1]))

                if self.attachments.lookup(fi, fj, predicted) is None:
                    lookup_missing += 1
                specs.append(_EdgeSpec(i, j, predicted, options))

        # Constraint satisfaction heuristic: branch on the edge with the fewest
        # attachment choices first. Predicted mode remains first within options.
        specs.sort(key=lambda s: (len(s.options), s.i, s.j))
        return specs, attempted, lookup_missing

    @staticmethod
    def _can_add_valence_safe(rw: Chem.RWMol, ai: int, aj: int) -> tuple[Optional[Chem.RWMol], str | None]:
        if ai == aj or rw.GetBondBetweenAtoms(ai, aj) is not None:
            return None, "duplicate"
        trial = Chem.RWMol(rw)
        try:
            trial.AddBond(ai, aj, Chem.BondType.SINGLE)
            # Cheap partial-state pruning. Full aromaticity/kekulization checks
            # are deferred to final Chem.SanitizeMol().
            trial.GetMol().UpdatePropertyCache(strict=True)
        except Exception:
            return None, "valence"
        return trial, None

    @staticmethod
    def _beam_score(state: _BeamState, n_fragments: int) -> tuple[int, int, int, int]:
        _, components = _fragment_connectivity(n_fragments, state.selected_pairs)
        return (state.realized, -components, -state.repaired, -state.skipped)

    @staticmethod
    def _final_score(state: _BeamState, n_fragments: int) -> tuple[int, int, int, int, int]:
        connected, components = _fragment_connectivity(n_fragments, state.selected_pairs)
        return (int(connected), -components, state.realized, -state.repaired, -state.skipped)

    def _spec_has_safe_option(self, rw: Chem.RWMol, spec: _EdgeSpec, starts: list[int]) -> bool:
        for _, local_i, local_j in spec.options:
            ai = starts[spec.i] + local_i
            aj = starts[spec.j] + local_j
            trial, _ = self._can_add_valence_safe(rw, ai, aj)
            if trial is not None:
                return True
        return False

    def _forward_feasible(
        self,
        rw: Chem.RWMol,
        selected_pairs: Sequence[tuple[int, int]],
        remaining_specs: Sequence[_EdgeSpec],
        starts: list[int],
        n_fragments: int,
    ) -> bool:
        """CSP-style forward checking.

        For each remaining generated edge, determine whether at least one
        attachment option is still valence-safe. We then ask whether the current
        selected edges plus *all still-viable remaining fragment pairs* could
        possibly connect the graph. If not, this branch can never yield a valid
        connected molecule and is pruned immediately.
        """

        if not self.require_connected:
            return True

        viable_pairs: list[tuple[int, int]] = []
        for spec in remaining_specs:
            if self._spec_has_safe_option(rw, spec, starts):
                viable_pairs.append((spec.i, spec.j))

        possible_edges = list(selected_pairs) + viable_pairs
        connected_possible, _ = _fragment_connectivity(n_fragments, possible_edges)
        return connected_possible

    def _beam_search(
        self,
        base_rw: Chem.RWMol,
        specs: Sequence[_EdgeSpec],
        starts: list[int],
        n_fragments: int,
        generated_tree: bool,
        counters: _SearchCounters,
    ) -> list[_BeamState]:
        beam: list[_BeamState] = [
            _BeamState(
                rw=base_rw,
                selected_pairs=tuple(),
                selected_modes={},
                realized=0,
                repaired=0,
                skipped=0,
            )
        ]

        for edge_idx, spec in enumerate(specs):
            remaining = specs[edge_idx + 1 :]
            expanded: list[_BeamState] = []

            for state in beam:
                # Improvement (1): for a connected tree every edge is a bridge.
                # Skipping even one edge makes a connected final molecule
                # impossible, so do not waste beam capacity on skip branches.
                allow_skip = not (self.require_connected and generated_tree)
                if allow_skip:
                    if self._forward_feasible(
                        state.rw,
                        state.selected_pairs,
                        remaining,
                        starts,
                        n_fragments,
                    ):
                        expanded.append(
                            _BeamState(
                                rw=state.rw,
                                selected_pairs=state.selected_pairs,
                                selected_modes=dict(state.selected_modes),
                                realized=state.realized,
                                repaired=state.repaired,
                                skipped=state.skipped + 1,
                            )
                        )
                    else:
                        counters.forward_pruned += 1

                for mode, local_i, local_j in spec.options:
                    ai = starts[spec.i] + local_i
                    aj = starts[spec.j] + local_j
                    trial, reject = self._can_add_valence_safe(state.rw, ai, aj)
                    if trial is None:
                        if reject == "duplicate":
                            counters.duplicate_rejected += 1
                        else:
                            counters.valence_rejected += 1
                        continue

                    pairs = state.selected_pairs + ((spec.i, spec.j),)
                    if not self._forward_feasible(
                        trial,
                        pairs,
                        remaining,
                        starts,
                        n_fragments,
                    ):
                        counters.forward_pruned += 1
                        continue

                    modes = dict(state.selected_modes)
                    modes[(spec.i, spec.j)] = mode
                    expanded.append(
                        _BeamState(
                            rw=trial,
                            selected_pairs=pairs,
                            selected_modes=modes,
                            realized=state.realized + 1,
                            repaired=state.repaired + int(mode != spec.predicted_mode),
                            skipped=state.skipped,
                        )
                    )

            if not expanded:
                return []
            expanded.sort(key=lambda s: self._beam_score(s, n_fragments), reverse=True)
            beam = expanded[: self.beam_size]

        beam.sort(key=lambda s: self._final_score(s, n_fragments), reverse=True)
        return beam

    def _first_sanitized_candidate(
        self,
        states: Sequence[_BeamState],
        n_fragments: int,
        sanitize: bool,
        counters: _SearchCounters,
    ) -> tuple[Optional[_BeamState], Optional[Chem.Mol], Optional[_BeamState], Optional[Chem.Mol]]:
        best_disconnected: Optional[_BeamState] = None
        best_disconnected_mol: Optional[Chem.Mol] = None

        for state in states:
            connected, _ = _fragment_connectivity(n_fragments, state.selected_pairs)
            mol = state.rw.GetMol()
            if sanitize:
                mol, _ = _sanitize_copy(mol)
                if mol is None:
                    counters.sanitize_rejected += 1
                    continue
            if connected:
                return state, mol, best_disconnected, best_disconnected_mol
            if best_disconnected is None:
                best_disconnected = state
                best_disconnected_mol = mol

        return None, None, best_disconnected, best_disconnected_mol

    def _exact_tree_search(
        self,
        base_rw: Chem.RWMol,
        specs: Sequence[_EdgeSpec],
        starts: list[int],
        n_fragments: int,
        sanitize: bool,
        counters: _SearchCounters,
    ) -> tuple[Optional[_BeamState], Optional[Chem.Mol]]:
        """Exact DFS over attachment modes for a connected tree.

        There are no skip branches: every generated edge is a bridge and is
        mandatory. If this routine returns None after completing the search,
        there is no attachment-mode assignment within the lookup table that
        passes the implemented valence + final sanitization constraints.
        """

        def dfs(
            edge_idx: int,
            rw: Chem.RWMol,
            selected_pairs: tuple[tuple[int, int], ...],
            selected_modes: Dict[Tuple[int, int], int],
            repaired: int,
        ) -> tuple[Optional[_BeamState], Optional[Chem.Mol]]:
            counters.exact_states_visited += 1

            if edge_idx == len(specs):
                mol = rw.GetMol()
                if sanitize:
                    mol, _ = _sanitize_copy(mol)
                    if mol is None:
                        counters.sanitize_rejected += 1
                        return None, None
                connected, _ = _fragment_connectivity(n_fragments, selected_pairs)
                if not connected:
                    return None, None
                return (
                    _BeamState(
                        rw=rw,
                        selected_pairs=selected_pairs,
                        selected_modes=dict(selected_modes),
                        realized=len(selected_pairs),
                        repaired=repaired,
                        skipped=0,
                    ),
                    mol,
                )

            spec = specs[edge_idx]
            remaining = specs[edge_idx + 1 :]

            # options are already ordered with the diffusion-predicted mode first.
            for mode, local_i, local_j in spec.options:
                ai = starts[spec.i] + local_i
                aj = starts[spec.j] + local_j
                trial, reject = self._can_add_valence_safe(rw, ai, aj)
                if trial is None:
                    if reject == "duplicate":
                        counters.duplicate_rejected += 1
                    else:
                        counters.valence_rejected += 1
                    continue

                pairs = selected_pairs + ((spec.i, spec.j),)
                if not self._forward_feasible(
                    trial,
                    pairs,
                    remaining,
                    starts,
                    n_fragments,
                ):
                    counters.forward_pruned += 1
                    continue

                modes = dict(selected_modes)
                modes[(spec.i, spec.j)] = mode
                result_state, result_mol = dfs(
                    edge_idx + 1,
                    trial,
                    pairs,
                    modes,
                    repaired + int(mode != spec.predicted_mode),
                )
                if result_state is not None:
                    return result_state, result_mol

            return None, None

        return dfs(0, base_rw, tuple(), {}, 0)

    @staticmethod
    def _result_from_state(
        state: _BeamState,
        mol: Chem.Mol,
        *,
        attempted: int,
        lookup_missing: int,
        n_fragments: int,
        generated_connected: bool,
        generated_tree: bool,
        counters: _SearchCounters,
        search_method: str,
        exact_search_used: bool,
    ) -> AssemblyResult:
        connected, components = _fragment_connectivity(n_fragments, state.selected_pairs)
        return AssemblyResult(
            mol=mol,
            skipped_edges=attempted - state.realized,
            attempted_edges=attempted,
            connected=connected,
            num_components=components,
            repaired_mode_edges=state.repaired,
            lookup_missing_edges=lookup_missing,
            valence_rejected_options=counters.valence_rejected,
            duplicate_rejected_options=counters.duplicate_rejected,
            sanitize_rejected_candidates=counters.sanitize_rejected,
            selected_edges=state.realized,
            selected_modes=state.selected_modes,
            generated_topology_connected=generated_connected,
            generated_topology_tree=generated_tree,
            forward_pruned_states=counters.forward_pruned,
            exact_search_used=exact_search_used,
            exact_states_visited=counters.exact_states_visited,
            search_method=search_method,
        )

    def assemble(self, frag_ids: torch.Tensor, adjacency: torch.Tensor, sanitize: bool = True) -> AssemblyResult:
        frag_ids = frag_ids.detach().cpu().long()
        adjacency = adjacency.detach().cpu().long()
        if frag_ids.ndim != 1 or adjacency.shape != (len(frag_ids), len(frag_ids)):
            raise ValueError("expected frag_ids [N] and adjacency [N,N]")
        n = len(frag_ids)
        if n == 0:
            return AssemblyResult(None, 0, 0, failure_reason="empty graph")

        _, base_rw, starts = self._make_base(frag_ids)
        specs, attempted, lookup_missing = self._edge_specs(frag_ids, adjacency)
        generated_pairs = [(s.i, s.j) for s in specs]
        generated_connected, generated_components = _fragment_connectivity(n, generated_pairs)
        generated_tree = generated_connected and attempted == max(n - 1, 0)

        # If connected output is required and the generated fragment-pair graph is
        # already disconnected, no attachment-mode repair can invent a missing
        # fragment pair. Return a topology failure immediately.
        if self.require_connected and not generated_connected:
            if attempted < max(n - 1, 0):
                reason = "generated fragment graph has too few edges to connect all fragments"
            else:
                reason = "generated fragment topology is disconnected"
            return AssemblyResult(
                mol=None,
                skipped_edges=attempted,
                attempted_edges=attempted,
                connected=False,
                num_components=generated_components,
                repaired_mode_edges=0,
                lookup_missing_edges=lookup_missing,
                selected_edges=0,
                failure_reason=reason,
                generated_topology_connected=False,
                generated_topology_tree=False,
                search_method="topology_reject",
            )

        counters = _SearchCounters()

        beam = self._beam_search(
            base_rw,
            specs,
            starts,
            n,
            generated_tree,
            counters,
        )
        state, mol, best_disconnected, best_disconnected_mol = self._first_sanitized_candidate(
            beam,
            n,
            sanitize,
            counters,
        )
        if state is not None and mol is not None:
            return self._result_from_state(
                state,
                mol,
                attempted=attempted,
                lookup_missing=lookup_missing,
                n_fragments=n,
                generated_connected=generated_connected,
                generated_tree=generated_tree,
                counters=counters,
                search_method="beam",
                exact_search_used=False,
            )

        # Improvement (2): exact attachment-mode search for connected trees.
        exact_attempted = False
        if (
            self.require_connected
            and generated_tree
            and self.exact_tree_fallback
            and len(specs) <= self.exact_tree_max_edges
        ):
            exact_attempted = True
            exact_state, exact_mol = self._exact_tree_search(
                base_rw,
                specs,
                starts,
                n,
                sanitize,
                counters,
            )
            if exact_state is not None and exact_mol is not None:
                return self._result_from_state(
                    exact_state,
                    exact_mol,
                    attempted=attempted,
                    lookup_missing=lookup_missing,
                    n_fragments=n,
                    generated_connected=generated_connected,
                    generated_tree=generated_tree,
                    counters=counters,
                    search_method="exact_tree_fallback",
                    exact_search_used=True,
                )

        if best_disconnected is not None and best_disconnected_mol is not None and not self.require_connected:
            return self._result_from_state(
                best_disconnected,
                best_disconnected_mol,
                attempted=attempted,
                lookup_missing=lookup_missing,
                n_fragments=n,
                generated_connected=generated_connected,
                generated_tree=generated_tree,
                counters=counters,
                search_method="beam_disconnected",
                exact_search_used=exact_attempted,
            )

        reason = "no connected valence-safe sanitized assembly"
        if generated_tree and self.exact_tree_fallback and len(specs) > self.exact_tree_max_edges:
            reason += f"; exact tree fallback skipped because edges={len(specs)} > {self.exact_tree_max_edges}"
        elif exact_attempted:
            reason += "; exact tree search exhausted"

        return AssemblyResult(
            mol=None,
            skipped_edges=attempted,
            attempted_edges=attempted,
            connected=False,
            num_components=generated_components,
            repaired_mode_edges=0,
            lookup_missing_edges=lookup_missing,
            valence_rejected_options=counters.valence_rejected,
            duplicate_rejected_options=counters.duplicate_rejected,
            sanitize_rejected_candidates=counters.sanitize_rejected,
            selected_edges=0,
            failure_reason=reason,
            generated_topology_connected=generated_connected,
            generated_topology_tree=generated_tree,
            forward_pruned_states=counters.forward_pruned,
            exact_search_used=exact_attempted,
            exact_states_visited=counters.exact_states_visited,
            search_method="failed_after_exact" if exact_attempted else "failed_after_beam",
        )

    def to_smiles(self, frag_ids: torch.Tensor, adjacency: torch.Tensor) -> tuple[Optional[str], AssemblyResult]:
        result = self.assemble(frag_ids, adjacency, sanitize=True)
        if result.mol is None:
            return None, result
        return Chem.MolToSmiles(result.mol, isomericSmiles=True), result


def _fragment_connectivity(n: int, edges: Sequence[tuple[int, int]]) -> tuple[bool, int]:
    if n == 0:
        return False, 0
    if n == 1:
        return True, 1
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)
    components = len({find(i) for i in range(n)})
    return components == 1, components


def _consume_one_explicit_attachment_h(rw: Chem.RWMol, atom_idx: int) -> bool:
    """Consume one explicit H on an attachment endpoint, if present.

    Fragment vocabulary SMILES are chemically capped standalone molecules.  In
    particular RDKit represents pyrrolic sites as ``[nH]``.  The attachment
    targets recovered from FragDiffusion, however, can legitimately point to
    that atom: assembly is a *substitution* reaction, so the cap H must be
    replaced by the new inter-fragment bond.  Adding the bond without removing
    the H produces an artificial AtomValenceException.

    Implicit hydrogens need no manual edit: RDKit recomputes them after the new
    bond is added.  Explicit hydrogens are decremented one at a time, which also
    handles sites such as [NH2] that may accept more than one substitution over
    a multi-edge assembly.
    """

    atom = rw.GetAtomWithIdx(int(atom_idx))
    n_explicit = int(atom.GetNumExplicitHs())
    if n_explicit <= 0:
        return False
    atom.SetNumExplicitHs(n_explicit - 1)
    return True


def _try_add_single_bond(
    rw: Chem.RWMol,
    ai: int,
    aj: int,
    *,
    replace_explicit_h: bool,
) -> tuple[Optional[Chem.RWMol], str | None, int]:
    """Try one inter-fragment single bond with optional cap-H substitution."""

    if ai == aj or rw.GetBondBetweenAtoms(ai, aj) is not None:
        return None, "duplicate", 0
    trial = Chem.RWMol(rw)
    replaced = 0
    try:
        if replace_explicit_h:
            replaced += int(_consume_one_explicit_attachment_h(trial, ai))
            replaced += int(_consume_one_explicit_attachment_h(trial, aj))
        trial.AddBond(ai, aj, Chem.BondType.SINGLE)
        # Cheap partial-state pruning. Full aromaticity/kekulization checks are
        # still deferred to final Chem.SanitizeMol().
        trial.GetMol().UpdatePropertyCache(strict=True)
    except Exception:
        return None, "valence", 0
    return trial, None, replaced


def _static_single_attachment_ok(mol: Chem.Mol, atom_idx: int, *, replace_explicit_h: bool) -> bool:
    """Whether a local fragment atom can accept at least one external C bond.

    This is only a *first-attachment* mask used before neural top-k selection;
    the beam search still performs state-dependent valence checks after every
    realized edge.  A dummy carbon is used merely to ask RDKit whether one
    additional single bond is chemically representable under the chosen cap-H
    semantics.
    """

    rw = Chem.RWMol(Chem.Mol(mol))
    dummy = Chem.Atom(6)
    dummy_idx = rw.AddAtom(dummy)
    trial, _, _ = _try_add_single_bond(
        rw, int(atom_idx), int(dummy_idx), replace_explicit_h=replace_explicit_h
    )
    if trial is None:
        return False
    probe, _ = _sanitize_copy_no_recursion(trial.GetMol())
    return probe is not None


def _sanitize_copy_no_recursion(mol: Chem.Mol) -> tuple[Optional[Chem.Mol], Optional[str]]:
    """Internal sanitize helper used while constructing static site masks."""
    probe = Chem.Mol(mol)
    try:
        flag = Chem.SanitizeMol(probe, catchErrors=True)
    except Exception as exc:
        return None, type(exc).__name__
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        return None, str(flag)
    return probe, None


def _sanitize_copy(mol: Chem.Mol) -> tuple[Optional[Chem.Mol], Optional[str]]:
    probe = Chem.Mol(mol)
    try:
        flag = Chem.SanitizeMol(probe, catchErrors=True)
    except Exception as exc:
        return None, type(exc).__name__
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        return None, str(flag)
    return probe, None


@dataclass
class _NeuralSiteState:
    rw: Chem.RWMol
    selected_pairs: tuple[tuple[int, int], ...]
    score: float
    repaired: int
    skipped: int
    h_replacements: int = 0


class NeuralSiteAssembler:
    """Lookup-free neural coarse-to-fine fragment assembler.

    Candidate atom endpoints come directly from the denoiser's directed
    ``site_logits``. For each generated coarse fragment edge i--j we take the
    top-k atom choices for i->j and j->i, form their Cartesian product, and rank
    the atom pairs by the sum of neural log-probabilities. A small RDKit-aware
    beam search then enforces local valence and final connectivity/sanitization.

    Importantly, ``AttachmentLibrary`` and ``fragment_edge_index.csv`` are never
    consulted here. Chemistry is used only as a hard feasibility constraint on
    neural proposals.
    """
    def __init__(
        self,
        fragments: FragmentLibrary,
        *,
        beam_size: int = 256,
        site_topk: int = 4,
        require_connected: bool = True,
        allow_cycle_edge_skip: bool = True,
        replace_explicit_h: bool = True,
        static_site_mask: bool = True,
        assembly_selection: str = "best",
        assembly_temperature: float = 0.1,
        max_valid_products: int = 16,
    ) -> None:
        self.fragments = fragments
        self.beam_size = max(1, int(beam_size))
        self.site_topk = max(1, int(site_topk))
        self.require_connected = bool(require_connected)
        self.allow_cycle_edge_skip = bool(allow_cycle_edge_skip)
        self.replace_explicit_h = bool(replace_explicit_h)
        self.static_site_mask = bool(static_site_mask)
        self._static_site_cache: Dict[int, tuple[bool, ...]] = {}

        if assembly_selection not in {"best", "uniform", "softmax"}:
            raise ValueError(
                "assembly_selection must be one of "
                "{'best', 'uniform', 'softmax'}"
            )

        if assembly_temperature <= 0 and assembly_selection == "softmax":
            raise ValueError(
                "assembly_temperature must be > 0 "
                "when assembly_selection='softmax'"
            )

        if max_valid_products < 1:
            raise ValueError("max_valid_products must be >= 1")

        self.assembly_selection = str(assembly_selection)
        self.assembly_temperature = float(assembly_temperature)
        self.max_valid_products = int(max_valid_products)

    def _try_add(
        self, rw: Chem.RWMol, ai: int, aj: int
    ) -> tuple[Optional[Chem.RWMol], str | None, int]:
        return _try_add_single_bond(
            rw,
            ai,
            aj,
            replace_explicit_h=self.replace_explicit_h,
        )

    def _static_valid_sites(self, frag_id: int, mol: Chem.Mol) -> tuple[bool, ...]:
        cached = self._static_site_cache.get(int(frag_id))
        if cached is not None:
            return cached
        if not self.static_site_mask:
            mask = tuple(True for _ in range(mol.GetNumAtoms()))
        else:
            mask = tuple(
                _static_single_attachment_ok(
                    mol,
                    a,
                    replace_explicit_h=self.replace_explicit_h,
                )
                for a in range(mol.GetNumAtoms())
            )
        self._static_site_cache[int(frag_id)] = mask
        return mask

    def _edge_candidates(
        self,
        i: int,
        j: int,
        mols: Sequence[Chem.Mol],
        frag_ids: torch.Tensor,
        sites: Optional[torch.Tensor],
        site_logits: Optional[torch.Tensor],
    ) -> list[tuple[float, int, int, bool]]:
        ni, nj = mols[i].GetNumAtoms(), mols[j].GetNumAtoms()
        if site_logits is None:
            if sites is None:
                return []
            ai, aj = int(sites[i, j]), int(sites[j, i])
            if 0 <= ai < ni and 0 <= aj < nj:
                return [(0.0, ai, aj, False)]
            return []

        li = site_logits[i, j, :ni].float().clone()
        lj = site_logits[j, i, :nj].float().clone()
        if li.numel() == 0 or lj.numel() == 0:
            return []

        # Do not spend top-k capacity on atoms that cannot accept even one
        # external single bond.  Crucially, [nH] remains available when
        # replace_explicit_h=True because attachment substitutes its explicit H.
        mi = torch.tensor(
            self._static_valid_sites(int(frag_ids[i]), mols[i]), dtype=torch.bool
        )
        mj = torch.tensor(
            self._static_valid_sites(int(frag_ids[j]), mols[j]), dtype=torch.bool
        )
        if not bool(mi.any()) or not bool(mj.any()):
            return []
        li = li.masked_fill(~mi, -1e9)
        lj = lj.masked_fill(~mj, -1e9)
        lpi = torch.log_softmax(li, dim=-1)
        lpj = torch.log_softmax(lj, dim=-1)
        ki = min(self.site_topk, int(mi.sum().item()))
        kj = min(self.site_topk, int(mj.sum().item()))
        vi, ii = torch.topk(lpi, ki)
        vj, jj = torch.topk(lpj, kj)

        top1 = (int(ii[0]), int(jj[0]))
        out: list[tuple[float, int, int, bool]] = []
        for a in range(ki):
            for b in range(kj):
                ai, aj = int(ii[a]), int(jj[b])
                out.append(
                    (
                        float(vi[a] + vj[b]),
                        ai,
                        aj,
                        (ai, aj) != top1,
                    )
                )
        out.sort(key=lambda z: z[0], reverse=True)
        return out

    def assemble_graph(self, graph, sanitize: bool = True) -> AssemblyResult:
        frag_ids = graph.x.detach().cpu().long()
        adjacency = graph.e.detach().cpu().long()
        sites = None if graph.sites is None else graph.sites.detach().cpu().long()
        site_logits = None if graph.site_logits is None else graph.site_logits.detach().cpu()
        n = len(frag_ids)
        if adjacency.shape != (n, n):
            raise ValueError("expected adjacency [N,N]")
        if n == 0:
            return AssemblyResult(None, 0, 0, failure_reason="empty graph", search_method="neural_sites")

        mols = [self.fragments.mol(int(i)) for i in frag_ids.tolist()]
        static_masked_atoms = 0
        if self.static_site_mask:
            for fid, mol in zip(frag_ids.tolist(), mols):
                static_masked_atoms += sum(
                    not ok for ok in self._static_valid_sites(int(fid), mol)
                )
        combined = mols[0]
        for mol in mols[1:]:
            combined = Chem.CombineMols(combined, mol)
        base_rw = Chem.RWMol(combined)

        starts: list[int] = []
        cur = 0
        for mol in mols:
            starts.append(cur)
            cur += mol.GetNumAtoms()

        edges = [
            (i, j)
            for i in range(n)
            for j in range(i + 1, n)
            if int(adjacency[i, j]) > 0
        ]
        generated_connected, generated_components = _fragment_connectivity(n, edges)
        generated_tree = generated_connected and len(edges) == max(n - 1, 0)
        if self.require_connected and not generated_connected:
            return AssemblyResult(
                None,
                skipped_edges=len(edges),
                attempted_edges=len(edges),
                connected=False,
                num_components=generated_components,
                failure_reason="generated fragment topology is disconnected",
                generated_topology_connected=False,
                generated_topology_tree=False,
                search_method="neural_sites_topology_reject",
            )

        attachment_h_replacements = 0
        beam_exhausted_edge = -1
        edge_options: list[tuple[int, int, list[tuple[float, int, int, bool]]]] = []
        candidate_count = 0
        for i, j in edges:
            opts = self._edge_candidates(i, j, mols, frag_ids, sites, site_logits)
            candidate_count += len(opts)
            if not opts:
                return AssemblyResult(
                    None,
                    skipped_edges=len(edges),
                    attempted_edges=len(edges),
                    connected=False,
                    num_components=generated_components,
                    failure_reason=f"no neural site candidate for fragment edge ({i},{j})",
                    generated_topology_connected=generated_connected,
                    generated_topology_tree=generated_tree,
                    search_method="neural_sites_no_candidate",
                    neural_site_candidate_pairs=candidate_count,
                    attachment_h_replacements=0,
                    static_site_masked_atoms=static_masked_atoms,
                    beam_exhausted_edge=beam_exhausted_edge,
                )
            edge_options.append((i, j, opts))

        beam = [_NeuralSiteState(base_rw, tuple(), 0.0, 0, 0, 0)]
        valence_rejected = 0
        duplicate_rejected = 0

        # Fail-fast on the most constrained edges first. This is a CSP-style
        # heuristic and preserves the same candidate scores/chemistry.
        edge_options.sort(key=lambda z: (len(z[2]), z[0], z[1]))

        for edge_ord, (i, j, opts) in enumerate(edge_options):
            expanded: list[_NeuralSiteState] = []
            # In a tree every coarse edge is a bridge. For cyclic topologies an
            # optional skip branch can absorb a spurious generated coarse edge.
            allow_skip = self.allow_cycle_edge_skip and not generated_tree
            for state in beam:
                if allow_skip:
                    expanded.append(
                        _NeuralSiteState(
                            state.rw,
                            state.selected_pairs,
                            state.score - 12.0,  # strong learned-edge omission penalty
                            state.repaired,
                            state.skipped + 1,
                            state.h_replacements,
                        )
                    )
                for score, local_i, local_j, repaired in opts:
                    ai = starts[i] + local_i
                    aj = starts[j] + local_j
                    trial, reject, n_h_replaced = self._try_add(state.rw, ai, aj)
                    if trial is None:
                        if reject == "duplicate":
                            duplicate_rejected += 1
                        else:
                            valence_rejected += 1
                        continue
                    expanded.append(
                        _NeuralSiteState(
                            trial,
                            state.selected_pairs + ((i, j),),
                            state.score + score,
                            state.repaired + int(repaired),
                            state.skipped,
                            state.h_replacements + n_h_replaced,
                        )
                    )
            if not expanded:
                beam_exhausted_edge = edge_ord
                beam = []
                break
            # Preserve topology first, then neural likelihood. This avoids a
            # high-probability but disconnected collection of local choices.
            expanded.sort(
                key=lambda st: (len(st.selected_pairs), -st.skipped, st.score, -st.repaired),
                reverse=True,
            )
            beam = expanded[: self.beam_size]

        sanitize_rejected = 0
        best_disconnected = None
        ordered_beam = sorted(
            beam,
            key=lambda st: (len(st.selected_pairs), -st.skipped, st.score, -st.repaired),
            reverse=True,
        )
        valid_products: list[tuple[_NeuralSiteState, Chem.Mol, int, str]] = []
        seen_smiles: set[str] = set()

        for state in ordered_beam:
            connected, components = _fragment_connectivity(n, state.selected_pairs)
            mol = state.rw.GetMol()
            if sanitize:
                mol, _ = _sanitize_copy(mol)
                if mol is None:
                    sanitize_rejected += 1
                    continue
            if connected or not self.require_connected:
                # --------------------------------------------------------------
                # BEST:
                # Preserve the original deterministic behavior exactly.
                #
                # ordered_beam is already sorted from best -> worst, so the first
                # sanitized connected candidate is the original best-valid output.
                # --------------------------------------------------------------
                if self.assembly_selection == "best":
                    return AssemblyResult(
                        mol=mol,
                        skipped_edges=state.skipped,
                        attempted_edges=len(edges),
                        connected=connected,
                        num_components=components,
                        repaired_mode_edges=0,
                        lookup_missing_edges=0,
                        valence_rejected_options=valence_rejected,
                        duplicate_rejected_options=duplicate_rejected,
                        sanitize_rejected_candidates=sanitize_rejected,
                        selected_edges=len(state.selected_pairs),
                        generated_topology_connected=generated_connected,
                        generated_topology_tree=generated_tree,
                        search_method="neural_site_topk_beam",
                        neural_site_repaired_edges=state.repaired,
                        neural_site_candidate_pairs=candidate_count,
                        attachment_h_replacements=state.h_replacements,
                        static_site_masked_atoms=static_masked_atoms,
                        beam_exhausted_edge=beam_exhausted_edge,
                        valid_unique_products_considered=1,
                    )

                # --------------------------------------------------------------
                # UNIFORM / SOFTMAX:
                # Collect only products that have ALREADY passed
                #
                #   1. RDKit sanitization
                #   2. connectivity requirement
                #   3. SMILES uniqueness filtering
                #
                # Thus stochastic selection itself cannot convert a valid
                # candidate into an invalid molecule.
                # --------------------------------------------------------------
                smi = Chem.MolToSmiles(
                    mol,
                    canonical=True,
                    isomericSmiles=True,
                )

                if smi not in seen_smiles:
                    seen_smiles.add(smi)
                    valid_products.append(
                        (state, mol, components, smi)
                    )

                    if len(valid_products) >= self.max_valid_products:
                        break
            
            elif best_disconnected is None:
                best_disconnected = (state, mol, components)

        if valid_products:
            # --------------------------------------------------------------
            # Select ONE candidate from the already-valid product pool.
            # --------------------------------------------------------------
            if self.assembly_selection == "uniform":
                chosen_idx = random.randrange(
                    len(valid_products)
                )

            elif self.assembly_selection == "softmax":
                scores = torch.tensor(
                    [product[0].score for product in valid_products],
                    dtype=torch.float64,
                )

                probs = torch.softmax(
                    scores / self.assembly_temperature,
                    dim=0,
                )

                chosen_idx = int(
                    torch.multinomial(
                        probs,
                        num_samples=1,
                    ).item()
                )

            else:
                raise RuntimeError(
                    "valid_products reached stochastic-selection block "
                    f"with assembly_selection={self.assembly_selection!r}"
                )

            state, mol, components, _ = valid_products[
                chosen_idx
            ]

            connected, _ = _fragment_connectivity(
                n,
                state.selected_pairs,
            )

            return AssemblyResult(
                mol=mol,
                skipped_edges=state.skipped,
                attempted_edges=len(edges),
                connected=connected,
                num_components=components,
                repaired_mode_edges=0,
                lookup_missing_edges=0,
                valence_rejected_options=valence_rejected,
                duplicate_rejected_options=duplicate_rejected,
                sanitize_rejected_candidates=sanitize_rejected,
                selected_edges=len(state.selected_pairs),
                generated_topology_connected=generated_connected,
                generated_topology_tree=generated_tree,
                search_method=(
                    "neural_site_valid_product_uniform"
                    if self.assembly_selection == "uniform"
                    else "neural_site_valid_product_softmax"
                ),
                neural_site_repaired_edges=state.repaired,
                neural_site_candidate_pairs=candidate_count,
                attachment_h_replacements=state.h_replacements,
                static_site_masked_atoms=static_masked_atoms,
                beam_exhausted_edge=beam_exhausted_edge,
                valid_unique_products_considered=len(
                    valid_products
                ),
            )

        return AssemblyResult(
            mol=None,
            skipped_edges=len(edges),
            attempted_edges=len(edges),
            connected=False,
            num_components=(generated_components if best_disconnected is None else best_disconnected[2]),
            valence_rejected_options=valence_rejected,
            duplicate_rejected_options=duplicate_rejected,
            sanitize_rejected_candidates=sanitize_rejected,
            selected_edges=0,
            failure_reason="no connected valence-safe sanitized neural-site assembly",
            generated_topology_connected=generated_connected,
            generated_topology_tree=generated_tree,
            search_method="neural_site_failed",
            neural_site_candidate_pairs=candidate_count,
            attachment_h_replacements=(max((st.h_replacements for st in beam), default=0)),
            static_site_masked_atoms=static_masked_atoms,
            beam_exhausted_edge=beam_exhausted_edge,
            valid_unique_products_considered=len(valid_products),
        )

    def to_smiles_graph(self, graph) -> tuple[Optional[str], AssemblyResult]:
        result = self.assemble_graph(graph, sanitize=True)
        if result.mol is None:
            return None, result
        return Chem.MolToSmiles(result.mol, isomericSmiles=True), result
