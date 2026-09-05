from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from fragflex.config import FragFlexConfig
from fragflex.data.stats import FragFlexStats
from fragflex.graph import DenseGraphBatch, FragmentGraph, collate_fragment_graphs, pack_variable_graphs

from .posterior import (
    ordinary_qbar,
    ordinary_qstep,
    posterior_from_clean_prediction,
    structural_qbar,
    structural_qstep,
)
from .schedules import DiffusionSchedules


@dataclass
class CorruptionBatch:
    noisy: DenseGraphBatch
    target_x: torch.Tensor
    target_e: torch.Tensor
    target_node_mask: torch.Tensor
    target_root_mask: torch.Tensor
    target_sites: Optional[torch.Tensor]
    delta_input: DenseGraphBatch
    delta_target: torch.Tensor


class OneAtomFragDiffusion:
    """FragFlex's single-root terminal process.

    Clean states are FragDiffusion fragment graphs. Forward structural diffusion
    protects one uniformly selected root and deletes every other fragment according
    to the configured structural-event schedule, so G_T always has exactly one
    graph node. ``root_terminal_prior`` independently controls the categorical
    identity of that protected root: ``atom`` reproduces the legacy one-atom
    latent start, while ``fragment`` uses the empirical fragment marginal and can
    therefore start from either a single-atom or multi-atom fragment.

    The same root prior is used in direct training corruption, exact reverse
    posteriors, and terminal sampling. This remains fully compatible with the
    early-insertion structural schedules in ``DiffusionSchedules``.
    """

    def __init__(self, config: FragFlexConfig, stats: FragFlexStats):
        self.config = config
        self.stats = stats
        self.schedules = DiffusionSchedules.build(
            config.diffusion_steps,
            config.zeta_D,
            config.zeta_w,
            config.cosine_s,
            zeta_schedule=config.zeta_schedule,
            zeta_sampling_peak_step=config.zeta_sampling_peak_step,
            zeta_sampling_tau=config.zeta_sampling_tau,
            zeta_event_rel_threshold=config.zeta_event_rel_threshold,
        )

        # Ordinary node state order:
        #   [fragment IDs | atom-latent IDs] + [DEL | DEL*]
        self.fragment_base_marginal = torch.cat(
            [stats.fragment_marginal, torch.zeros(stats.num_atom_latents)]
        )
        self.atom_base_marginal = torch.cat(
            [torch.zeros(stats.num_fragments), stats.atom_marginal]
        )

        prior = str(config.root_terminal_prior).lower()
        if prior not in {"atom", "fragment"}:
            raise ValueError(
                "root_terminal_prior must be 'atom' or 'fragment', "
                f"got {config.root_terminal_prior!r}"
            )
        self.root_terminal_prior = prior

        # Sampling uses this pure-Python lookup to decide whether the auxiliary
        # DEL* count network can possibly matter at timestep t. This avoids a
        # GPU scalar synchronization and, more importantly, skips the entire
        # delta-model forward on structural-event-free timesteps.
        self.event_active = tuple(bool(v > 0) for v in self.schedules.event_pmf.tolist())
        self._resident_device = torch.device("cpu")

    @property
    def T(self) -> int:
        return self.config.diffusion_steps

    @property
    def node_state_count(self) -> int:
        return self.stats.base_node_states + 2

    @property
    def edge_state_count(self) -> int:
        return self.stats.num_edge_types + 2

    @property
    def node_del_id(self) -> int:
        return self.stats.node_del_id

    @property
    def node_delt_id(self) -> int:
        return self.stats.node_delt_id

    @property
    def edge_del_id(self) -> int:
        return self.stats.edge_del_id

    @property
    def edge_delt_id(self) -> int:
        return self.stats.edge_delt_id

    def to(self, device: torch.device | str) -> "OneAtomFragDiffusion":
        """Make non-Module diffusion tensors resident on ``device`` exactly once.

        ``OneAtomFragDiffusion`` is intentionally not an ``nn.Module``, so
        ``FragFlexModel.to(cuda)`` does not move these tensors automatically.
        The old sampler consequently rebuilt/copied schedules inside every
        reverse step. Caching residency removes that transfer without changing
        any transition probabilities.
        """

        device = torch.device(device)
        if self._resident_device == device:
            return self
        self.schedules = self.schedules.to(device)
        self.fragment_base_marginal = self.fragment_base_marginal.to(device)
        self.atom_base_marginal = self.atom_base_marginal.to(device)
        self.stats.fragment_marginal = self.stats.fragment_marginal.to(device)
        self.stats.atom_marginal = self.stats.atom_marginal.to(device)
        self.stats.edge_marginal = self.stats.edge_marginal.to(device)
        if self.stats.fragment_atom_counts is not None:
            self.stats.fragment_atom_counts = self.stats.fragment_atom_counts.to(device)
        self._resident_device = device
        return self

    @property
    def root_base_marginal(self) -> torch.Tensor:
        """Terminal categorical marginal of the single protected root."""

        if self.root_terminal_prior == "fragment":
            return self.fragment_base_marginal
        return self.atom_base_marginal

    def sample_event_times(self, shape: tuple[int, ...], device: torch.device | str) -> torch.Tensor:
        """Sample structural event times in the explicit integer range 1..T.

        GrIDDD's implementation slices ``_d_zetas[1:]`` before multinomial
        sampling. FragFlex deliberately adds one back to the sampled index, making
        the convention unambiguous and unit-testable.
        """

        pmf = self.schedules.event_pmf.to(device)[1:]
        n = 1
        for s in shape:
            n *= s
        idx = torch.multinomial(pmf, n, replacement=True).reshape(shape)
        return idx + 1

    @staticmethod
    def _draw(prob: torch.Tensor) -> torch.Tensor:
        prob = prob.clamp_min(0)
        prob = prob / prob.sum(-1, keepdim=True).clamp_min(1e-12)
        return torch.multinomial(prob.reshape(-1, prob.shape[-1]), 1).reshape(prob.shape[:-1])

    def _node_forward_probs(self, clean_ids: torch.Tensor, t: int, root: bool) -> torch.Tensor:
        device = clean_ids.device
        d = self.node_state_count
        clean_oh = F.one_hot(clean_ids, num_classes=d).float()
        marginal = (self.root_base_marginal if root else self.fragment_base_marginal).to(device)
        m = torch.zeros(d, device=device)
        m[: self.stats.base_node_states] = marginal
        abar = self.schedules.alpha_bar[t].to(device)
        return abar * clean_oh + (1.0 - abar) * m

    def _edge_forward_probs(self, clean_ids: torch.Tensor, t: int) -> torch.Tensor:
        device = clean_ids.device
        d = self.edge_state_count
        clean_oh = F.one_hot(clean_ids, num_classes=d).float()
        m = torch.zeros(d, device=device)
        m[: self.stats.num_edge_types] = self.stats.edge_marginal.to(device)
        abar = self.schedules.alpha_bar[t].to(device)
        return abar * clean_oh + (1.0 - abar) * m

    def corrupt(
        self,
        graphs: Iterable[FragmentGraph],
        *,
        device: torch.device | str = "cpu",
        timesteps: Optional[torch.Tensor] = None,
        root_indices: Optional[torch.Tensor] = None,
    ) -> CorruptionBatch:
        """Sample a direct clean->G_t corruption for training."""

        graphs = list(graphs)
        x0, e0, node_mask0 = collate_fragment_graphs(graphs)
        x0, e0, node_mask0 = x0.to(device), e0.to(device), node_mask0.to(device)
        B = x0.shape[0]

        if timesteps is None:
            timesteps = torch.randint(1, self.T + 1, (B,), device=device)
        else:
            timesteps = timesteps.to(device).long().reshape(B)
        if root_indices is None:
            roots = []
            for b in range(B):
                n = int(node_mask0[b].sum())
                roots.append(torch.randint(0, n, (1,), device=device))
            root_indices = torch.cat(roots).long()
        else:
            root_indices = root_indices.to(device).long().reshape(B)

        xs: list[torch.Tensor] = []
        es: list[torch.Tensor] = []
        roots_out: list[torch.Tensor] = []
        delts_out: list[torch.Tensor] = []
        txs: list[torch.Tensor] = []
        tes: list[torch.Tensor] = []
        tss: list[torch.Tensor] = []

        aux_xs: list[torch.Tensor] = []
        aux_es: list[torch.Tensor] = []
        aux_roots: list[torch.Tensor] = []
        aux_delts: list[torch.Tensor] = []
        delta_targets: list[int] = []

        for b in range(B):
            n = int(node_mask0[b].sum())
            t = int(timesteps[b])
            r = int(root_indices[b])
            if not (0 <= r < n):
                raise ValueError(f"root index {r} out of range for graph with {n} nodes")

            events = self.sample_event_times((n,), device)
            events[r] = self.T + 1  # root is protected from structural deletion.
            is_root = torch.arange(n, device=device) == r
            completed = (~is_root) & (events < t)
            at_boundary = (~is_root) & (events == t)
            keep = ~completed
            keep_idx = torch.nonzero(keep, as_tuple=False).flatten()

            target_x = x0[b, :n][keep_idx]
            target_e = e0[b, :n, :n][keep_idx][:, keep_idx]
            if self.stats.uses_neural_sites:
                if graphs[b].sites is None:
                    raise ValueError("neural_sites training graph is missing site targets")
                full_sites = graphs[b].sites.to(device)
                target_sites = full_sites[keep_idx][:, keep_idx]
                tss.append(target_sites)
            root_mask = is_root[keep_idx]
            delt_mask = at_boundary[keep_idx]

            noisy_x = torch.empty_like(target_x)
            for j in range(target_x.numel()):
                if bool(delt_mask[j]):
                    noisy_x[j] = self.node_delt_id
                elif bool(root_mask[j]):
                    noisy_x[j] = self._draw(self._node_forward_probs(target_x[j : j + 1], t, True))[0]
                else:
                    noisy_x[j] = self._draw(self._node_forward_probs(target_x[j : j + 1], t, False))[0]

            m = target_x.numel()
            noisy_e = torch.zeros((m, m), dtype=torch.long, device=device)
            for i in range(m):
                for j in range(i + 1, m):
                    if bool(delt_mask[i] or delt_mask[j]):
                        val = self.edge_delt_id
                    else:
                        val = int(self._draw(self._edge_forward_probs(target_e[i, j].reshape(1), t))[0])
                    noisy_e[i, j] = noisy_e[j, i] = val

            xs.append(noisy_x)
            es.append(noisy_e)
            roots_out.append(root_mask)
            delts_out.append(delt_mask)
            txs.append(target_x)
            tes.append(target_e)

            # Auxiliary DEL* count model sees the same corrupted graph with all
            # boundary DEL* nodes hidden, exactly as in GrIDDD.
            aux_keep = ~delt_mask
            aux_x = noisy_x[aux_keep]
            aux_e = noisy_e[aux_keep][:, aux_keep]
            aux_root = root_mask[aux_keep]
            aux_delt = torch.zeros_like(aux_root)
            aux_xs.append(aux_x)
            aux_es.append(aux_e)
            aux_roots.append(aux_root)
            aux_delts.append(aux_delt)
            delta_targets.append(int(delt_mask.sum()))

        noisy = pack_variable_graphs(xs, es, roots_out, delts_out, timesteps)
        target_batch = pack_variable_graphs(
            txs,
            tes,
            roots_out,
            [torch.zeros_like(d) for d in delts_out],
            timesteps,
        )
        target_sites_batch: Optional[torch.Tensor] = None
        if self.stats.uses_neural_sites:
            target_sites_batch = torch.full_like(target_batch.e, -1)
            for b, site in enumerate(tss):
                m = site.shape[0]
                target_sites_batch[b, :m, :m] = site

        delta_input = pack_variable_graphs(aux_xs, aux_es, aux_roots, aux_delts, timesteps)
        delta_target = torch.tensor(delta_targets, dtype=torch.long, device=device)
        delta_target.clamp_(max=self.config.max_delta_per_step)

        return CorruptionBatch(
            noisy=noisy,
            target_x=target_batch.x,
            target_e=target_batch.e,
            target_node_mask=target_batch.node_mask,
            target_root_mask=target_batch.root_mask,
            target_sites=target_sites_batch,
            delta_input=delta_input,
            delta_target=delta_target,
        )

    def sample_limit(self, batch_size: int, device: torch.device | str) -> DenseGraphBatch:
        """Sample G_T with one root from the configured categorical prior."""

        if self.root_terminal_prior == "fragment":
            x = torch.multinomial(
                self.stats.fragment_marginal.to(device), batch_size, replacement=True
            ).reshape(batch_size, 1).long()
        else:
            atom = torch.multinomial(
                self.stats.atom_marginal.to(device), batch_size, replacement=True
            )
            x = (self.stats.num_fragments + atom).reshape(batch_size, 1).long()
        e = torch.zeros((batch_size, 1, 1), dtype=torch.long, device=device)
        mask = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
        root = torch.ones_like(mask)
        delt = torch.zeros_like(mask)
        t = torch.full((batch_size,), self.T, dtype=torch.long, device=device)
        return DenseGraphBatch(x=x, e=e, node_mask=mask, root_mask=root, t=t, delt_mask=delt)

    def insert_delt(self, state: DenseGraphBatch, n_add: torch.Tensor) -> DenseGraphBatch:
        """Append DEL* placeholders with a batched tensor update.

        Semantics are identical to v1: valid nodes stay in their original
        prefix order, newly inserted nodes are appended, and every edge touching
        an inserted node starts in the structural DEL* edge state. The only host
        synchronization is a single ``max()`` used to determine the padded batch
        width, replacing the previous per-sample scalar conversions and packing
        loop.
        """

        B, old_nmax = state.x.shape
        counts = state.node_mask.sum(-1).long()
        capacity = (self.config.max_nodes - counts).clamp_min(0)
        add = torch.minimum(n_add.long().clamp_min(0), capacity)
        new_counts = counts + add

        # Dynamic dense tensors need a Python shape. One scalar sync per actual
        # insertion step is much cheaper than B syncs plus B Python repacks.
        new_nmax = int(new_counts.max().item())
        if new_nmax <= old_nmax:
            if int(add.max().item()) == 0:
                return state
            new_nmax = old_nmax

        device = state.x.device
        x = torch.zeros((B, new_nmax), dtype=state.x.dtype, device=device)
        e = torch.zeros((B, new_nmax, new_nmax), dtype=state.e.dtype, device=device)
        node_mask = torch.zeros((B, new_nmax), dtype=torch.bool, device=device)
        root_mask = torch.zeros((B, new_nmax), dtype=torch.bool, device=device)
        delt_mask = torch.zeros((B, new_nmax), dtype=torch.bool, device=device)

        x[:, :old_nmax] = state.x
        e[:, :old_nmax, :old_nmax] = state.e
        node_mask[:, :old_nmax] = state.node_mask
        root_mask[:, :old_nmax] = state.root_mask
        if state.delt_mask is not None:
            delt_mask[:, :old_nmax] = state.delt_mask

        pos = torch.arange(new_nmax, device=device).unsqueeze(0)
        new_valid = pos < new_counts.unsqueeze(1)
        inserted = (pos >= counts.unsqueeze(1)) & new_valid

        node_mask |= new_valid
        x[inserted] = self.node_delt_id
        delt_mask[inserted] = True

        pair_valid = new_valid[:, :, None] & new_valid[:, None, :]
        touches_inserted = inserted[:, :, None] | inserted[:, None, :]
        e[touches_inserted & pair_valid] = self.edge_delt_id
        diag = torch.arange(new_nmax, device=device)
        e[:, diag, diag] = 0

        return DenseGraphBatch(
            x=x,
            e=e,
            node_mask=node_mask,
            root_mask=root_mask,
            t=state.t,
            delt_mask=delt_mask,
        )

    @staticmethod
    def _pad_clean_node_probs(
        probs: torch.Tensor, num_fragments: int, base_node_states: int
    ) -> torch.Tensor:
        # Denoiser outputs fragment identities only. Atom latent and DEL states
        # have zero probability at clean time t=0.
        D = base_node_states + 2
        out = torch.zeros((*probs.shape[:-1], D), dtype=probs.dtype, device=probs.device)
        out[..., :num_fragments] = probs
        return out

    @staticmethod
    def _pad_clean_edge_probs(probs: torch.Tensor, num_edge_types: int) -> torch.Tensor:
        D = num_edge_types + 2
        out = torch.zeros((*probs.shape[:-1], D), dtype=probs.dtype, device=probs.device)
        out[..., :num_edge_types] = probs
        return out

    @staticmethod
    def _real_edge_components(e: torch.Tensor, active: torch.Tensor, num_edge_types: int) -> list[list[int]]:
        """Connected components induced by ordinary non-zero edge classes."""

        nodes = torch.nonzero(active, as_tuple=False).flatten().tolist()
        if not nodes:
            return []
        parent = {i: i for i in nodes}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for ai, i in enumerate(nodes):
            for j in nodes[ai + 1 :]:
                edge_cls = int(e[i, j])
                if 0 < edge_cls < num_edge_types:
                    union(i, j)

        groups: dict[int, list[int]] = {}
        for i in nodes:
            groups.setdefault(find(i), []).append(i)
        return list(groups.values())

    def _conditional_real_edge_sample_batch(
        self,
        posterior: torch.Tensor,
        clean_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Sample non-zero ordinary edge classes without Python scalar syncs.

        ``posterior`` and ``clean_probs`` may have any leading shape and a final
        edge-class dimension. Class 0 is no-edge; classes 1..K-1 are ordinary
        FragDiffusion attachment classes. The fallback hierarchy is posterior ->
        clean prediction -> empirical edge marginal -> uniform.
        """

        if self.stats.num_edge_types <= 1:
            return torch.zeros(posterior.shape[:-1], dtype=torch.long, device=posterior.device)

        eps = 1e-12
        p = posterior[..., 1 : self.stats.num_edge_types].clone()
        clean = clean_probs[..., 1 : self.stats.num_edge_types]

        mass = p.sum(-1, keepdim=True)
        p = torch.where(mass > eps, p, clean)

        marginal = self.stats.edge_marginal.to(p.device)[1 : self.stats.num_edge_types]
        marginal = marginal.expand_as(p)
        mass = p.sum(-1, keepdim=True)
        p = torch.where(mass > eps, p, marginal)

        mass = p.sum(-1, keepdim=True)
        p = torch.where(mass > eps, p, torch.ones_like(p))
        p = p / p.sum(-1, keepdim=True).clamp_min(eps)
        return 1 + self._draw(p)

    def _conditional_real_edge_sample(
        self,
        posterior: torch.Tensor,
        clean_probs: torch.Tensor,
    ) -> int:
        """Scalar compatibility wrapper around the batched implementation."""

        return int(
            self._conditional_real_edge_sample_batch(
                posterior.reshape(1, -1),
                clean_probs.reshape(1, -1),
            )[0]
        )

    def _force_new_nodes_to_old_component_batch(
        self,
        e_s: torch.Tensor,
        pair_probs: torch.Tensor,
        p0e: torch.Tensor,
        x_s: torch.Tensor,
        was_delt: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> None:
        """Batched equivalent of v1's new-node connectivity repair."""

        if self.stats.num_edge_types <= 1:
            return

        resolved = (x_s < self.node_del_id) & node_mask
        old_nodes = (~was_delt) & resolved
        new_nodes = was_delt & resolved

        real_edges = (e_s > 0) & (e_s < self.stats.num_edge_types)
        already_connected_to_old = (real_edges & old_nodes[:, None, :]).any(dim=-1)
        need_repair = new_nodes & (~already_connected_to_old)

        post_scores = pair_probs[..., 1 : self.stats.num_edge_types].sum(-1)
        clean_scores = p0e[..., 1 : self.stats.num_edge_types].sum(-1)
        scores = torch.where(post_scores > 1e-12, post_scores, clean_scores)
        scores = scores.masked_fill(~old_nodes[:, None, :], float("-inf"))
        best_parent = scores.argmax(dim=-1)

        bb, ii = torch.nonzero(need_repair, as_tuple=True)
        if bb.numel() == 0:
            return
        jj = best_parent[bb, ii]
        chosen_post = pair_probs[bb, ii, jj]
        chosen_clean = p0e[bb, ii, jj]
        chosen_cls = self._conditional_real_edge_sample_batch(chosen_post, chosen_clean)
        e_s[bb, ii, jj] = chosen_cls
        e_s[bb, jj, ii] = chosen_cls

    def _force_new_nodes_to_old_component(
        self,
        e_s: torch.Tensor,
        pair_probs: torch.Tensor,
        p0e: torch.Tensor,
        x_s: torch.Tensor,
        was_delt: torch.Tensor,
    ) -> None:
        """Attach newly resolved DEL* nodes to the pre-existing component.

        This version is intentionally GPU-vectorized. The previous implementation
        iterated over node pairs and repeatedly converted CUDA scalars with
        ``int(...)``/``float(...)``, forcing device synchronization inside every
        reverse step. Here all candidate scoring and attachment-class sampling are
        performed as tensor operations and only the selected entries are written
        back to ``e_s``.
        """

        if self.stats.num_edge_types <= 1:
            return

        resolved = x_s < self.node_del_id
        old_nodes = (~was_delt) & resolved
        new_nodes = was_delt & resolved

        # The protected root is always an old resolved node in normal FragFlex
        # trajectories. Keeping the code fully tensorized avoids an ``.item()``
        # synchronization just to test whether a repair is needed.
        real_edges = (e_s > 0) & (e_s < self.stats.num_edge_types)
        already_connected_to_old = (real_edges & old_nodes.unsqueeze(0)).any(dim=1)
        need_repair = new_nodes & (~already_connected_to_old)

        # Score every possible parent by posterior mass on any real attachment.
        # If the posterior has zero real-edge mass, fall back to the denoiser's
        # clean edge distribution. Non-old columns are masked out.
        post_scores = pair_probs[..., 1 : self.stats.num_edge_types].sum(-1)
        clean_scores = p0e[..., 1 : self.stats.num_edge_types].sum(-1)
        scores = torch.where(post_scores > 1e-12, post_scores, clean_scores)
        scores = scores.masked_fill(~old_nodes.unsqueeze(0), float("-inf"))
        best_parent = scores.argmax(dim=1)

        rows = torch.arange(e_s.shape[0], device=e_s.device)
        chosen_post = pair_probs[rows, best_parent]
        chosen_clean = p0e[rows, best_parent]
        chosen_cls = self._conditional_real_edge_sample_batch(chosen_post, chosen_clean)

        repair_rows = rows[need_repair]
        repair_cols = best_parent[need_repair]
        repair_cls = chosen_cls[need_repair]
        e_s[repair_rows, repair_cols] = repair_cls
        e_s[repair_cols, repair_rows] = repair_cls

    def _force_global_connectivity(
        self,
        e_s: torch.Tensor,
        pair_probs: torch.Tensor,
        p0e: torch.Tensor,
        x_s: torch.Tensor,
    ) -> None:
        """Repair any remaining disconnected active components using model probabilities.

        New-node attachment is the primary rule. This second pass is defensive:
        ordinary old-old edges can themselves denoise to no-edge at a later step,
        so without it a trajectory that was connected at insertion time could
        become disconnected again. Each repair adds the highest-confidence
        cross-component real edge and therefore adds only the minimum number of
        edges needed to connect the active state.
        """

        if self.stats.num_edge_types <= 1:
            return
        active = x_s < self.node_del_id
        while True:
            components = self._real_edge_components(e_s, active, self.stats.num_edge_types)
            if len(components) <= 1:
                return

            comp_id: dict[int, int] = {}
            for c, nodes in enumerate(components):
                for node in nodes:
                    comp_id[node] = c

            active_nodes = torch.nonzero(active, as_tuple=False).flatten().tolist()
            best_pair: tuple[int, int] | None = None
            best_score = -1.0
            for ai, i in enumerate(active_nodes):
                for j in active_nodes[ai + 1 :]:
                    if comp_id[i] == comp_id[j]:
                        continue
                    score = float(pair_probs[i, j, 1 : self.stats.num_edge_types].sum())
                    if score <= 1e-12:
                        score = float(p0e[i, j, 1 : self.stats.num_edge_types].sum())
                    if score > best_score:
                        best_score = score
                        best_pair = (i, j)

            if best_pair is None:
                return
            i, j = best_pair
            edge_cls = self._conditional_real_edge_sample(pair_probs[i, j], p0e[i, j])
            if edge_cls <= 0:
                return
            e_s[i, j] = e_s[j, i] = edge_cls

    def reverse_categorical(
        self,
        state: DenseGraphBatch,
        node_clean_probs: torch.Tensor,
        edge_clean_probs: torch.Tensor,
        s: int,
        *,
        enforce_connectivity: bool = True,
    ) -> DenseGraphBatch:
        """Vectorized v1 reverse categorical update.

        The transition kernels and connectivity rules are unchanged. Compared
        with the reference implementation, posterior evaluation is batched over
        all graphs, each node/edge is evaluated only under the kernel it actually
        uses, and the padded state is updated in place rather than unpacked and
        repacked graph-by-graph.
        """

        if s < 0:
            raise ValueError("s must be >= 0")

        device = state.x.device
        if self._resident_device != device:
            self.to(device)
        sched = self.schedules
        t = s + 1
        p0x = self._pad_clean_node_probs(
            node_clean_probs, self.stats.num_fragments, self.stats.base_node_states
        )
        p0e = self._pad_clean_edge_probs(edge_clean_probs, self.stats.num_edge_types)

        frag_m = self.fragment_base_marginal
        root_m = self.root_base_marginal
        edge_m = self.stats.edge_marginal

        q_frag = ordinary_qstep(sched.alpha_step[t], frag_m)
        qb_frag_t = ordinary_qbar(sched.alpha_bar[t], frag_m)
        qb_frag_s = ordinary_qbar(sched.alpha_bar[s], frag_m)

        q_root = ordinary_qstep(sched.alpha_step[t], root_m)
        qb_root_t = ordinary_qbar(sched.alpha_bar[t], root_m)
        qb_root_s = ordinary_qbar(sched.alpha_bar[s], root_m)

        q_struct_x = structural_qstep(t, sched, frag_m)
        qb_struct_x_t = structural_qbar(t, sched, frag_m)
        qb_struct_x_s = structural_qbar(s, sched, frag_m)

        q_edge = ordinary_qstep(sched.alpha_step[t], edge_m)
        qb_edge_t = ordinary_qbar(sched.alpha_bar[t], edge_m)
        qb_edge_s = ordinary_qbar(sched.alpha_bar[s], edge_m)
        q_struct_e = structural_qstep(t, sched, edge_m)
        qb_struct_e_t = structural_qbar(t, sched, edge_m)
        qb_struct_e_s = structural_qbar(s, sched, edge_m)

        B, N = state.x.shape
        valid = state.node_mask
        root = state.root_mask & valid
        delt = (
            state.delt_mask & valid
            if state.delt_mask is not None
            else (state.x == self.node_delt_id) & valid
        )
        ordinary = valid & (~root) & (~delt)

        # Evaluate each node only with its applicable reverse kernel. The v1
        # implementation evaluated all three D x D posteriors for every node and
        # selected afterward. This keeps the same probabilities while reducing
        # the dominant O(D^2) posterior work.
        px_s = torch.zeros_like(p0x)
        px_s[ordinary] = posterior_from_clean_prediction(
            state.x[ordinary], p0x[ordinary], q_frag, qb_frag_s, qb_frag_t
        )
        px_s[root] = posterior_from_clean_prediction(
            state.x[root], p0x[root], q_root, qb_root_s, qb_root_t
        )
        px_s[delt] = posterior_from_clean_prediction(
            state.x[delt], p0x[delt], q_struct_x, qb_struct_x_s, qb_struct_x_t
        )

        px_valid = px_s[valid]
        px_valid[:, self.node_del_id] = 0.0
        px_valid = px_valid / px_valid.sum(-1, keepdim=True).clamp_min(1e-12)
        x_s = torch.zeros_like(state.x)
        x_s[valid] = self._draw(px_valid)

        # Upper-triangular valid pairs from every graph are processed in one
        # batch. Regular and structural edges likewise use only their relevant
        # posterior kernel.
        pair_valid = valid[:, :, None] & valid[:, None, :]
        upper = torch.triu(
            torch.ones((N, N), dtype=torch.bool, device=device), diagonal=1
        ).unsqueeze(0)
        upper_valid = pair_valid & upper
        structural_full = delt[:, :, None] | delt[:, None, :]
        structural_pair = structural_full[upper_valid]

        edge_xt = state.e[upper_valid]
        edge_p0 = p0e[upper_valid]
        M = edge_xt.numel()
        pe = torch.empty((M, self.edge_state_count), dtype=p0e.dtype, device=device)
        reg = ~structural_pair
        pe[reg] = posterior_from_clean_prediction(
            edge_xt[reg], edge_p0[reg], q_edge, qb_edge_s, qb_edge_t
        )
        pe[structural_pair] = posterior_from_clean_prediction(
            edge_xt[structural_pair],
            edge_p0[structural_pair],
            q_struct_e,
            qb_struct_e_s,
            qb_struct_e_t,
        )
        if M:
            pe[:, self.edge_del_id] = 0.0
            pe = pe / pe.sum(-1, keepdim=True).clamp_min(1e-12)
            vals = self._draw(pe)
        else:
            vals = torch.empty((0,), dtype=torch.long, device=device)

        e_s = torch.zeros_like(state.e)
        e_s[upper_valid] = vals
        e_s = e_s + e_s.transpose(1, 2)

        pair_probs = torch.zeros(
            (B, N, N, self.edge_state_count), dtype=p0e.dtype, device=device
        )
        pair_probs[upper_valid] = pe
        pair_probs = pair_probs + pair_probs.transpose(1, 2)

        if enforce_connectivity and N > 1:
            self._force_new_nodes_to_old_component_batch(
                e_s, pair_probs, p0e, x_s, delt, valid
            )

            # Keep the exact v1 final global repair. It runs once per trajectory,
            # so its small Python component search does not affect the 500-step
            # hot path. Valid nodes are prefix-packed by construction.
            if s == 0:
                counts = valid.sum(-1).tolist()
                for b, n in enumerate(counts):
                    n = int(n)
                    if n > 1:
                        self._force_global_connectivity(
                            e_s[b, :n, :n],
                            pair_probs[b, :n, :n],
                            p0e[b, :n, :n],
                            x_s[b, :n],
                        )

        new_delt = (x_s == self.node_delt_id) & valid
        next_t = torch.full_like(state.t, s)
        return DenseGraphBatch(
            x=x_s,
            e=e_s,
            node_mask=valid,
            root_mask=state.root_mask,
            t=next_t,
            delt_mask=new_delt,
        )

    def finalize(
        self,
        state: DenseGraphBatch,
        *,
        site_logits: Optional[torch.Tensor] = None,
        site_temperature: float = 1.0,
    ) -> list[FragmentGraph]:
        """Convert a t=0 state to clean fragment graphs.

        For ``assembly_mode=neural_sites`` the fragment diffusion only predicts
        binary coarse topology. Atom-level assembly endpoints are decoded by the
        neural site head. The full directed logits are retained on generated
        graphs so the assembler can perform a small valence-constrained top-k
        search without consulting a lookup table.
        """

        if not torch.all(state.t == 0):
            raise ValueError("can only finalize a t=0 state")
        if self.stats.uses_neural_sites and site_logits is None:
            raise ValueError("neural_sites finalization requires site_logits")

        out: list[FragmentGraph] = []
        atom_counts = self.stats.fragment_atom_counts.to(state.x.device)
        temp = max(float(site_temperature), 1e-6)

        for b in range(state.batch_size):
            valid = state.node_mask[b]
            x = state.x[b, valid]
            e = state.e[b][valid][:, valid]
            if torch.any(x >= self.stats.num_fragments):
                raise RuntimeError("t=0 contains non-fragment node states")
            if torch.any(e >= self.stats.num_edge_types):
                raise RuntimeError("t=0 contains structural edge states")

            sites = None
            logits_cpu = None
            if self.stats.uses_neural_sites:
                logits = site_logits[b][valid][:, valid].clone() / temp
                n, _, mmax = logits.shape
                # The source fragment identity determines the support of each
                # directed endpoint distribution. Mask padded atom positions.
                counts = atom_counts[x].clamp(min=0, max=mmax)
                ar = torch.arange(mmax, device=logits.device)
                valid_atom = ar[None, :] < counts[:, None]
                logits = logits.masked_fill(~valid_atom[:, None, :], -1e9)

                sites = torch.full((n, n), -1, dtype=torch.long, device=logits.device)
                real = e > 0
                if real.any():
                    pred = logits.argmax(-1)
                    sites[real] = pred[real]
                sites.fill_diagonal_(-1)
                logits_cpu = logits.detach().cpu()

            out.append(
                FragmentGraph(
                    x=x.detach().cpu(),
                    e=e.detach().cpu(),
                    sites=None if sites is None else sites.detach().cpu(),
                    site_logits=logits_cpu,
                )
            )
        return out

