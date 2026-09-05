from __future__ import annotations

import math

import torch
from torch import nn

from fragflex.graph import DenseGraphBatch

from .digress_cycles import DiGressCycleFeatures


class SinusoidalTimeEmbedding(nn.Module):
    """Lookup-table version of the original sinusoidal time embedding.

    FragFlex timesteps are discrete integers in ``[0, max_steps]``. The original
    implementation recomputed sin/cos for every denoiser and delta-model call.
    Sampling invokes both networks hundreds of times, so the exact same values
    are precomputed once and stored as a non-persistent buffer.
    """

    def __init__(self, dim: int, max_steps: int):
        super().__init__()
        self.dim = dim
        self.max_steps = max_steps
        half = max(1, dim // 2)
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half).float() / max(half - 1, 1)
        )
        t = torch.arange(max_steps + 1, dtype=torch.float32).unsqueeze(-1)
        x = t / max(max_steps, 1)
        y = x * freq.unsqueeze(0)
        table = torch.cat([torch.sin(y), torch.cos(y)], dim=-1)
        if table.shape[-1] < dim:
            table = torch.nn.functional.pad(table, (0, dim - table.shape[-1]))
        self.register_buffer("table", table[..., :dim].contiguous(), persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.table[t.long()]


class EdgeAwareBlock(nn.Module):
    """Dense graph-transformer block with edge-conditioned attention bias."""

    def __init__(self, d_model: int, d_edge: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.attn_scale = self.d_head ** -0.5

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.edge_bias = nn.Linear(d_edge, n_heads)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
        )

        self.edge_norm1 = nn.LayerNorm(d_edge)
        self.edge_norm2 = nn.LayerNorm(d_edge)
        self.node_to_edge_i = nn.Linear(d_model, d_edge)
        self.node_to_edge_j = nn.Linear(d_model, d_edge)
        self.edge_ff = nn.Sequential(
            nn.Linear(d_edge, ff_mult * d_edge),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_edge, d_edge),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = h.shape
        H, dh = self.n_heads, self.d_head

        q = self.q(h).view(B, N, H, dh).transpose(1, 2)  # B,H,N,dh
        k = self.k(h).view(B, N, H, dh).transpose(1, 2)
        v = self.v(h).view(B, N, H, dh).transpose(1, 2)

        # torch.matmul dispatches to the optimized batched-GEMM path more
        # reliably than the equivalent einsum for these small dense graphs.
        score = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        score = score + self.edge_bias(edge).permute(0, 3, 1, 2)

        pair_mask = node_mask[:, None, :, None] & node_mask[:, None, None, :]
        score = score.masked_fill(~pair_mask, -1e9)
        attn = torch.softmax(score, dim=-1)
        attn = attn * pair_mask.to(attn.dtype)
        msg = torch.matmul(attn, v)
        msg = msg.transpose(1, 2).contiguous().view(B, N, self.d_model)
        h = self.norm1(h + self.drop(self.out(msg)))
        h = self.norm2(h + self.drop(self.ff(h)))
        h = h * node_mask.unsqueeze(-1)

        pair = node_mask[:, :, None] & node_mask[:, None, :]
        edge_update = self.node_to_edge_i(h)[:, :, None, :] + self.node_to_edge_j(h)[:, None, :, :]
        edge = self.edge_norm1(edge + self.drop(edge_update))
        edge = self.edge_norm2(edge + self.drop(self.edge_ff(edge)))
        edge = 0.5 * (edge + edge.transpose(1, 2))
        edge = edge * pair.unsqueeze(-1)
        return h, edge


class DenseGraphEncoder(nn.Module):
    def __init__(
        self,
        *,
        node_states: int,
        edge_states: int,
        max_steps: int,
        d_model: int,
        d_edge: int,
        n_layers: int,
        n_heads: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
        use_digress_cycles: bool = False,
        max_nodes: int = 32,
        num_real_edge_types: int = 0,
    ):
        super().__init__()
        self.node_emb = nn.Embedding(node_states, d_model)
        self.edge_emb = nn.Embedding(edge_states, d_edge)
        self.root_emb = nn.Embedding(2, d_model)
        self.time = SinusoidalTimeEmbedding(d_model, max_steps)
        self.time_to_edge = nn.Linear(d_model, d_edge)

        # Faithful DiGress ``cycles`` feature content only. DiGress concatenates
        # these continuous node/global features to its native X/y inputs.
        # FragFlex uses categorical embeddings rather than one-hot X/y tensors,
        # so linear projections are the algebraically equivalent minimal adapter
        # into the existing node/edge embedding spaces. No extra edge features
        # are introduced, matching DiGress cycles mode.
        self.use_digress_cycles = bool(use_digress_cycles)
        if self.use_digress_cycles:
            if num_real_edge_types < 1:
                raise ValueError("num_real_edge_types must be >= 1 with DiGress cycles")
            self.cycle_features = DiGressCycleFeatures(
                max_n_nodes=max_nodes,
                num_edge_types=num_real_edge_types,
            )
            self.cycle_node_proj = nn.Linear(DiGressCycleFeatures.node_dim, d_model, bias=False)
            self.cycle_global_to_node = nn.Linear(
                DiGressCycleFeatures.global_dim, d_model, bias=False
            )
            self.cycle_global_to_edge = nn.Linear(
                DiGressCycleFeatures.global_dim, d_edge, bias=False
            )
        else:
            self.cycle_features = None
            self.cycle_node_proj = None
            self.cycle_global_to_node = None
            self.cycle_global_to_edge = None

        self.blocks = nn.ModuleList(
            [EdgeAwareBlock(d_model, d_edge, n_heads, ff_mult, dropout) for _ in range(n_layers)]
        )
        self.node_norm = nn.LayerNorm(d_model)
        self.edge_norm = nn.LayerNorm(d_edge)

    def forward(self, graph: DenseGraphBatch) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.node_emb(graph.x)
        edge = self.edge_emb(graph.e)
        te = self.time(graph.t)
        h = h + te[:, None, :] + self.root_emb(graph.root_mask.long())
        edge = edge + self.time_to_edge(te)[:, None, None, :]

        if self.cycle_features is not None:
            x_cycles, y_cycles = self.cycle_features(graph)
            h = (
                h
                + self.cycle_node_proj(x_cycles)
                + self.cycle_global_to_node(y_cycles)[:, None, :]
            )
            edge = edge + self.cycle_global_to_edge(y_cycles)[:, None, None, :]

        pair = graph.node_mask[:, :, None] & graph.node_mask[:, None, :]
        h = h * graph.node_mask.unsqueeze(-1)
        edge = edge * pair.unsqueeze(-1)
        for block in self.blocks:
            h, edge = block(h, edge, graph.node_mask)
        return self.node_norm(h), self.edge_norm(edge)


class FragFlexDenoiser(nn.Module):
    def __init__(
        self,
        *,
        node_states: int,
        edge_states: int,
        num_fragments: int,
        num_edge_types: int,
        max_steps: int,
        d_model: int,
        d_edge: int,
        n_layers: int,
        n_heads: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
        max_fragment_atoms: int = 0,
        site_d_hidden: int = 128,
        use_digress_cycles: bool = False,
        max_nodes: int = 32,
    ):
        super().__init__()
        self.encoder = DenseGraphEncoder(
            node_states=node_states,
            edge_states=edge_states,
            max_steps=max_steps,
            d_model=d_model,
            d_edge=d_edge,
            n_layers=n_layers,
            n_heads=n_heads,
            ff_mult=ff_mult,
            dropout=dropout,
            use_digress_cycles=use_digress_cycles,
            max_nodes=max_nodes,
            num_real_edge_types=num_edge_types,
        )
        self.node_head = nn.Linear(d_model, num_fragments)
        self.edge_head = nn.Linear(d_edge, num_edge_types)
        self.max_fragment_atoms = int(max_fragment_atoms)

        if self.max_fragment_atoms > 0:
            self.site_src = nn.Linear(d_model, site_d_hidden)
            self.site_dst = nn.Linear(d_model, site_d_hidden)
            self.site_edge = nn.Linear(d_edge, site_d_hidden)
            self.site_head = nn.Sequential(
                nn.GELU(),
                nn.Linear(site_d_hidden, self.max_fragment_atoms),
            )
        else:
            self.site_src = None
            self.site_dst = None
            self.site_edge = None
            self.site_head = None

    def forward(
        self,
        graph: DenseGraphBatch,
        *,
        compute_sites: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Predict clean node/edge states and, optionally, neural endpoints.

        ``compute_sites=False`` is sampling-only optimization. Site logits do not
        enter the diffusion posterior at any timestep; v1 uses only the logits
        produced at t=1 for final assembly. Training keeps the default True.
        """

        h, edge = self.encoder(graph)
        node_logits = self.node_head(h)
        edge_logits = self.edge_head(edge)
        edge_logits = 0.5 * (edge_logits + edge_logits.transpose(1, 2))

        site_logits = None
        if compute_sites and self.site_head is not None:
            z = (
                self.site_src(h)[:, :, None, :]
                + self.site_dst(h)[:, None, :, :]
                + self.site_edge(edge)
            )
            site_logits = self.site_head(z)
            pair = graph.node_mask[:, :, None] & graph.node_mask[:, None, :]
            site_logits = site_logits.masked_fill(~pair.unsqueeze(-1), -1e9)
        return node_logits, edge_logits, site_logits


class DeltaCountModel(nn.Module):
    def __init__(
        self,
        *,
        node_states: int,
        edge_states: int,
        max_steps: int,
        max_delta: int,
        d_model: int,
        d_edge: int,
        n_layers: int,
        n_heads: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = DenseGraphEncoder(
            node_states=node_states,
            edge_states=edge_states,
            max_steps=max_steps,
            d_model=d_model,
            d_edge=d_edge,
            n_layers=n_layers,
            n_heads=n_heads,
            ff_mult=ff_mult,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, max_delta + 1),
        )

    def forward(self, graph: DenseGraphBatch) -> torch.Tensor:
        h, _ = self.encoder(graph)
        w = graph.node_mask.float().unsqueeze(-1)
        pooled = (h * w).sum(1) / w.sum(1).clamp_min(1.0)
        return self.head(pooled)
