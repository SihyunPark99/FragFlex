from __future__ import annotations

"""DiGress-style cycle auxiliary features for FragFlex.

This module intentionally implements only the ``cycles`` structural features
from cvignac/DiGress ``src/diffusion/extra_features.py``:

* node-level counts of 3-, 4-, and 5-cycles;
* graph-level counts of 3-, 4-, 5-, and 6-cycles;
* graph-level normalized node count n / max_n_nodes.

As in DiGress, cycle counts are divided by 10 and clipped at 1.  No spectral,
molecular, degree, bridge, cyclomatic-number, or edge-level features are added.
The features are computed from the *current noisy graph* at every denoiser call.
"""

import torch
from torch import nn

from fragflex.graph import DenseGraphBatch


def batch_trace(x: torch.Tensor) -> torch.Tensor:
    """Trace over the last two dimensions, batched over leading dimensions."""

    return torch.diagonal(x, dim1=-2, dim2=-1).sum(dim=-1)


def batch_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Extract the diagonal from the last two dimensions."""

    return torch.diagonal(x, dim1=-2, dim2=-1)


class KNodeCycles:
    """Cycle-count formulas used by DiGress.

    The formulas are copied algebraically from DiGress' ``KNodeCycles``.  The
    adjacency is expected to be a dense, simple, undirected 0/1 matrix.
    """

    def __call__(self, adj_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = adj_matrix.float()
        d = a.sum(dim=-1)

        a2 = a @ a
        a3 = a2 @ a
        a4 = a3 @ a
        a5 = a4 @ a
        a6 = a5 @ a

        # 3-cycles: per-node membership and graph count.
        c3_raw = batch_diagonal(a3)
        k3x = (c3_raw / 2).unsqueeze(-1).float()
        k3y = (c3_raw.sum(dim=-1) / 6).unsqueeze(-1).float()

        # 4-cycles.
        diag_a4 = batch_diagonal(a4)
        c4_raw = diag_a4 - d * (d - 1) - (a @ d.unsqueeze(-1)).sum(dim=-1)
        k4x = (c4_raw / 2).unsqueeze(-1).float()
        k4y = (c4_raw.sum(dim=-1) / 8).unsqueeze(-1).float()

        # 5-cycles.
        triangles = c3_raw
        diag_a5 = batch_diagonal(a5)
        c5_raw = (
            diag_a5
            - 2 * triangles * d
            - (a @ triangles.unsqueeze(-1)).sum(dim=-1)
            + triangles
        )
        k5x = (c5_raw / 2).unsqueeze(-1).float()
        k5y = (c5_raw.sum(dim=-1) / 10).unsqueeze(-1).float()

        # 6-cycles: DiGress uses only a graph-level count.
        term_1 = batch_trace(a6)
        term_2 = batch_trace(a3**2)
        term_3 = torch.sum(a * a2.pow(2), dim=(-2, -1))
        d_t4 = batch_diagonal(a2)
        a_4_t = batch_diagonal(a4)
        term_4 = (d_t4 * a_4_t).sum(dim=-1)
        term_5 = batch_trace(a4)
        term_6 = batch_trace(a3)
        term_7 = batch_diagonal(a2).pow(3).sum(-1)
        term_8 = torch.sum(a3, dim=(-2, -1))
        term_9 = batch_diagonal(a2).pow(2).sum(-1)
        term_10 = batch_trace(a2)
        c6_raw = (
            term_1
            - 3 * term_2
            + 9 * term_3
            - 6 * term_4
            + 6 * term_5
            - 4 * term_6
            + 4 * term_7
            + 3 * term_8
            - 12 * term_9
            + 4 * term_10
        )
        k6y = (c6_raw / 12).unsqueeze(-1).float()

        # Match DiGress' numerical sanity checks while tolerating tiny floating
        # roundoff by clipping to zero before feature normalization.
        kcycles_x = torch.cat([k3x, k4x, k5x], dim=-1).clamp_min(0)
        kcycles_y = torch.cat([k3y, k4y, k5y, k6y], dim=-1).clamp_min(0)
        return kcycles_x, kcycles_y


class DiGressCycleFeatures(nn.Module):
    """Compute exactly the auxiliary feature set used by DiGress ``cycles`` mode.

    Returns
    -------
    node_features:
        ``[B, N, 3]`` containing normalized/clipped C3, C4, C5 membership
        counts for each active fragment node.
    global_features:
        ``[B, 5]`` containing ``n/max_n_nodes`` followed by normalized/clipped
        graph-level C3, C4, C5, C6 counts.
    """

    node_dim = 3
    global_dim = 5

    def __init__(self, *, max_n_nodes: int, num_edge_types: int):
        super().__init__()
        self.max_n_nodes = int(max_n_nodes)
        self.num_edge_types = int(num_edge_types)
        self.kcycles = KNodeCycles()

    def forward(self, graph: DenseGraphBatch) -> tuple[torch.Tensor, torch.Tensor]:
        # DiGress defines adjacency as the sum over all real edge categories.
        # FragFlex has two additional structural edge states (DEL and DEL*), so
        # only the original real edge classes 1..num_edge_types-1 are included.
        active_pair = graph.node_mask[:, :, None] & graph.node_mask[:, None, :]
        adjacency = (
            (graph.e > 0)
            & (graph.e < self.num_edge_types)
            & active_pair
        ).to(dtype=torch.float32)

        x_cycles, y_cycles = self.kcycles(adjacency)
        x_cycles = x_cycles.type_as(adjacency) * graph.node_mask.unsqueeze(-1)

        # Exact DiGress normalization/clipping for cycle features.
        x_cycles = (x_cycles / 10).clamp(max=1)
        y_cycles = (y_cycles / 10).clamp(max=1)

        n = graph.node_mask.sum(dim=1, keepdim=True).to(adjacency.dtype)
        n = n / max(self.max_n_nodes, 1)
        global_features = torch.cat([n, y_cycles], dim=-1)
        return x_cycles, global_features
