from __future__ import annotations

import torch

from fragflex.graph import DenseGraphBatch
from fragflex.models.digress_cycles import DiGressCycleFeatures, KNodeCycles
from fragflex.models.graph_transformer import DenseGraphEncoder


def _ring_adjacency(n: int) -> torch.Tensor:
    a = torch.zeros((1, n, n), dtype=torch.float32)
    for i in range(n):
        j = (i + 1) % n
        a[0, i, j] = 1
        a[0, j, i] = 1
    return a


def test_kcycles_exact_simple_rings() -> None:
    counter = KNodeCycles()

    x3, y3 = counter(_ring_adjacency(3))
    assert torch.allclose(x3[0, :, 0], torch.ones(3))
    assert torch.equal(x3[0, :, 1:], torch.zeros((3, 2)))
    assert torch.allclose(y3, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))

    x4, y4 = counter(_ring_adjacency(4))
    assert torch.allclose(x4[0, :, 1], torch.ones(4))
    assert torch.allclose(y4, torch.tensor([[0.0, 1.0, 0.0, 0.0]]))

    x5, y5 = counter(_ring_adjacency(5))
    assert torch.allclose(x5[0, :, 2], torch.ones(5))
    assert torch.allclose(y5, torch.tensor([[0.0, 0.0, 1.0, 0.0]]))

    x6, y6 = counter(_ring_adjacency(6))
    assert torch.equal(x6, torch.zeros_like(x6))
    assert torch.allclose(y6, torch.tensor([[0.0, 0.0, 0.0, 1.0]]))


def test_digress_cycle_normalization_and_structural_edge_exclusion() -> None:
    # num_edge_types=2 means 0=no edge, 1=real coarse edge; ids >=2 are
    # FragFlex structural DEL/DEL* states and must not be counted as adjacency.
    e = torch.zeros((1, 4, 4), dtype=torch.long)
    e[0, 0, 1] = e[0, 1, 0] = 1
    e[0, 1, 2] = e[0, 2, 1] = 1
    e[0, 2, 0] = e[0, 0, 2] = 1  # triangle 0-1-2
    e[0, 2, 3] = e[0, 3, 2] = 2  # structural state, not a real edge

    graph = DenseGraphBatch(
        x=torch.zeros((1, 4), dtype=torch.long),
        e=e,
        node_mask=torch.tensor([[True, True, True, True]]),
        root_mask=torch.tensor([[True, False, False, False]]),
        t=torch.tensor([10]),
        delt_mask=torch.zeros((1, 4), dtype=torch.bool),
    )
    feat = DiGressCycleFeatures(max_n_nodes=32, num_edge_types=2)
    node, glob = feat(graph)

    # DiGress divides cycle counts by 10.
    assert torch.allclose(node[0, :3, 0], torch.full((3,), 0.1))
    assert node[0, 3, 0].item() == 0.0
    assert torch.allclose(glob[0], torch.tensor([4 / 32, 0.1, 0.0, 0.0, 0.0]))


def test_encoder_accepts_digress_cycles() -> None:
    graph = DenseGraphBatch(
        x=torch.tensor([[0, 1, 2]], dtype=torch.long),
        e=torch.tensor([[[0, 1, 1], [1, 0, 1], [1, 1, 0]]], dtype=torch.long),
        node_mask=torch.tensor([[True, True, True]]),
        root_mask=torch.tensor([[True, False, False]]),
        t=torch.tensor([5]),
        delt_mask=torch.zeros((1, 3), dtype=torch.bool),
    )
    enc = DenseGraphEncoder(
        node_states=6,
        edge_states=4,
        max_steps=10,
        d_model=32,
        d_edge=16,
        n_layers=1,
        n_heads=4,
        use_digress_cycles=True,
        max_nodes=32,
        num_real_edge_types=2,
    )
    h, edge = enc(graph)
    assert h.shape == (1, 3, 32)
    assert edge.shape == (1, 3, 3, 16)
    assert torch.isfinite(h).all()
    assert torch.isfinite(edge).all()
