from __future__ import annotations

import pandas as pd
import torch

from fragflex.chemistry import AttachmentLibrary, FragmentLibrary, FragDiffusionAssembler
from fragflex.config import FragFlexConfig
from fragflex.data.stats import FragFlexStats
from fragflex.diffusion.one_atom import OneAtomFragDiffusion
from fragflex.diffusion.posterior import structural_matrices
from fragflex.graph import FragmentGraph
from fragflex.models import FragFlexModel


def stats() -> FragFlexStats:
    return FragFlexStats(
        num_fragments=3,
        num_atom_latents=2,
        num_edge_types=2,
        fragment_marginal=torch.tensor([0.4, 0.35, 0.25]),
        atom_marginal=torch.tensor([0.8, 0.2]),
        edge_marginal=torch.tensor([0.7, 0.3]),
        atom_labels=["C", "N"],
    )


def cfg() -> FragFlexConfig:
    return FragFlexConfig(
        diffusion_steps=8,
        max_nodes=6,
        max_delta_per_step=3,
        d_model=32,
        d_edge=16,
        d_time=16,
        n_layers=2,
        n_heads=4,
        delta_d_model=32,
        delta_d_edge=16,
        delta_n_layers=1,
        delta_n_heads=4,
    )


def graph3() -> FragmentGraph:
    return FragmentGraph(
        x=torch.tensor([0, 1, 2]),
        e=torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
    )


def test_terminal_is_exactly_one_atom_token():
    d = OneAtomFragDiffusion(cfg(), stats())
    g = graph3()
    c = d.corrupt([g], timesteps=torch.tensor([d.T]), root_indices=torch.tensor([1]))
    assert int(c.noisy.node_mask.sum()) == 1
    x = int(c.noisy.x[0, 0])
    assert d.stats.num_fragments <= x < d.stats.base_node_states
    assert bool(c.noisy.root_mask[0, 0])


def test_event_index_convention_is_explicit():
    d = OneAtomFragDiffusion(cfg(), stats())
    e = d.sample_event_times((1000,), "cpu")
    assert int(e.min()) >= 1
    assert int(e.max()) <= d.T


def test_auxiliary_model_hides_boundary_delt_nodes():
    d = OneAtomFragDiffusion(cfg(), stats())
    g = graph3()

    # root=0 is protected; node 1 is exactly at the boundary; node 2 is future.
    d.sample_event_times = lambda shape, device: torch.tensor([d.T, 2, 4], device=device)  # type: ignore[method-assign]
    c = d.corrupt([g], timesteps=torch.tensor([2]), root_indices=torch.tensor([0]))
    assert int(c.delta_target[0]) == 1
    assert int(c.noisy.node_mask.sum()) == 3
    assert int(c.delta_input.node_mask.sum()) == 2
    assert int(c.noisy.delt_mask.sum()) == 1


def test_delstar_forward_matrix_semantics():
    marginal = torch.tensor([0.6, 0.4])
    A, B, C, D = structural_matrices(marginal)
    # normal -> DEL* under C; DEL* -> DEL under A/B; everything -> DEL under D.
    assert C[0, -1] == 1
    assert A[-1, -2] == 1
    assert B[-1, -2] == 1
    assert torch.all(D[:, -2] == 1)


def test_loss_and_untrained_sampling_execute_end_to_end():
    model = FragFlexModel(cfg(), stats())
    g = graph3()
    out = model.training_loss([g, g])
    assert torch.isfinite(out.loss)
    model.config.sample_delta = False
    samples, trace = model.sample(2, return_trace=True)
    assert len(samples) == 2
    assert all(torch.all(s.x < model.stats.num_fragments) for s in samples)
    # Reverse graph size is monotone non-decreasing because FragFlex has no
    # forward insertion / reverse activation-time removal branch.
    sizes = torch.stack([x.node_mask.sum(-1) for x in trace], dim=0)
    assert torch.all(sizes[1:] >= sizes[:-1])


def test_fragdiffusion_attachment_lookup_and_assembly():
    fragments = FragmentLibrary({0: "C", 1: "N"})
    table = pd.DataFrame(
        [
            {
                "fragment_index_1": 0,
                "fragment_index_2": 1,
                "atom_idx_1": 0,
                "atom_idx_2": 0,
                "edge_id": 0,
            }
        ]
    )
    attachments = AttachmentLibrary(fragments, table)
    assembler = FragDiffusionAssembler(fragments, attachments)
    g = FragmentGraph(torch.tensor([0, 1]), torch.tensor([[0, 1], [1, 0]]))
    smiles, result = assembler.to_smiles(g.x, g.e)
    assert smiles in {"CN", "NC"}
    assert result.skipped_edges == 0

    bad = FragmentGraph(torch.tensor([0, 1]), torch.tensor([[0, 2], [2, 0]]))
    _, result_bad = assembler.to_smiles(bad.x, bad.e)
    assert result_bad.skipped_edges == 1



def test_constrained_assembler_repairs_valence_conflicting_mode():
    from fragflex.chemistry import ConstrainedFragDiffusionAssembler

    # The predicted mode attaches to pyrrolic [nH] (atom 3), which is already
    # valence-saturated. Mode 1 attaches to an aromatic carbon and is valid.
    fragments = FragmentLibrary({0: "C", 1: "c1cc[nH]c1"})
    table = pd.DataFrame(
        [
            {
                "fragment_index_1": 0,
                "fragment_index_2": 1,
                "atom_idx_1": 0,
                "atom_idx_2": 3,
                "edge_id": 0,
            },
            {
                "fragment_index_1": 0,
                "fragment_index_2": 1,
                "atom_idx_1": 0,
                "atom_idx_2": 0,
                "edge_id": 1,
            },
        ]
    )
    attachments = AttachmentLibrary(fragments, table)
    baseline = FragDiffusionAssembler(fragments, attachments)
    constrained = ConstrainedFragDiffusionAssembler(fragments, attachments, beam_size=16)
    x = torch.tensor([1, 0])
    e = torch.tensor([[0, 1], [1, 0]])  # predicted lookup mode 0

    smi_a, _ = baseline.to_smiles(x, e)
    smi_b, result_b = constrained.to_smiles(x, e)
    assert smi_a is None
    assert smi_b is not None
    assert result_b.connected
    assert result_b.repaired_mode_edges == 1
    assert result_b.valence_rejected_options >= 1


def neural_stats() -> FragFlexStats:
    return FragFlexStats(
        num_fragments=3,
        num_atom_latents=2,
        num_edge_types=2,  # 0=no edge, 1=linked
        fragment_marginal=torch.tensor([0.4, 0.35, 0.25]),
        atom_marginal=torch.tensor([0.8, 0.2]),
        edge_marginal=torch.tensor([0.7, 0.3]),
        atom_labels=["C", "N"],
        assembly_mode="neural_sites",
        max_fragment_atoms=2,
        fragment_atom_counts=torch.tensor([1, 1, 1]),
    )


def neural_graph3() -> FragmentGraph:
    return FragmentGraph(
        x=torch.tensor([0, 1, 2]),
        e=torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
        sites=torch.tensor([[-1, 0, -1], [0, -1, 0], [-1, 0, -1]]),
    )


def test_neural_site_head_trains_jointly_with_diffusion():
    model = FragFlexModel(cfg(), neural_stats())
    g = neural_graph3()
    out = model.training_loss([g, g])
    assert torch.isfinite(out.loss)
    assert torch.isfinite(out.site_loss)
    assert torch.isfinite(out.site_accuracy)
    assert torch.isfinite(out.site_top2_accuracy)
    assert torch.isfinite(out.site_top4_accuracy)
    assert out.site_top2_accuracy >= out.site_accuracy
    assert out.site_top4_accuracy >= out.site_top2_accuracy

    model.config.sample_delta = False
    generated = model.sample(2)
    assert len(generated) == 2
    for sample in generated:
        assert sample.sites is not None
        assert sample.site_logits is not None
        assert sample.site_logits.shape[:2] == (sample.num_nodes, sample.num_nodes)


def test_neural_site_assembler_is_lookup_free():
    from fragflex.chemistry import NeuralSiteAssembler

    fragments = FragmentLibrary({0: "C", 1: "N"})
    assembler = NeuralSiteAssembler(fragments, beam_size=8, site_topk=1)
    logits = torch.zeros((2, 2, 1), dtype=torch.float32)
    graph = FragmentGraph(
        x=torch.tensor([0, 1]),
        e=torch.tensor([[0, 1], [1, 0]]),
        sites=torch.tensor([[-1, 0], [0, -1]]),
        site_logits=logits,
    )
    smiles, result = assembler.to_smiles_graph(graph)
    assert smiles in {"CN", "NC"}
    assert result.connected
    assert result.lookup_missing_edges == 0
    assert result.search_method == "neural_site_topk_beam"


def test_neural_site_assembler_replaces_explicit_h_without_lookup():
    from fragflex.chemistry import NeuralSiteAssembler

    fragments = FragmentLibrary({0: "c1cc[nH]c1", 1: "C"})
    assembler = NeuralSiteAssembler(fragments, beam_size=16, site_topk=2)

    logits = torch.full((2, 2, 5), -20.0)
    # ring -> methyl: top-1 chooses pyrrolic [nH] atom 3. The current
    # assembler replaces the explicit H before forming the external bond.
    logits[0, 1, 3] = 10.0
    logits[0, 1, 0] = 8.0
    # methyl -> ring has only local atom 0.
    logits[1, 0, 0] = 10.0
    graph = FragmentGraph(
        x=torch.tensor([0, 1]),
        e=torch.tensor([[0, 1], [1, 0]]),
        sites=torch.tensor([[-1, 3], [0, -1]]),
        site_logits=logits,
    )
    smiles, result = assembler.to_smiles_graph(graph)
    assert smiles is not None
    assert result.connected
    assert result.attachment_h_replacements >= 1
    assert result.lookup_missing_edges == 0


def test_sampling_site_head_skip_preserves_node_and_edge_predictions():
    model = FragFlexModel(cfg(), neural_stats())
    model.eval()
    state = model.diffusion.sample_limit(2, "cpu")
    state.t.fill_(model.config.diffusion_steps)
    with torch.inference_mode():
        n1, e1, s1 = model.denoiser(state, compute_sites=True)
        n2, e2, s2 = model.denoiser(state, compute_sites=False)
    assert s1 is not None
    assert s2 is None
    assert torch.equal(n1, n2)
    assert torch.equal(e1, e2)


def test_batched_insert_delt_matches_v1_layout_semantics():
    d = OneAtomFragDiffusion(cfg(), stats())
    state = d.sample_limit(2, "cpu")
    out = d.insert_delt(state, torch.tensor([2, 1]))
    assert out.x.shape == (2, 3)
    assert torch.equal(out.node_mask[0], torch.tensor([True, True, True]))
    assert torch.equal(out.node_mask[1], torch.tensor([True, True, False]))
    assert torch.equal(out.x[0], torch.tensor([int(state.x[0, 0]), d.node_delt_id, d.node_delt_id]))
    assert torch.equal(out.x[1, :2], torch.tensor([int(state.x[1, 0]), d.node_delt_id]))
    assert bool(out.delt_mask[0, 1]) and bool(out.delt_mask[0, 2])
    assert bool(out.delt_mask[1, 1])
    # Every valid edge touching a newly inserted node starts structural.
    assert int(out.e[0, 0, 1]) == d.edge_delt_id
    assert int(out.e[0, 1, 2]) == d.edge_delt_id
    assert int(out.e[1, 0, 1]) == d.edge_delt_id
    assert torch.all(torch.diagonal(out.e, dim1=1, dim2=2) == 0)


def test_event_free_steps_are_known_without_tensor_scalar_reads():
    d = OneAtomFragDiffusion(cfg(), stats())
    assert len(d.event_active) == d.T + 1
    for t, active in enumerate(d.event_active):
        assert active == bool(float(d.schedules.event_pmf[t]) > 0.0)


def test_sampling_logistic_peak_is_expressed_in_reverse_steps():
    c = cfg()
    c.zeta_schedule = "sampling_logistic"
    c.zeta_sampling_peak_step = 2
    d = OneAtomFragDiffusion(c, stats())
    # T=8: two reverse iterations after t=8 is forward t=6.
    assert int(d.schedules.event_pmf.argmax()) == 6
    assert float(d.schedules.event_pmf[0]) == 0.0
    assert float(d.schedules.event_pmf[d.T]) == 0.0


def test_sampling_exponential_is_front_loaded_in_reverse_time():
    c = cfg()
    c.zeta_schedule = "sampling_exponential"
    c.zeta_sampling_tau = 0.25
    d = OneAtomFragDiffusion(c, stats())
    # t=T is reserved for the one-root terminal state, so T-1 is the first
    # legal insertion boundary and must carry the largest event mass.
    assert int(d.schedules.event_pmf.argmax()) == d.T - 1


def test_old_checkpoint_config_dict_uses_legacy_schedule_by_default():
    old = cfg().to_dict()
    for key in [
        "zeta_schedule",
        "zeta_sampling_peak_step",
        "zeta_sampling_tau",
        "zeta_event_rel_threshold",
    ]:
        old.pop(key)
    restored = FragFlexConfig.from_dict(old)
    assert restored.zeta_schedule == "legacy_logistic"


def test_fragment_terminal_can_start_from_any_fragment_token():
    c = cfg()
    c.root_terminal_prior = "fragment"
    d = OneAtomFragDiffusion(c, stats())
    g = graph3()

    corr = d.corrupt([g], timesteps=torch.tensor([d.T]), root_indices=torch.tensor([1]))
    assert int(corr.noisy.node_mask.sum()) == 1
    assert 0 <= int(corr.noisy.x[0, 0]) < d.stats.num_fragments
    assert bool(corr.noisy.root_mask[0, 0])

    limit = d.sample_limit(256, "cpu")
    assert torch.all(limit.x >= 0)
    assert torch.all(limit.x < d.stats.num_fragments)
    assert torch.all(limit.root_mask)


def test_atom_terminal_remains_backward_compatible():
    c = cfg()
    c.root_terminal_prior = "atom"
    d = OneAtomFragDiffusion(c, stats())
    limit = d.sample_limit(128, "cpu")
    assert torch.all(limit.x >= d.stats.num_fragments)
    assert torch.all(limit.x < d.stats.base_node_states)


def test_fragment_root_and_sampling_exponential_are_composable():
    c = cfg()
    c.root_terminal_prior = "fragment"
    c.zeta_schedule = "sampling_exponential"
    c.zeta_sampling_tau = 0.10
    c.zeta_event_rel_threshold = 0.001
    d = OneAtomFragDiffusion(c, stats())

    # The root support is the fragment vocabulary.
    limit = d.sample_limit(64, "cpu")
    assert torch.all(limit.x < d.stats.num_fragments)

    # Peak-0 means the first legal reverse insertion boundary has maximal mass.
    event = d.schedules.event_pmf
    assert int(torch.argmax(event).item()) == d.T - 1
    assert float(event[d.T - 1]) > float(event[d.T - 2])


def test_invalid_root_terminal_prior_is_rejected():
    c = cfg()
    c.root_terminal_prior = "not-a-prior"
    try:
        OneAtomFragDiffusion(c, stats())
    except ValueError as exc:
        assert "root_terminal_prior" in str(exc)
    else:
        raise AssertionError("invalid root_terminal_prior should fail")


def test_datamodule_prefers_precomputed_split_indices(tmp_path):
    import numpy as np

    from fragflex.data import FragmentGraphDataset, FragFlexDataModule

    graphs = [
        FragmentGraph(x=torch.tensor([0]), e=torch.zeros((1, 1), dtype=torch.long))
        for _ in range(5)
    ]
    FragmentGraphDataset(graphs).save(tmp_path / "graphs.pt")
    torch.save(stats().state_dict(), tmp_path / "stats.pt")
    np.savez(
        tmp_path / "split_idxs.npz",
        train_idxs=np.array([4, 0, 2]),
        val_idxs=np.array([1]),
        test_idxs=np.array([3]),
    )

    dm = FragFlexDataModule(tmp_path, batch_size=2)
    dm.setup("fit")
    assert list(dm.train_dataset.indices) == [4, 0, 2]
    assert list(dm.val_dataset.indices) == [1]
    assert list(dm.test_dataset.indices) == [3]
