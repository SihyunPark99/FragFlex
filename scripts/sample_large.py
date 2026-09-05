#!/usr/bin/env python3
"""Large-scale FragFlex sampling with streaming SMILES output.

This version exposes the same neural-assembly controls as scripts.sample:
explicit-H replacement, static site masking, stochastic selection among
sanitized connected beam products, and the maximum number of valid products
considered.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from rdkit import RDLogger

from fragflex.chemistry import (
    AttachmentLibrary,
    ConstrainedFragDiffusionAssembler,
    FragDiffusionAssembler,
    FragmentLibrary,
    NeuralSiteAssembler,
)
from fragflex.models import FragFlexModel


_NEURAL_WORKER_ASSEMBLER = None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _init_neural_assembly_worker(
    fragment_smiles: dict[int, str],
    beam_size: int,
    site_topk: int,
    require_connected: bool,
    replace_explicit_h: bool,
    static_site_mask: bool,
    assembly_selection: str,
    assembly_temperature: float,
    max_valid_products: int,
    show_rdkit_errors: bool,
    base_seed: int,
) -> None:
    """Create one process-local NeuralSiteAssembler for CPU-parallel assembly."""
    global _NEURAL_WORKER_ASSEMBLER

    torch.set_num_threads(1)

    # Each worker gets a different reproducible RNG stream.
    identity = mp.current_process()._identity
    worker_idx = int(identity[0]) if identity else 0
    worker_seed = int(base_seed) + 1_000_003 * worker_idx

    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))
    torch.manual_seed(worker_seed)

    if not show_rdkit_errors:
        RDLogger.DisableLog("rdApp.error")

    worker_fragments = FragmentLibrary(
        dict(fragment_smiles)
    )

    _NEURAL_WORKER_ASSEMBLER = NeuralSiteAssembler(
        worker_fragments,
        beam_size=beam_size,
        site_topk=site_topk,
        require_connected=require_connected,
        replace_explicit_h=replace_explicit_h,
        static_site_mask=static_site_mask,
        assembly_selection=assembly_selection,
        assembly_temperature=assembly_temperature,
        max_valid_products=max_valid_products,
    )


def _assemble_neural_worker(graph) -> str | None:
    if _NEURAL_WORKER_ASSEMBLER is None:
        raise RuntimeError("neural assembly worker was not initialized")
    smiles, _ = _NEURAL_WORKER_ASSEMBLER.to_smiles_graph(graph)
    return smiles


def _cpu_graph(graph):
    from fragflex.graph import FragmentGraph

    return FragmentGraph(
        x=graph.x.detach().cpu(),
        e=graph.e.detach().cpu(),
        smiles=graph.smiles,
        sites=None if graph.sites is None else graph.sites.detach().cpu(),
        site_logits=(
            None
            if graph.site_logits is None
            else graph.site_logits.detach().cpu()
        ),
    )


def _sync_if_cuda(device: str | torch.device) -> None:
    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(dev)


def _format_timing(name: str, seconds: float, n: int) -> str:
    per_sample_ms = 1000.0 * seconds / max(n, 1)
    samples_per_sec = n / max(seconds, 1e-12)
    return (
        f"{name:<12} {seconds:9.2f} s | "
        f"{per_sample_ms:9.2f} ms/sample | "
        f"{samples_per_sec:7.3f} samples/s"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Large-scale FragFlex sampler. Writes one SMILES (or None) per line "
            "and flushes after every batch."
        )
    )

    p.add_argument("--checkpoint", required=True)
    p.add_argument("--fragment-index", required=True)
    p.add_argument(
        "--fragment-edge-index",
        help="Required only for legacy lookup/constrained assemblers",
    )
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--out", default="fragflex_samples.txt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    p.add_argument("--compile", action="store_true")
    p.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
    )
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--argmax-delta", action="store_true")

    p.add_argument(
        "--assembler",
        choices=["fragdiffusion", "constrained", "neural"],
        default="neural",
    )
    p.add_argument("--beam-size", type=int, default=256)
    p.add_argument("--site-topk", type=int, default=4)
    p.add_argument("--assembly-workers", type=int, default=1)

    p.add_argument(
        "--no-replace-explicit-h",
        action="store_true",
        help=(
            "Ablation: disable explicit-H replacement during neural assembly."
        ),
    )

    p.add_argument(
        "--no-static-site-mask",
        action="store_true",
        help=(
            "Ablation: disable the static neural-site feasibility mask."
        ),
    )

    p.add_argument(
        "--assembly-selection",
        choices=["best", "uniform", "softmax"],
        default="best",
        help=(
            "How to select the final molecule among sanitized, connected, "
            "unique neural-assembly products. "
            "'best' keeps the original deterministic decoder; "
            "'uniform' samples uniformly; "
            "'softmax' samples according to neural beam scores."
        ),
    )

    p.add_argument(
        "--assembly-temperature",
        type=float,
        default=0.1,
        help=(
            "Softmax temperature used only when "
            "--assembly-selection softmax."
        ),
    )

    p.add_argument(
        "--max-valid-products",
        type=int,
        default=16,
        help=(
            "Maximum number of unique sanitized connected products "
            "considered for stochastic assembly selection."
        ),
    )

    p.add_argument("--no-connectivity-constraint", action="store_true")
    p.add_argument("--no-mode-repair", action="store_true")
    p.add_argument("--allow-disconnected", action="store_true")
    p.add_argument("--exact-tree-fallback", action="store_true")
    p.add_argument("--exact-tree-max-edges", type=int, default=12)
    p.add_argument(
        "--no-exact-tree-fallback",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    p.add_argument("--show-rdkit-errors", action="store_true")
    p.add_argument("--print-every-batch", action="store_true")

    args = p.parse_args()

    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.assembly_workers < 1:
        raise ValueError("--assembly-workers must be >= 1")
    if args.assembly_workers > 1 and args.assembler != "neural":
        raise ValueError("--assembly-workers > 1 requires --assembler neural")
    if args.assembly_temperature < 0:
        raise ValueError("--assembly-temperature must be >= 0")
    if args.max_valid_products < 1:
        raise ValueError("--max-valid-products must be >= 1")

    if not args.show_rdkit_errors:
        RDLogger.DisableLog("rdApp.error")

    _seed_everything(args.seed)

    payload = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model = FragFlexModel.from_checkpoint(payload).to(args.device)
    model.eval()
    model.diffusion.to(args.device)

    if (
        args.tf32
        and torch.cuda.is_available()
        and torch.device(args.device).type == "cuda"
    ):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("--compile requires PyTorch 2.x")
        model.denoiser = torch.compile(
            model.denoiser,
            mode=args.compile_mode,
            dynamic=True,
        )
        model.delta_model = torch.compile(
            model.delta_model,
            mode=args.compile_mode,
            dynamic=True,
        )

    if args.argmax_delta:
        model.config.sample_delta = False

    fragments = FragmentLibrary.from_csv(args.fragment_index)
    use_exact_tree = bool(
        args.exact_tree_fallback and not args.no_exact_tree_fallback
    )

    replace_explicit_h = not args.no_replace_explicit_h
    static_site_mask = not args.no_static_site_mask

    if args.assembler == "neural":
        if not model.stats.uses_neural_sites:
            raise ValueError(
                "--assembler neural requires a checkpoint trained on neural_sites data"
            )
        assembler = NeuralSiteAssembler(
            fragments,
            beam_size=args.beam_size,
            site_topk=args.site_topk,
            require_connected=not args.allow_disconnected,
            replace_explicit_h=replace_explicit_h,
            static_site_mask=static_site_mask,
            assembly_selection=args.assembly_selection,
            assembly_temperature=args.assembly_temperature,
            max_valid_products=args.max_valid_products,
        )
    else:
        if model.stats.uses_neural_sites:
            raise ValueError(
                "This checkpoint predicts neural atom sites; use --assembler neural."
            )
        if not args.fragment_edge_index:
            raise ValueError(
                "--fragment-edge-index is required for legacy assemblers"
            )
        attachments = AttachmentLibrary.from_csv(
            fragments,
            args.fragment_edge_index,
        )
        if args.assembler == "constrained":
            assembler = ConstrainedFragDiffusionAssembler(
                fragments,
                attachments,
                beam_size=args.beam_size,
                allow_mode_repair=not args.no_mode_repair,
                require_connected=not args.allow_disconnected,
                exact_tree_fallback=use_exact_tree,
                exact_tree_max_edges=args.exact_tree_max_edges,
            )
        else:
            assembler = FragDiffusionAssembler(fragments, attachments)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    sampling_time = 0.0
    assembly_time = 0.0
    written = 0
    valid = 0
    remaining = args.num_samples
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    batch_idx = 0
    measured_start = time.perf_counter()

    assembly_pool = None
    if args.assembler == "neural" and args.assembly_workers > 1:
        assembly_pool = ProcessPoolExecutor(
            max_workers=args.assembly_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_neural_assembly_worker,
            initargs=(
                fragments.id_to_smiles,
                args.beam_size,
                args.site_topk,
                not args.allow_disconnected,
                replace_explicit_h,
                static_site_mask,
                args.assembly_selection,
                args.assembly_temperature,
                args.max_valid_products,
                args.show_rdkit_errors,
                args.seed,
            ),
        )

    try:
        with out.open("w", encoding="utf-8") as f:
            while remaining > 0:
                batch_idx += 1
                b = min(args.batch_size, remaining)

                _sync_if_cuda(args.device)
                t0 = time.perf_counter()
                generated = model.sample(
                    b,
                    enforce_connectivity=not args.no_connectivity_constraint,
                )
                _sync_if_cuda(args.device)
                sampling_time += time.perf_counter() - t0

                t0 = time.perf_counter()
                if assembly_pool is not None:
                    cpu_graphs = [_cpu_graph(g) for g in generated]
                    smiles_batch = list(
                        assembly_pool.map(
                            _assemble_neural_worker,
                            cpu_graphs,
                        )
                    )
                else:
                    smiles_batch = []
                    for g in generated:
                        if args.assembler == "neural":
                            smiles, _ = assembler.to_smiles_graph(g)
                        else:
                            smiles, _ = assembler.to_smiles(g.x, g.e)
                        smiles_batch.append(smiles)

                assembly_time += time.perf_counter() - t0

                lines = []
                batch_valid = 0
                for smiles in smiles_batch:
                    if smiles:
                        lines.append(smiles + "\n")
                        batch_valid += 1
                    else:
                        lines.append("None\n")

                f.writelines(lines)
                f.flush()

                written += b
                valid += batch_valid
                remaining -= b

                if args.print_every_batch:
                    elapsed = time.perf_counter() - measured_start
                    print(
                        f"[batch {batch_idx}/{num_batches}] "
                        f"samples={written}/{args.num_samples} | "
                        f"valid={valid / written:.2%} | "
                        f"elapsed={elapsed:.1f}s | "
                        f"sampling={sampling_time:.1f}s | "
                        f"assembly={assembly_time:.1f}s",
                        flush=True,
                    )
    finally:
        if assembly_pool is not None:
            assembly_pool.shutdown(wait=True)

    compute_wall_time = time.perf_counter() - measured_start

    print(
        f"wrote {written} samples to {out}; "
        f"valid={valid / written:.2%}; "
        f"seed={args.seed}; "
        f"assembler={args.assembler}; "
        f"assembly_workers={args.assembly_workers}; "
        f"replace_explicit_h={replace_explicit_h}; "
        f"static_site_mask={static_site_mask}; "
        f"assembly_temperature={args.assembly_temperature}; "
        f"max_valid_products={args.max_valid_products}; "
        f"sampling_impl=fast_v1"
    )

    print("\nTiming profile")
    print(_format_timing("sampling", sampling_time, written))
    print(_format_timing("assembly", assembly_time, written))
    print(_format_timing("compute wall", compute_wall_time, written))


if __name__ == "__main__":
    main()