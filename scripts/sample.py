import csv
import json
import time
from pathlib import Path
import argparse
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


def _sync_if_cuda(device: str | torch.device) -> None:
    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(dev)


def _format_timing(name: str, seconds: float, n: int) -> str:
    per_sample_ms = 1000.0 * seconds / max(n, 1)
    samples_per_sec = n / max(seconds, 1e-12)
    return (
        f"{name:<12} {seconds:9.2f} s | "
        f"{per_sample_ms:9.2f} ms/sample | {samples_per_sec:7.3f} samples/s"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--fragment-index", required=True)
    p.add_argument("--fragment-edge-index", help="Required only for legacy lookup/constrained assemblers")
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default="fragflex_samples.csv")
    p.add_argument(
        "--failure-out",
        default=None,
        help=(
            "Optional failure-only CSV. If omitted, <out_stem>_failures.csv is written. "
            "A failure means the final assembler returned no sanitized SMILES."
        ),
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--compile",
        action="store_true",
        help=(
            "Optionally torch.compile the denoiser and DEL* count model for sampling. "
            "This can improve throughput for large runs but has a one-time compilation cost."
        ),
    )
    p.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
        help="torch.compile mode used with --compile",
    )
    p.add_argument(
        "--tf32",
        action="store_true",
        help=(
            "Allow TF32 matmul on Ampere/Hopper GPUs. Faster, but may change floating-point "
            "rounding slightly; disabled by default for strict numerical conservatism."
        ),
    )
    p.add_argument("--argmax-delta", action="store_true", help="Use argmax rather than sampling for DEL* count")
    p.add_argument(
        "--show-rdkit-errors",
        action="store_true",
        help="Show RDKit error messages. By default rdApp.error output is suppressed during assembly.",
    )
    p.add_argument(
        "--print-every-batch",
        action="store_true",
        help="Print cumulative sampling/validity/timing progress after every completed batch.",
    )
    p.add_argument(
        "--assembler",
        choices=["fragdiffusion", "constrained", "neural"],
        default="neural",
        help=(
            "Final fragment-to-molecule decoder. 'neural' uses the learned atom-site head "
            "and requires a checkpoint trained on neural_sites data."
        ),
    )
    p.add_argument("--beam-size", type=int, default=512, help="Beam size for constrained/neural assembly")
    p.add_argument("--site-topk", type=int, default=4, help="Per-endpoint top-k neural atom sites used by the neural assembler")
    p.add_argument(
        "--no-replace-explicit-h",
        action="store_true",
        help="Ablation: reproduce the old assembler that adds bonds on top of explicit [nH]/[NH] caps.",
    )
    p.add_argument(
        "--no-static-site-mask",
        action="store_true",
        help="Ablation: allow neural top-k to include atoms that cannot accept even one external single bond.",
    )
    p.add_argument(
        "--assembly-selection",
        choices=["best", "uniform", "softmax"],
        default="uniform",
        help=(
            "How to choose the final molecule among sanitized, connected, unique "
            "neural-assembly products."
        ),
    )
    p.add_argument(
        "--assembly-temperature",
        type=float,
        default=0.1,
        help="Softmax temperature used only with --assembly-selection softmax.",
    )
    p.add_argument(
        "--max-valid-products",
        type=int,
        default=16,
        help="Maximum number of unique valid products considered when --assembly-temperature > 0.",
    )
    p.add_argument(
        "--exact-tree-fallback",
        action="store_true",
        help=(
            "Opt in to exact DFS/backtracking after beam failure on small connected trees. "
            "Disabled by default because the 1,000-sample benchmark recovered no extra valid molecules."
        ),
    )
    p.add_argument(
        "--exact-tree-max-edges",
        type=int,
        default=12,
        help="Maximum number of tree edges for the optional exact DFS fallback",
    )
    # Backward-compatible hidden alias from the previous script. Exact search is
    # now already disabled by default, so this only forces the same behavior.
    p.add_argument("--no-exact-tree-fallback", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--no-connectivity-constraint",
        action="store_true",
        help="Disable reverse-sampling connectivity repair (ablation)",
    )
    p.add_argument(
        "--no-mode-repair",
        action="store_true",
        help="For constrained assembly, forbid switching to another lookup mode for the same generated fragment pair",
    )
    p.add_argument(
        "--allow-disconnected",
        action="store_true",
        help="For constrained assembly, return a sanitized disconnected molecule when no connected assembly exists",
    )
    args = p.parse_args()

    if not args.show_rdkit_errors:
        # RDKit prints sanitize/valence failures directly to stderr even when
        # Python exceptions are caught. These failures are already recorded in
        # the CSV diagnostics, so keep normal sampling output clean by default.
        RDLogger.DisableLog("rdApp.error")

    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = FragFlexModel.from_checkpoint(payload).to(args.device)
    model.eval()

    # OneAtomFragDiffusion is not an nn.Module, so explicitly make its schedules
    # and empirical marginals resident on the sampling device before timing.
    model.diffusion.to(args.device)

    if args.tf32 and torch.cuda.is_available() and torch.device(args.device).type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("--compile requires PyTorch 2.x")
        print(
            f"torch.compile enabled (mode={args.compile_mode}, dynamic=True); "
            "the first batch includes compilation overhead",
            flush=True,
        )
        model.denoiser = torch.compile(
            model.denoiser, mode=args.compile_mode, dynamic=True
        )
        model.delta_model = torch.compile(
            model.delta_model, mode=args.compile_mode, dynamic=True
        )

    if args.argmax_delta:
        model.config.sample_delta = False

    fragments = FragmentLibrary.from_csv(args.fragment_index)
    use_exact_tree = bool(args.exact_tree_fallback and not args.no_exact_tree_fallback)

    if args.assembler == "neural":
        if not model.stats.uses_neural_sites:
            raise ValueError(
                "--assembler neural requires a checkpoint trained on a dataset prepared "
                "with --assembly-target neural_sites"
            )
        assembler = NeuralSiteAssembler(
            fragments,
            beam_size=args.beam_size,
            site_topk=args.site_topk,
            require_connected=not args.allow_disconnected,
            replace_explicit_h=not args.no_replace_explicit_h,
            static_site_mask=not args.no_static_site_mask,
            assembly_selection=args.assembly_selection,
            assembly_temperature=args.assembly_temperature,
            max_valid_products=args.max_valid_products,
        )
    else:
        if model.stats.uses_neural_sites:
            raise ValueError(
                "This checkpoint predicts binary topology + neural atom sites, not legacy "
                "FragDiffusion attachment modes. Use --assembler neural."
            )
        if not args.fragment_edge_index:
            raise ValueError("--fragment-edge-index is required for legacy assemblers")
        attachments = AttachmentLibrary.from_csv(fragments, args.fragment_edge_index)
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

    rows = []
    remaining = args.num_samples
    sampling_time = 0.0
    assembly_time = 0.0
    measured_start = time.perf_counter()
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    batch_idx = 0

    while remaining > 0:
        batch_idx += 1
        b = min(args.batch_size, remaining)

        # CUDA kernels are asynchronous. Synchronize around the model call so the
        # sampling timer reflects the actual GPU work rather than launch latency.
        _sync_if_cuda(args.device)
        t0 = time.perf_counter()
        generated = model.sample(
            b,
            enforce_connectivity=not args.no_connectivity_constraint,
        )
        _sync_if_cuda(args.device)
        sampling_time += time.perf_counter() - t0

        for g in generated:
            t0 = time.perf_counter()
            if args.assembler == "neural":
                smiles, result = assembler.to_smiles_graph(g)
            else:
                smiles, result = assembler.to_smiles(g.x, g.e)
            sample_assembly_sec = time.perf_counter() - t0
            assembly_time += sample_assembly_sec

            rows.append(
                {
                    "smiles": smiles or "",
                    "num_fragments": g.num_nodes,
                    "fragment_ids": json.dumps(g.x.tolist()),
                    "attachment_matrix": json.dumps(g.e.tolist()),
                    "site_matrix": "" if g.sites is None else json.dumps(g.sites.tolist()),
                    "attempted_attachment_edges": result.attempted_edges,
                    "selected_attachment_edges": result.selected_edges,
                    "skipped_attachment_edges": result.skipped_edges,
                    "connected": int(result.connected),
                    "num_components": result.num_components,
                    "repaired_mode_edges": result.repaired_mode_edges,
                    "lookup_missing_edges": result.lookup_missing_edges,
                    "valence_rejected_options": result.valence_rejected_options,
                    "sanitize_rejected_candidates": result.sanitize_rejected_candidates,
                    "assembly_failure_reason": result.failure_reason or "",
                    "generated_topology_connected": int(result.generated_topology_connected),
                    "generated_topology_tree": int(result.generated_topology_tree),
                    "forward_pruned_states": result.forward_pruned_states,
                    "exact_search_used": int(result.exact_search_used),
                    "exact_states_visited": result.exact_states_visited,
                    "assembly_search_method": result.search_method,
                    "neural_site_repaired_edges": result.neural_site_repaired_edges,
                    "neural_site_candidate_pairs": result.neural_site_candidate_pairs,
                    "attachment_h_replacements": result.attachment_h_replacements,
                    "static_site_masked_atoms": result.static_site_masked_atoms,
                    "beam_exhausted_edge": result.beam_exhausted_edge,
                    "valid_unique_products_considered": result.valid_unique_products_considered,
                    "assembly_time_ms": 1000.0 * sample_assembly_sec,
                }
            )
        remaining -= b

        if args.print_every_batch:
            done = len(rows)
            valid_so_far = sum(bool(r["smiles"]) for r in rows)
            connected_so_far = sum(
                bool(r["connected"]) and bool(r["smiles"]) for r in rows
            )
            elapsed = time.perf_counter() - measured_start
            print(
                f"[batch {batch_idx}/{num_batches}] "
                f"samples={done}/{args.num_samples} | "
                f"sanitized={valid_so_far / max(done, 1):.2%} | "
                f"connected+sanitized={connected_so_far / max(done, 1):.2%} | "
                f"elapsed={elapsed:.1f}s | "
                f"sampling={sampling_time:.1f}s | assembly={assembly_time:.1f}s",
                flush=True,
            )

    compute_wall_time = time.perf_counter() - measured_start

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    failure_out = (
        Path(args.failure_out)
        if args.failure_out
        else out.with_name(f"{out.stem}_failures{out.suffix or '.csv'}")
    )
    failure_out.parent.mkdir(parents=True, exist_ok=True)
    failure_rows = [r for r in rows if not bool(r["smiles"])]

    io_start = time.perf_counter()
    fieldnames = list(rows[0].keys())
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    with failure_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(failure_rows)
    io_time = time.perf_counter() - io_start

    valid = sum(bool(r["smiles"]) for r in rows)
    connected = sum(bool(r["connected"]) and bool(r["smiles"]) for r in rows)
    n = len(rows)

    print(
        f"wrote {n} samples to {out}; "
        f"sanitized={valid/n:.2%}; connected+sanitized={connected/n:.2%}; "
        f"assembler={args.assembler}; exact_tree_fallback={use_exact_tree}; "
        f"sampling_impl=fast_v1; compile={args.compile}; tf32={args.tf32}"
    )
    print(f"failure-only CSV: {failure_out} ({len(failure_rows)} rows)")
    print("\nTiming profile (model/assembly only; checkpoint loading excluded)")
    print(_format_timing("sampling", sampling_time, n))
    print(_format_timing("assembly", assembly_time, n))
    print(_format_timing("compute wall", compute_wall_time, n))
    print(f"{'csv I/O':<12} {io_time:9.2f} s")
    accounted = sampling_time + assembly_time
    if compute_wall_time > 0:
        print(
            f"phase share: sampling={100.0*sampling_time/compute_wall_time:.1f}% | "
            f"assembly={100.0*assembly_time/compute_wall_time:.1f}% | "
            f"other={100.0*max(compute_wall_time-accounted, 0.0)/compute_wall_time:.1f}%"
        )


if __name__ == "__main__":
    main()