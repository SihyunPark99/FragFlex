#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import torch

from fragflex.chemistry import AttachmentLibrary, FragmentLibrary
from fragflex.data import (
    compute_stats,
    load_fragdiffusion_pt,
    load_fragdiffusion_pt_neural_sites,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert released FragDiffusion fragment graphs into FragFlex training targets"
    )
    p.add_argument("--frag-graphs", required=True, help="FragDiffusion mol_frag_graphs_*.pt or .pt.gz file")
    p.add_argument("--fragment-index", required=True, help="FragDiffusion fragment_index.csv")
    p.add_argument(
        "--fragment-edge-index",
        help=(
            "FragDiffusion fragment_edge_index.csv. Required only for --assembly-target "
            "neural_sites; it is used once to convert legacy edge-mode labels into atom-site targets."
        ),
    )
    p.add_argument(
        "--assembly-target",
        choices=["lookup", "neural_sites"],
        default="lookup",
        help=(
            "lookup: preserve FragDiffusion attachment-mode edges; neural_sites: save binary "
            "fragment topology plus directed atom-site targets for learned assembly."
        ),
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--split-idxs",
        help=(
            "Optional FragDiffusion split_idxs.npz. When supplied it is copied into "
            "the FragFlex output directory so training uses the exact upstream split."
        ),
    )
    p.add_argument(
        "--attr-has-no-edge",
        action="store_true",
        help=(
            "Use only if edge_attr already includes class 0=no-edge. Original stored "
            "FragDiffusion graphs normally do not."
        ),
    )
    p.add_argument(
        "--allow-unresolved-sites",
        action="store_true",
        help="Skip legacy edges whose atom endpoints cannot be recovered instead of failing preprocessing.",
    )
    args = p.parse_args()

    fragments = FragmentLibrary.from_csv(args.fragment_index)
    if args.assembly_target == "neural_sites":
        if not args.fragment_edge_index:
            raise ValueError("--fragment-edge-index is required for --assembly-target neural_sites")
        attachments = AttachmentLibrary.from_csv(fragments, args.fragment_edge_index)
        ds = load_fragdiffusion_pt_neural_sites(
            args.frag_graphs,
            attachments=attachments,
            attr_has_no_edge=args.attr_has_no_edge,
            strict=not args.allow_unresolved_sites,
        )
        # Binary coarse topology: 0=no edge, 1=linked.
        stats = compute_stats(
            ds.graphs,
            fragments,
            num_edge_types=2,
            assembly_mode="neural_sites",
        )
    else:
        ds = load_fragdiffusion_pt(
            args.frag_graphs,
            attr_has_no_edge=args.attr_has_no_edge,
        )
        stats = compute_stats(ds.graphs, fragments, assembly_mode="lookup")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds.save(out / "graphs.pt")
    torch.save(stats.state_dict(), out / "stats.pt")
    if args.split_idxs:
        shutil.copy2(args.split_idxs, out / "split_idxs.npz")

    print(f"saved {len(ds):,} graphs to {out / 'graphs.pt'}")
    print(f"assembly target: {stats.assembly_mode}")
    print(f"fragment states: {stats.num_fragments}")
    print(f"atom latent states: {stats.num_atom_latents} ({', '.join(stats.atom_labels)})")
    print(f"edge states including no-edge: {stats.num_edge_types}")
    if stats.uses_neural_sites:
        print(f"max fragment atoms / site classes: {stats.max_fragment_atoms}")
        print("runtime fragment_edge_index.csv dependency: NO")
    print(f"saved statistics to {out / 'stats.pt'}")
    if args.split_idxs:
        print(f"saved precomputed split indices to {out / 'split_idxs.npz'}")


if __name__ == "__main__":
    main()
