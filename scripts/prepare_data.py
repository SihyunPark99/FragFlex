#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

UPSTREAM_COMMIT = "564951e964dce68246530d6486d4eebe04fa50b8"
UPSTREAM_BASE = (
    "https://raw.githubusercontent.com/danielTLevy/FragDiffusion/"
    f"{UPSTREAM_COMMIT}/data/frag"
)
UPSTREAM_FILES = {
    "mol_frag_graphs_100000.pt.gz": b"\x1f\x8b",
    "split_idxs.npz": b"PK",
}


def _download(url: str, destination: Path, *, force: bool = False) -> None:
    if destination.exists() and not force:
        print(f"Using existing {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "FragFlex-data-preparation"},
    )
    print(f"Downloading {url}")
    try:
        with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    if tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded an empty file from {url}")

    tmp.replace(destination)
    print(f"Saved {destination} ({destination.stat().st_size:,} bytes)")


def _validate_magic(path: Path, expected_prefix: bytes) -> None:
    with path.open("rb") as f:
        prefix = f.read(len(expected_prefix))
    if prefix != expected_prefix:
        raise RuntimeError(
            f"{path} does not look like the expected upstream binary artifact. "
            "Delete it and rerun with --force-download."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download the pinned FragDiffusion ZINC artifacts and convert them "
            "into FragFlex neural-site training data."
        )
    )
    parser.add_argument(
        "--data-dir",
        default="data/frag",
        help="Directory containing the upstream FragDiffusion artifacts and CSV tables.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/fragflex_neural",
        help="Directory for FragFlex graphs.pt, stats.pt, and split_idxs.npz.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload the two upstream binary artifacts even if local copies exist.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    if not data_dir.is_absolute():
        data_dir = repo_root / data_dir
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir

    fragment_index = data_dir / "fragment_index.csv"
    fragment_edge_index = data_dir / "fragment_edge_index.csv"
    for required in (fragment_index, fragment_edge_index):
        if not required.is_file():
            raise FileNotFoundError(
                f"Missing repository data file: {required}. "
                "The two CSV lookup tables should be included with FragFlex."
            )

    for filename, magic in UPSTREAM_FILES.items():
        destination = data_dir / filename
        _download(
            f"{UPSTREAM_BASE}/{filename}",
            destination,
            force=args.force_download,
        )
        _validate_magic(destination, magic)

    frag_graphs = data_dir / "mol_frag_graphs_100000.pt.gz"
    split_idxs = data_dir / "split_idxs.npz"
    out_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "scripts.convert_fragdiffusion",
        "--frag-graphs",
        str(frag_graphs),
        "--fragment-index",
        str(fragment_index),
        "--fragment-edge-index",
        str(fragment_edge_index),
        "--assembly-target",
        "neural_sites",
        "--split-idxs",
        str(split_idxs),
        "--out-dir",
        str(out_dir),
    ]

    print("\nRunning FragFlex conversion:")
    print(" ".join(command))
    subprocess.run(command, cwd=repo_root, check=True)

    expected_outputs = [
        out_dir / "graphs.pt",
        out_dir / "stats.pt",
        out_dir / "split_idxs.npz",
    ]
    missing = [str(path) for path in expected_outputs if not path.is_file()]
    if missing:
        raise RuntimeError(f"Data preparation did not create expected outputs: {missing}")

    print("\nFragFlex data preparation is complete:")
    for path in expected_outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
