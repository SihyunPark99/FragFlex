# FragDiffusion preprocessing artifacts

FragFlex starts from the public ZINC fragment preprocessing released with [FragDiffusion](https://github.com/danielTLevy/FragDiffusion/tree/main/data/frag).

The two small lookup tables in this directory are included directly:

- `fragment_index.csv`
- `fragment_edge_index.csv`

The following upstream binary artifacts are intentionally not committed:

- `mol_frag_graphs_100000.pt.gz`
- `split_idxs.npz`

From the repository root, run:

```bash
python -m scripts.prepare_data
```

The script downloads those two files from FragDiffusion commit `564951e964dce68246530d6486d4eebe04fa50b8` and then runs the FragFlex-specific conversion step.

The resulting training data are written to:

```text
data/fragflex_neural/
├── graphs.pt
├── stats.pt
└── split_idxs.npz
```

The data flow is:

```text
ZINC
  ↓
FragDiffusion preprocessing (upstream)
  ↓
mol_frag_graphs_100000.pt.gz
fragment_index.csv
fragment_edge_index.csv
split_idxs.npz
  ↓
FragFlex preprocessing (scripts/convert_fragdiffusion.py)
  ↓
graphs.pt
stats.pt
split_idxs.npz
  ↓
FragFlex training
```
