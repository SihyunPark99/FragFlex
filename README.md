# FragFlex: Fragment-Based Flexible Molecular Generation for Exploring Vast Chemical Space

<p align="center">
  <img src="example.gif" width="800" alt="FragFlex sampling example">
</p>

FragFlex is a fragment-level discrete diffusion model for flexible-size molecular generation. This repository contains the compact ZINC experiment code path used for the paper and a single-GPU reproduction configuration.

## Data flow

FragFlex starts from the public ZINC fragment preprocessing released with [FragDiffusion](https://github.com/danielTLevy/FragDiffusion/tree/main/data/frag) and performs the additional conversion needed for neural attachment-site targets.

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
FragFlex preprocessing
  ↓
graphs.pt
stats.pt
split_idxs.npz
  ↓
FragFlex training
```

The repository includes `fragment_index.csv` and `fragment_edge_index.csv`. The data preparation script downloads the two pinned FragDiffusion binary artifacts and converts them to the FragFlex representation.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`torch-geometric` is required because the released FragDiffusion `.pt.gz` file contains serialized PyTorch Geometric graph objects.

## Python entry points

The public workflow uses Python modules directly; no shell launcher scripts are required.

- `scripts/prepare_data.py`: downloads the pinned FragDiffusion binary artifacts and runs the FragFlex-specific conversion.
- `scripts/convert_fragdiffusion.py`: converts an already available FragDiffusion fragment graph file into `graphs.pt` and `stats.pt`; this is the lower-level conversion entry point used by `prepare_data.py`.
- `scripts/train.py`: trains FragFlex with Hydra configuration. The default config is the single-GPU paper reproduction preset.
- `scripts/sample.py`: diagnostic sampler that writes per-molecule CSV output with assembly diagnostics.
- `scripts/sample_large.py`: streaming sampler for large-scale generation; it writes one SMILES or `None` per line.

## 1. Prepare ZINC data

From the repository root, run:

```bash
python -m scripts.prepare_data
```

The command downloads these files from the pinned FragDiffusion revision if they are not already present:

```text
data/frag/mol_frag_graphs_100000.pt.gz
data/frag/split_idxs.npz
```

It then uses the two CSV files already included in this repository:

```text
data/frag/fragment_index.csv
data/frag/fragment_edge_index.csv
```

and produces:

```text
data/fragflex_neural/
├── graphs.pt
├── stats.pt
└── split_idxs.npz
```

The released FragDiffusion split is preserved through `split_idxs.npz`. If a manually prepared dataset does not contain that file, the data module falls back to a deterministic 64/16/20 train/validation/test split with seed 1234.

For a manual conversion when all upstream artifacts are already available:

```bash
python -m scripts.convert_fragdiffusion \
  --frag-graphs data/frag/mol_frag_graphs_100000.pt.gz \
  --fragment-index data/frag/fragment_index.csv \
  --fragment-edge-index data/frag/fragment_edge_index.csv \
  --assembly-target neural_sites \
  --split-idxs data/frag/split_idxs.npz \
  --out-dir data/fragflex_neural
```

See `data/frag/README.md` for the upstream artifact provenance.

## Paper configuration

The default Hydra composition selects:

- `model=fragflex_paper`
- `train=paper_single_gpu`
- `general=paper_single_gpu`
- `dataset=fragflex`

Key settings are:

| Component | Setting |
| --- | --- |
| Diffusion | `T=500`, cosine schedule, `s=0.008` |
| Deletion/insertion | logistic structural schedule, `s_p=0`, `w=0.05` |
| Terminal prior | one fragment from the empirical fragment marginal |
| Main denoiser | 7 layers, node width 384, edge width 96, 8 heads |
| Count model | 2 layers, node width 128, edge width 32, 8 heads |
| Maximum graph size | 32 fragments |
| Training | 200 epochs, dropout 0.05 |
| Optimizer | AdamW, learning rate `2e-4`, weight decay `1e-3` |
| LR schedule | 5% warmup from 0.1x LR, cosine decay to 0.05x LR |
| Assembly | top-k 4, beam 512, at most 16 valid products, uniform selection |

The model also uses DiGress-style cycle auxiliary features. Loss weights are `lambda_edge=5`, `lambda_delta=1`, and `lambda_site=2`.

## 2. Train on one GPU

The original experiment used 4 GPUs with batch size 32 per GPU, giving an effective batch size of 128. The public single-GPU preset uses a micro-batch of 32 with four-step gradient accumulation:

```yaml
batch_size: 32
accumulate_grad_batches: 4
```

After data preparation, start training with:

```bash
python -m scripts.train
```

Hydra writes each run under `outputs/<date>/<time>-<run-name>/`. Training saves the best validation checkpoint and the last checkpoint.

Common overrides can be passed directly on the command line. For example:

```bash
python -m scripts.train train.seed=1 general.name=fragflex-zinc-seed1
```

The default execution config requires one CUDA GPU. CPU execution can be used for debugging with:

```bash
python -m scripts.train general.accelerator=cpu general.gpus=1
```

## 3. Sample molecules

### Diagnostic sampling

`scripts.sample` writes detailed CSV diagnostics and is convenient for small runs:

```bash
python -m scripts.sample \
  --checkpoint outputs/.../checkpoints/best-epochXXXX.ckpt \
  --fragment-index data/frag/fragment_index.csv \
  --assembler neural \
  --beam-size 512 \
  --site-topk 4 \
  --assembly-selection uniform \
  --max-valid-products 16 \
  --num-samples 100 \
  --out samples_100.csv
```

### Large-scale sampling

For the paper-scale sampling path, use the streaming sampler:

```bash
python -m scripts.sample_large \
  --checkpoint outputs/.../checkpoints/best-epochXXXX.ckpt \
  --fragment-index data/frag/fragment_index.csv \
  --num-samples 100000 \
  --batch-size 512 \
  --assembler neural \
  --beam-size 512 \
  --site-topk 4 \
  --assembly-selection uniform \
  --max-valid-products 16 \
  --assembly-workers 8 \
  --seed 0 \
  --out samples_100k_seed0.txt \
  --print-every-batch
```

`--assembly-workers` affects CPU assembly throughput only; reduce it if fewer CPU cores are available.

## Tests

```bash
python -m pytest -q
```

## Repository structure

```text
FragFlex/
├── fragflex/
├── scripts/
│   ├── prepare_data.py
│   ├── convert_fragdiffusion.py
│   ├── train.py
│   ├── sample.py
│   └── sample_large.py
├── configs/
│   ├── config.yaml
│   ├── dataset/
│   │   └── fragflex.yaml
│   ├── model/
│   │   └── fragflex_paper.yaml
│   ├── train/
│   │   └── paper_single_gpu.yaml
│   └── general/
│       └── paper_single_gpu.yaml
├── data/
│   └── frag/
│       ├── README.md
│       ├── fragment_index.csv
│       └── fragment_edge_index.csv
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

`configs/config.yaml` is the Hydra composition file used by `scripts/train.py`.

## Upstream attribution

The ZINC fragment preprocessing artifacts originate from the public FragDiffusion repository by Daniel T. Levy and collaborators.
