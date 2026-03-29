# DGPO: RL-Steered Graph Diffusion for Neural Architecture Generation

[![arXiv](https://img.shields.io/badge/arXiv-2602.19261-b31b1b.svg)](https://arxiv.org/abs/2602.19261)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![IJCNN 2026](https://img.shields.io/badge/IJCNN-2026-green.svg)](https://2026.ijcnn.org/)

Companion code for the paper accepted at **IJCNN 2026** (Maastricht):

> A. Liuliakov, L. Hermes, B. Hammer. *DGPO: RL-Steered Graph Diffusion for Neural Architecture Generation.* IJCNN 2026. [arXiv:2602.19261](https://arxiv.org/abs/2602.19261)

This is a simplified, standalone reimplementation designed for transparency and ease of use. It implements the core DGPO experiments on NAS-Bench-101 and NAS-Bench-201. All code, vendored APIs, configs, and experiment scripts are self-contained.

This codebase builds upon [DiGress](https://github.com/cvignac/DiGress) (discrete graph diffusion) and [GDPO](https://github.com/sail-sg/GDPO) (RL fine-tuning of diffusion models).

## Quick Start

```bash
make setup                                  # install deps + download NB101 data
make pretrain-nb201                         # pretrain diffusion model
make rlft-nb201                             # RL fine-tune with DGPO
```

Or run a full experiment pipeline (pretrain, baselines, RL-FT across 3 seeds, evaluate, aggregate):

```bash
./scripts/run_experiment_1_nb201.sh
```

## Installation

**Prerequisites:** Python 3.9+, CUDA-capable GPU, conda (recommended).

**Step 1: Create environment and install dependencies.**

```bash
conda create -n dgpo python=3.9 -y
conda activate dgpo

# PyTorch -- choose your CUDA version:
pip install "torch>=2.0" --index-url https://download.pytorch.org/whl/cu118    # CUDA 11.8
# pip install "torch>=2.0" --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1

# TensorFlow CPU (for NB101 tfrecord parsing only)
pip install "tensorflow-cpu>=2.10"

# Install companion package (adds numpy, pyyaml, protobuf, absl-py)
pip install -e .
```

For exact pinned versions, see `requirements.txt`.

**Step 2: Download benchmark data.**

```bash
make setup    # auto-downloads NB101 (~500 MB), checks NB201
```

NAS-Bench-201 (~2.3 GB, manual download required):
1. Download `NAS-Bench-201-v1_1-096897.pth` from the [NAS-Bench-201 repository](https://github.com/D-X-Y/NAS-Bench-201).
2. Place it at `data/NASBench201/raw/NAS-Bench-201-v1_1-096897.pth`.
3. Build the unified cache (reduces runtime memory from ~13 GB to ~5 MB):

```bash
make build-nb201-cache
```

**Step 3: Verify installation.**

```bash
python scripts/preflight.py
```

### Optional: Pretrained Checkpoints

To skip pretraining and go directly to RL fine-tuning, download pretrained checkpoints:

**Baseline DGPO (full pretrain):**

- NB101: [pretrain_nb101_full_seed42.pt](https://drive.google.com/file/d/1qnlr0v8KlhAsBwhGcXtOAeVljFag6rXR/view?usp=sharing) (108 MB)
- NB201: [pretrain_nb201_full_seed42.pt](https://drive.google.com/file/d/152bb6YCsL2Fi4QkjxNaf9Gkay3fSiKHF/view?usp=sharing) (108 MB)

**OOD Discovery (filtered pretrain):**

- NB101: [pretrain_nb101_filtered_seed42.pt](https://drive.google.com/file/d/143tW53pgQ0uCWilHw-1qXw8Omj5rVz_k/view?usp=sharing) (108 MB)
- NB201: [pretrain_nb201_filtered_seed42.pt](https://drive.google.com/file/d/1-_HxtARbfl1ze-U5njwEwZPU9h4DvIPV/view?usp=sharing) (108 MB)

Place at:

```
outputs/pretrain_nb101_seed42/checkpoints/best.pt          # full
outputs/pretrain_nb201_seed42/checkpoints/best.pt          # full
outputs/pretrain_nb101_filtered_seed42/checkpoints/best.pt  # filtered
outputs/pretrain_nb201_filtered_seed42/checkpoints/best.pt  # filtered
```

Then run RL-FT directly:

```bash
make rlft-nb101 PRETRAIN_NB101_CKPT=outputs/pretrain_nb101_seed42/checkpoints/best.pt
make rlft-nb201 PRETRAIN_NB201_CKPT=outputs/pretrain_nb201_seed42/checkpoints/best.pt
```

## Experiments

### Experiment 1: Baseline DGPO

Pretrain a graph diffusion model on the full benchmark, then RL fine-tune with DGPO to maximize validation accuracy.

**NAS-Bench-201** (3 tasks: CIFAR-10, CIFAR-100, ImageNet-16-120):

```bash
./scripts/run_experiment_1_nb201.sh           # full pipeline
./scripts/run_experiment_1_nb201.sh --quick   # smoke test (~10 min)
```

**NAS-Bench-101:**

```bash
./scripts/run_experiment_1_nb101.sh
```

Each script runs: pretrain, compute baselines, RL fine-tune (3 seeds), evaluate, aggregate.

### Experiment 2: OOD Discovery via Filtered Pretraining

Pretrain on a filtered subset (architectures below an accuracy threshold), then RL fine-tune. Tests whether the model can discover high-accuracy architectures not seen during pretraining.

```bash
./scripts/run_experiment_2_nb201.sh    # NB201 (threshold=0.85)
./scripts/run_experiment_2_nb101.sh    # NB101 (threshold=0.87)
```

### Experiment 3: Bidirectional Control

Demonstrates that DGPO can steer generation in both directions: toward high accuracy (forward) and toward low accuracy (inverse, with negated reward).

```bash
./scripts/run_inverse_control.sh
```

## Code-to-Paper Guide

The scripts in this repository aim to illustrate the experiments described in the paper. An interested reader will recognize the corresponding results.

| Experiment | Script |
|-----------|--------|
| Baseline DGPO | `scripts/run_experiment_1_nb{101,201}.sh` |
| OOD Discovery | `scripts/run_experiment_2_nb{101,201}.sh` |
| Bidirectional control | `scripts/run_inverse_control.sh` |
| Training dynamics | `train_log.jsonl` from RL-FT runs |
| Distribution shift | Distribution sampling CSV from RL-FT |

## Cluster Deployment

Generic SLURM example:

```bash
#!/bin/bash
#SBATCH --job-name=dgpo_exp1_nb201
#SBATCH --partition=gpu           # adjust to your cluster
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j_%x.out

cd /path/to/dgpo-companion
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dgpo
./scripts/run_experiment_1_nb201.sh
```

This repo includes Claude Code skills in `.claude/skills/` for AI-assisted environment setup, experiment deployment, and validation.

## Project Structure

```
DGPO/
├── .claude/skills/             # AI agent skills (optional)
├── configs/                    # 12 YAML experiment configs
├── data/                       # Benchmark data (downloaded at setup)
├── expected_outputs/           # Reference metrics for automated validation
├── scripts/
│   ├── setup.sh                # Environment setup and data download
│   ├── build_nb201_cache.py    # Build unified NB201 cache
│   ├── preflight.py            # Pre-flight installation checks
│   ├── run_experiment_*.sh     # Experiment pipelines
│   └── run_inverse_control.sh  # Bidirectional control
├── src/
│   ├── model.py                # Graph Transformer with positional encoding
│   ├── diffusion.py            # Discrete diffusion (noise schedule, sampling)
│   ├── train_pretrain.py       # Pretraining loop
│   ├── train_rlft.py           # DGPO RL fine-tuning
│   ├── rewards.py              # Reward computation via NAS-Bench APIs
│   ├── evaluate.py             # Sampling, evaluation, aggregation
│   ├── baselines.py            # Random search and pretrained-only baselines
│   ├── dataset.py              # NB101/NB201 data loading with caching
│   ├── ops.py                  # Operation primitives and adjacency helpers
│   └── utils.py                # Config loading, masking utilities
├── vendored/                   # Bundled NAS-Bench-101/201 APIs
├── Makefile                    # Top-level interface (run `make help`)
├── pyproject.toml              # Package metadata and dependencies
├── requirements.txt            # Pinned tested versions
└── LICENSE                     # MIT
```

## Acknowledgments

This implementation builds upon:
- [DiGress](https://github.com/cvignac/DiGress) -- Discrete denoising diffusion for graph generation (Vignac et al., 2023)
- [GDPO](https://github.com/sail-sg/GDPO) -- RL fine-tuning of discrete diffusion models (Fan et al., 2024)
- [NAS-Bench-101](https://github.com/google-research/nasbench) -- Tabular NAS benchmark (Ying et al., 2019)
- [NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201) -- Cell-based NAS benchmark (Dong & Yang, 2020)

## Citation

```bibtex
@inproceedings{liuliakov2026dgpo,
  title={DGPO: RL-Steered Graph Diffusion for Neural Architecture Generation},
  author={Liuliakov, Aleksandr and Hermes, Luca and Hammer, Barbara},
  booktitle={International Joint Conference on Neural Networks (IJCNN)},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).

Third-party code (vendored NAS-Bench APIs) retains its original licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
