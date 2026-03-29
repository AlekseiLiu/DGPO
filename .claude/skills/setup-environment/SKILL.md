---
name: setup-environment
description: >
  Guide environment installation for the DGPO companion repo. Use when the user
  asks how to install, set up, configure the environment, install dependencies,
  or encounters import errors, CUDA issues, or package conflicts. Interactively
  determines system details and walks through setup step by step.
---

# Setup DGPO Environment

Install Python environment and dependencies for running DGPO experiments.

## Step 1: Determine system details

Ask the user (or detect automatically where possible):
- **OS:** Linux, macOS, Windows (WSL)?
- **CUDA version:** `nvidia-smi` or `nvcc --version` (common: 11.8, 12.1)
- **Package manager:** conda (recommended) or pip/venv?
- **GPU available?** Required for training, not for data prep.

## Step 2: Install

### Option A: Conda (recommended)

```bash
conda create -n dgpo python=3.9 -y
conda activate dgpo
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118  # adjust for CUDA version
pip install tensorflow-cpu==2.15.0
pip install -e .
```

### Option B: pip + venv

```bash
python3.9 -m venv .venv && source .venv/bin/activate
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow-cpu==2.15.0
pip install -e .
```

Adjust the PyTorch index URL for the user's CUDA version:
- CUDA 11.8: `https://download.pytorch.org/whl/cu118`
- CUDA 12.1: `https://download.pytorch.org/whl/cu121`

## Step 3: Download data

```bash
make setup                    # auto-downloads NB101 (~500 MB), installs package
make build-nb201-cache        # after manually placing NB201 API file
```

NB201 API (~2.3 GB) must be downloaded manually from https://github.com/D-X-Y/NAS-Bench-201 and placed at `data/NASBench201/raw/NAS-Bench-201-v1_1-096897.pth`.

## Step 4: Verify

```bash
python scripts/preflight.py
```

All checks should pass. Data-related warnings are expected if datasets aren't downloaded yet.

## Known issues

- **protobuf conflict:** If tensorflow pulls protobuf 3.x, fix with `pip install "protobuf>=4.0"`
- **numpy 2.0:** Not compatible. `pyproject.toml` enforces `numpy<2.0`.
- **tensorflow-cpu vs tensorflow:** Either works; tensorflow-cpu is lighter.

## Tested configuration

Python 3.9, PyTorch 2.0.1+cu118, tensorflow-cpu 2.15.0, numpy 1.26.4, NVIDIA A40, CUDA 11.8, Ubuntu 20.04.
