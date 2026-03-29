---
name: deploy-experiment
description: >
  Generate cluster job submission scripts (SLURM, PBS, or bash) for running DGPO
  experiments. Use when the user wants to run experiments on a cluster, asks about
  SLURM/sbatch setup, needs help with GPU resource allocation, or mentions deploying
  to HPC. Interactively gathers cluster details and produces ready-to-submit scripts.
---

# Deploy Experiment to Cluster

Generate a job submission script tailored to the user's cluster and experiment.

## Step 1: Gather cluster details

Ask the user:
- **Scheduler:** SLURM (sbatch), PBS (qsub), or plain bash?
- **Partition/queue:** (default: `gpu`)
- **Conda path:** Where is `conda.sh`? (e.g., `~/miniconda3/etc/profile.d/conda.sh`)
- **Conda env name:** (default: `dgpo`)
- **Repo path on cluster:** Where is the repo checked out?

## Step 2: Select experiment

| ID | Experiment | Pipeline script |
|----|-----------|----------------|
| 1a | Baseline DGPO, NB101 | `./scripts/run_experiment_1_nb101.sh` |
| 1b | Baseline DGPO, NB201 (3 tasks) | `./scripts/run_experiment_1_nb201.sh` |
| 2a | OOD Discovery, NB101 | `./scripts/run_experiment_2_nb101.sh` |
| 2b | OOD Discovery, NB201 | `./scripts/run_experiment_2_nb201.sh` |
| 3  | Bidirectional control | `./scripts/run_inverse_control.sh` |

Also ask:
- **Seeds:** (default: `42 123 456`)
- **Phase:** full pipeline, pretrain only, or RL-FT only (needs checkpoint path)?
- **Quick mode?** (`--quick` for ~10 min smoke test)

## Step 3: Resource allocation

Recommended minimums (actual needs may vary by hardware):

| Experiment | GPU | RAM | Wall time |
|-----------|-----|-----|-----------|
| Pretrain NB101 | 1x | 32 GB | 72h |
| Pretrain NB201 | 1x | 16 GB | 48h |
| RL-FT (any) | 1x | 16-32 GB | 24h |
| Quick mode | 1x | 8 GB | 30 min |

## Step 4: Generate script

For SLURM, use this template:

```bash
#!/bin/bash
#SBATCH --job-name=dgpo_{experiment}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output=logs/%j_%x.out

set -euo pipefail
mkdir -p logs

cd {repo_path}
source {conda_sh}
conda activate {conda_env}

{commands}
```

Write the script to a file (e.g., `run_exp1_nb201.sbatch`) and tell the user:

```bash
sbatch run_exp1_nb201.sbatch
```
