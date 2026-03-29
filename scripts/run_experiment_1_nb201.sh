#!/bin/bash
# Experiment 1: Baseline DGPO on NAS-Bench-201 (Table 1 NB201 rows, Fig 1, Fig 2)
#
# Runs all three NB201 tasks (CIFAR-10, CIFAR-100, ImageNet-16-120) using one
# shared pretrain checkpoint and three separate RL-FT configs.
#
# Usage:
#   ./scripts/run_experiment_1_nb201.sh            # full run
#   ./scripts/run_experiment_1_nb201.sh --quick     # CI smoke test
set -euo pipefail
cd "$(dirname "$0")/.."

QUICK=false
for arg in "$@"; do [[ "$arg" == "--quick" ]] && QUICK=true; done

if $QUICK; then
  PRETRAIN_EPOCHS=5; RLFT_EPOCHS=3; N_SAMPLES=50; LIMIT="--limit 100"
else
  PRETRAIN_EPOCHS=2000; RLFT_EPOCHS=80; N_SAMPLES=300; LIMIT=""
fi

PRETRAIN_SEED=42
PRETRAIN_RUN="pretrain_nb201_seed${PRETRAIN_SEED}"
SEEDS=(42 123 456)
TASKS=(cifar10 cifar100 imagenet16)

# Map task shortname to config file name (POSIX-safe, no associative arrays)
task_config() {
  case "$1" in
    cifar10)     echo "rlft_nb201_cifar10" ;;
    cifar100)    echo "rlft_nb201_cifar100" ;;
    imagenet16)  echo "rlft_nb201_imagenet16" ;;
    *)           echo "ERROR: unknown task $1" >&2; return 1 ;;
  esac
}

# ===================================================================
# Phase 1: Pretrain (shared across all 3 tasks — NB201 pretrain is task-agnostic)
# ===================================================================
echo "=== Phase 1: Pretraining NB201 ==="
python src/train_pretrain.py --config configs/pretrain_nb201.yaml \
  --epochs "$PRETRAIN_EPOCHS" $LIMIT --seed "$PRETRAIN_SEED"

PRETRAIN_CKPT="outputs/${PRETRAIN_RUN}/checkpoints/best.pt"
if [[ ! -f "$PRETRAIN_CKPT" ]]; then
  echo "ERROR: Pretrain checkpoint not found: $PRETRAIN_CKPT"
  exit 1
fi

# ===================================================================
# Baselines (3 seeds, matching paper protocol)
# ===================================================================
echo "=== Computing baselines ==="
python src/baselines.py --mode random --benchmark nb201 --n-samples "$N_SAMPLES" \
  --seeds "${SEEDS[@]}"
python src/baselines.py --mode pretrained --checkpoint "$PRETRAIN_CKPT" \
  --benchmark nb201 --n-samples "$N_SAMPLES" --seeds "${SEEDS[@]}"

# ===================================================================
# Phase 2: RL-FT for each task x seed
# ===================================================================
for TASK in "${TASKS[@]}"; do
  CONFIG_NAME="$(task_config "$TASK")"
  echo "=== Phase 2: RL-FT NB201 ${TASK} ==="
  for SEED in "${SEEDS[@]}"; do
    echo "  seed=${SEED}"
    python src/train_rlft.py --config "configs/${CONFIG_NAME}.yaml" \
      --resume "$PRETRAIN_CKPT" --epochs "$RLFT_EPOCHS" --seed "$SEED"
  done
done

# ===================================================================
# Phase 3: Evaluate each task x seed
# ===================================================================
echo "=== Evaluation ==="
for TASK in "${TASKS[@]}"; do
  CONFIG_NAME="$(task_config "$TASK")"
  for SEED in "${SEEDS[@]}"; do
    RLFT_RUN="${CONFIG_NAME}_seed${SEED}"
    RLFT_CKPT="outputs/${RLFT_RUN}/checkpoints/best.pt"
    if [[ ! -f "$RLFT_CKPT" ]]; then
      # Fallback: last epoch checkpoint (zero-padded, e.g. epoch_079.pt)
      LAST_EPOCH=$(printf "%03d" $((RLFT_EPOCHS - 1)))
      RLFT_CKPT="outputs/${RLFT_RUN}/checkpoints/epoch_${LAST_EPOCH}.pt"
      if [[ ! -f "$RLFT_CKPT" ]]; then
        echo "ERROR: No checkpoint found for ${RLFT_RUN} (tried best.pt and epoch_${LAST_EPOCH}.pt)"
        exit 1
      fi
      echo "WARN: best.pt not found, using ${RLFT_CKPT}"
    fi
    python src/evaluate.py \
      --checkpoint "$RLFT_CKPT" \
      --config "configs/${CONFIG_NAME}.yaml" \
      --n-samples "$N_SAMPLES" --seed "$SEED" \
      --output "outputs/eval_${RLFT_RUN}/results.json"
  done
done

# ===================================================================
# Phase 4: Aggregate per-seed results and compare against expected
# ===================================================================
echo "=== Aggregation & Pass/Fail ==="
# Collect only this experiment's eval results into a staging dir to avoid
# picking up stale/unrelated runs (filtered, inverse) from outputs/
EXP1_STAGING="outputs/_exp1_nb201_staging"
rm -rf "$EXP1_STAGING"
mkdir -p "$EXP1_STAGING"
for TASK in "${TASKS[@]}"; do
  CONFIG_NAME="$(task_config "$TASK")"
  for SEED in "${SEEDS[@]}"; do
    RLFT_RUN="${CONFIG_NAME}_seed${SEED}"
    SRC="outputs/eval_${RLFT_RUN}/results.json"
    DEST_DIR="${EXP1_STAGING}/eval_${RLFT_RUN}"
    mkdir -p "$DEST_DIR"
    cp "$SRC" "$DEST_DIR/results.json"
  done
done

python src/evaluate.py --aggregate \
  --results-dir "$EXP1_STAGING" \
  --pattern "eval_*/results.json" \
  --expected expected_outputs/experiment_1_nb201.json \
  --output outputs/experiment_1_nb201_summary.json

echo "=== Done. Summary: outputs/experiment_1_nb201_summary.json ==="
