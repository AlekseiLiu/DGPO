#!/bin/bash
# Experiment 1: Baseline DGPO on NAS-Bench-101 (Table 1 NB101 row)
#
# Usage:
#   ./scripts/run_experiment_1_nb101.sh            # full run
#   ./scripts/run_experiment_1_nb101.sh --quick     # CI smoke test
set -euo pipefail
cd "$(dirname "$0")/.."

QUICK=false
for arg in "$@"; do [[ "$arg" == "--quick" ]] && QUICK=true; done

if $QUICK; then
  PRETRAIN_EPOCHS=5; RLFT_EPOCHS=3; N_SAMPLES=50; LIMIT="--limit 100"
else
  PRETRAIN_EPOCHS=800; RLFT_EPOCHS=60; N_SAMPLES=300; LIMIT=""
fi

PRETRAIN_SEED=42
PRETRAIN_RUN="pretrain_nb101_seed${PRETRAIN_SEED}"
SEEDS=(42 123 456)

# ===================================================================
# Phase 1: Pretrain
# ===================================================================
echo "=== Phase 1: Pretraining NB101 ==="
python src/train_pretrain.py --config configs/pretrain_nb101.yaml \
  --epochs "$PRETRAIN_EPOCHS" $LIMIT --seed "$PRETRAIN_SEED"

PRETRAIN_CKPT="outputs/${PRETRAIN_RUN}/checkpoints/best.pt"
if [[ ! -f "$PRETRAIN_CKPT" ]]; then
  echo "ERROR: Pretrain checkpoint not found: $PRETRAIN_CKPT"
  exit 1
fi

# ===================================================================
# Baselines
# ===================================================================
echo "=== Computing baselines ==="
python src/baselines.py --mode random --benchmark nb101 --n-samples "$N_SAMPLES" \
  --seeds "${SEEDS[@]}"
python src/baselines.py --mode pretrained --checkpoint "$PRETRAIN_CKPT" \
  --benchmark nb101 --n-samples "$N_SAMPLES" --seeds "${SEEDS[@]}"

# ===================================================================
# Phase 2: RL-FT for each seed
# ===================================================================
echo "=== Phase 2: RL-FT NB101 ==="
for SEED in "${SEEDS[@]}"; do
  echo "  seed=${SEED}"
  python src/train_rlft.py --config configs/rlft_nb101.yaml \
    --resume "$PRETRAIN_CKPT" --epochs "$RLFT_EPOCHS" --seed "$SEED"
done

# ===================================================================
# Phase 3: Evaluate each seed
# ===================================================================
echo "=== Evaluation ==="
for SEED in "${SEEDS[@]}"; do
  RLFT_RUN="rlft_nb101_seed${SEED}"
  RLFT_CKPT="outputs/${RLFT_RUN}/checkpoints/best.pt"
  if [[ ! -f "$RLFT_CKPT" ]]; then
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
    --config configs/rlft_nb101.yaml \
    --n-samples "$N_SAMPLES" --seed "$SEED" \
    --output "outputs/eval_${RLFT_RUN}/results.json"
done

# ===================================================================
# Phase 4: Aggregate and compare
# ===================================================================
echo "=== Aggregation & Pass/Fail ==="
EXP1_STAGING="outputs/_exp1_nb101_staging"
rm -rf "$EXP1_STAGING"
mkdir -p "$EXP1_STAGING"
for SEED in "${SEEDS[@]}"; do
  RLFT_RUN="rlft_nb101_seed${SEED}"
  DEST_DIR="${EXP1_STAGING}/eval_${RLFT_RUN}"
  mkdir -p "$DEST_DIR"
  cp "outputs/eval_${RLFT_RUN}/results.json" "$DEST_DIR/results.json"
done

python src/evaluate.py --aggregate \
  --results-dir "$EXP1_STAGING" \
  --pattern "eval_*/results.json" \
  --expected expected_outputs/experiment_1_nb101.json \
  --output outputs/experiment_1_nb101_summary.json

echo "=== Done. Summary: outputs/experiment_1_nb101_summary.json ==="
