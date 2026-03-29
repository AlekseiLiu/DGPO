#!/bin/bash
# Experiment 2: OOD Discovery via Filtered Pretraining — NB201 (Table 2, Fig 3)
# Threshold: 0.85 (CIFAR-10 only)
#
# Usage:
#   ./scripts/run_experiment_2_nb201.sh            # full run
#   ./scripts/run_experiment_2_nb201.sh --quick     # CI smoke test
set -euo pipefail
cd "$(dirname "$0")/.."

QUICK=false
for arg in "$@"; do [[ "$arg" == "--quick" ]] && QUICK=true; done

if $QUICK; then
  PRETRAIN_EPOCHS=5; RLFT_EPOCHS=3; N_SAMPLES=50; LIMIT="--limit 100"
else
  PRETRAIN_EPOCHS=500; RLFT_EPOCHS=80; N_SAMPLES=300; LIMIT=""
fi

PRETRAIN_SEED=42
PRETRAIN_RUN="pretrain_nb201_filtered_seed${PRETRAIN_SEED}"
SEEDS=(42 123 456)
THRESHOLD=0.85

# ===================================================================
# Phase 1: Filtered pretrain
# ===================================================================
echo "=== Phase 1: Filtered Pretraining NB201 (threshold=${THRESHOLD}) ==="
python src/train_pretrain.py --config configs/pretrain_nb201_filtered.yaml \
  --epochs "$PRETRAIN_EPOCHS" $LIMIT --seed "$PRETRAIN_SEED"

PRETRAIN_CKPT="outputs/${PRETRAIN_RUN}/checkpoints/best.pt"
if [[ ! -f "$PRETRAIN_CKPT" ]]; then
  echo "ERROR: Pretrain checkpoint not found: $PRETRAIN_CKPT"
  exit 1
fi

# ===================================================================
# Baselines (filtered pretrain variant for OOD reference)
# ===================================================================
echo "=== Computing baselines ==="
python src/baselines.py --mode random --benchmark nb201 --n-samples "$N_SAMPLES" \
  --seeds "${SEEDS[@]}"
python src/baselines.py --mode pretrained --checkpoint "$PRETRAIN_CKPT" \
  --benchmark nb201 --variant filtered --n-samples "$N_SAMPLES" --seeds "${SEEDS[@]}"

# ===================================================================
# Phase 2: RL-FT for each seed
# ===================================================================
echo "=== Phase 2: RL-FT NB201 filtered ==="
for SEED in "${SEEDS[@]}"; do
  echo "  seed=${SEED}"
  python src/train_rlft.py --config configs/rlft_nb201_filtered.yaml \
    --resume "$PRETRAIN_CKPT" --epochs "$RLFT_EPOCHS" --seed "$SEED"
done

# ===================================================================
# Phase 3: Evaluate each seed (with OOD metrics)
# ===================================================================
echo "=== Evaluation ==="
for SEED in "${SEEDS[@]}"; do
  RLFT_RUN="rlft_nb201_filtered_seed${SEED}"
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
  # Pass pretrain baseline per-seed file for OOD discovery rate computation
  PRETRAIN_BASELINE="outputs/baselines/pretrained_nb201_filtered_seed${SEED}.json"
  PRETRAIN_ARG=""
  if [[ -f "$PRETRAIN_BASELINE" ]]; then
    PRETRAIN_ARG="--pretrain-samples ${PRETRAIN_BASELINE}"
  else
    echo "WARN: Pretrain baseline not found: $PRETRAIN_BASELINE (ood_discovery_rate will be null)"
  fi
  python src/evaluate.py \
    --checkpoint "$RLFT_CKPT" \
    --config configs/rlft_nb201_filtered.yaml \
    --n-samples "$N_SAMPLES" --seed "$SEED" \
    --filter-threshold "$THRESHOLD" \
    $PRETRAIN_ARG \
    --output "outputs/eval_${RLFT_RUN}/results.json"
done

# ===================================================================
# Phase 4: Aggregate and compare
# ===================================================================
echo "=== Aggregation & Pass/Fail ==="
EXP2_STAGING="outputs/_exp2_nb201_staging"
rm -rf "$EXP2_STAGING"
mkdir -p "$EXP2_STAGING"
for SEED in "${SEEDS[@]}"; do
  RLFT_RUN="rlft_nb201_filtered_seed${SEED}"
  DEST_DIR="${EXP2_STAGING}/eval_${RLFT_RUN}"
  mkdir -p "$DEST_DIR"
  cp "outputs/eval_${RLFT_RUN}/results.json" "$DEST_DIR/results.json"
done

python src/evaluate.py --aggregate \
  --results-dir "$EXP2_STAGING" \
  --pattern "eval_*/results.json" \
  --expected expected_outputs/experiment_2_nb201.json \
  --output outputs/experiment_2_nb201_summary.json

echo "=== Done. Summary: outputs/experiment_2_nb201_summary.json ==="
