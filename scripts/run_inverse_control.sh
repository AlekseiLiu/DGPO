#!/bin/bash
# Bonus: Bidirectional Control (Fig 4)
# Runs both forward (normal) and inverse (negated reward) RL-FT on NB101 and NB201.
#
# Usage:
#   ./scripts/run_inverse_control.sh            # full run
#   ./scripts/run_inverse_control.sh --quick     # CI smoke test
set -euo pipefail
cd "$(dirname "$0")/.."

QUICK=false
for arg in "$@"; do [[ "$arg" == "--quick" ]] && QUICK=true; done

if $QUICK; then
  PRETRAIN_EPOCHS_101=5; PRETRAIN_EPOCHS_201=5
  RLFT_EPOCHS_101=3; RLFT_EPOCHS_201=3
  N_SAMPLES=50; LIMIT="--limit 100"
else
  PRETRAIN_EPOCHS_101=800; PRETRAIN_EPOCHS_201=2000
  RLFT_EPOCHS_101=60; RLFT_EPOCHS_201=80
  N_SAMPLES=300; LIMIT=""
fi

PRETRAIN_SEED=42
SEEDS=(42 123 456)

# Helper: resolve best.pt or fallback to last epoch checkpoint
find_ckpt() {
  local run_dir="$1" max_epoch="$2"
  local best="outputs/${run_dir}/checkpoints/best.pt"
  if [[ -f "$best" ]]; then echo "$best"; return; fi
  local last_epoch
  last_epoch=$(printf "%03d" "$max_epoch")
  local fallback="outputs/${run_dir}/checkpoints/epoch_${last_epoch}.pt"
  if [[ -f "$fallback" ]]; then
    echo "WARN: best.pt not found, using ${fallback}" >&2
    echo "$fallback"; return
  fi
  echo "ERROR: No checkpoint for ${run_dir}" >&2; return 1
}

# ===================================================================
# NB101: Pretrain + Forward + Inverse
# ===================================================================
echo "=== NB101: Pretrain ==="
python src/train_pretrain.py --config configs/pretrain_nb101.yaml \
  --epochs "$PRETRAIN_EPOCHS_101" $LIMIT --seed "$PRETRAIN_SEED"

NB101_PRETRAIN=$(find_ckpt "pretrain_nb101_seed${PRETRAIN_SEED}" $((PRETRAIN_EPOCHS_101 - 1)))

echo "=== NB101: Baselines ==="
python src/baselines.py --mode random --benchmark nb101 --n-samples "$N_SAMPLES" \
  --seeds "${SEEDS[@]}"
python src/baselines.py --mode pretrained --checkpoint "$NB101_PRETRAIN" \
  --benchmark nb101 --n-samples "$N_SAMPLES" --seeds "${SEEDS[@]}"

echo "=== NB101: Forward RL-FT ==="
for SEED in "${SEEDS[@]}"; do
  python src/train_rlft.py --config configs/rlft_nb101.yaml \
    --resume "$NB101_PRETRAIN" --epochs "$RLFT_EPOCHS_101" --seed "$SEED"
done

echo "=== NB101: Inverse RL-FT ==="
for SEED in "${SEEDS[@]}"; do
  python src/train_rlft.py --config configs/rlft_nb101_inverse.yaml \
    --resume "$NB101_PRETRAIN" --epochs "$RLFT_EPOCHS_101" --seed "$SEED"
done

# ===================================================================
# NB201: Pretrain + Forward + Inverse
# ===================================================================
echo "=== NB201: Pretrain ==="
python src/train_pretrain.py --config configs/pretrain_nb201.yaml \
  --epochs "$PRETRAIN_EPOCHS_201" $LIMIT --seed "$PRETRAIN_SEED"

NB201_PRETRAIN=$(find_ckpt "pretrain_nb201_seed${PRETRAIN_SEED}" $((PRETRAIN_EPOCHS_201 - 1)))

echo "=== NB201: Baselines ==="
python src/baselines.py --mode random --benchmark nb201 --n-samples "$N_SAMPLES" \
  --seeds "${SEEDS[@]}"
python src/baselines.py --mode pretrained --checkpoint "$NB201_PRETRAIN" \
  --benchmark nb201 --n-samples "$N_SAMPLES" --seeds "${SEEDS[@]}"

echo "=== NB201: Forward RL-FT ==="
for SEED in "${SEEDS[@]}"; do
  python src/train_rlft.py --config configs/rlft_nb201_cifar10.yaml \
    --resume "$NB201_PRETRAIN" --epochs "$RLFT_EPOCHS_201" --seed "$SEED"
done

echo "=== NB201: Inverse RL-FT ==="
for SEED in "${SEEDS[@]}"; do
  python src/train_rlft.py --config configs/rlft_nb201_inverse.yaml \
    --resume "$NB201_PRETRAIN" --epochs "$RLFT_EPOCHS_201" --seed "$SEED"
done

# ===================================================================
# Evaluate all runs
# ===================================================================
echo "=== Evaluation ==="

for SEED in "${SEEDS[@]}"; do
  # NB101 forward
  RLFT_CKPT=$(find_ckpt "rlft_nb101_seed${SEED}" $((RLFT_EPOCHS_101 - 1)))
  python src/evaluate.py --checkpoint "$RLFT_CKPT" --config configs/rlft_nb101.yaml \
    --n-samples "$N_SAMPLES" --seed "$SEED" \
    --output "outputs/eval_rlft_nb101_seed${SEED}/results.json"

  # NB101 inverse
  RLFT_CKPT=$(find_ckpt "rlft_nb101_inverse_seed${SEED}" $((RLFT_EPOCHS_101 - 1)))
  python src/evaluate.py --checkpoint "$RLFT_CKPT" --config configs/rlft_nb101_inverse.yaml \
    --n-samples "$N_SAMPLES" --seed "$SEED" \
    --output "outputs/eval_rlft_nb101_inverse_seed${SEED}/results.json"

  # NB201 forward
  RLFT_CKPT=$(find_ckpt "rlft_nb201_cifar10_seed${SEED}" $((RLFT_EPOCHS_201 - 1)))
  python src/evaluate.py --checkpoint "$RLFT_CKPT" --config configs/rlft_nb201_cifar10.yaml \
    --n-samples "$N_SAMPLES" --seed "$SEED" \
    --output "outputs/eval_rlft_nb201_cifar10_fwd_seed${SEED}/results.json"

  # NB201 inverse
  RLFT_CKPT=$(find_ckpt "rlft_nb201_inverse_seed${SEED}" $((RLFT_EPOCHS_201 - 1)))
  python src/evaluate.py --checkpoint "$RLFT_CKPT" --config configs/rlft_nb201_inverse.yaml \
    --n-samples "$N_SAMPLES" --seed "$SEED" \
    --output "outputs/eval_rlft_nb201_inverse_seed${SEED}/results.json"
done

echo "=== Done. Inverse control results in outputs/eval_rlft_*_inverse_*/ ==="
echo "Use train_log.jsonl files from forward + inverse runs to reproduce Fig 4."
