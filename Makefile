.PHONY: help setup build-nb201-cache pretrain-nb101 pretrain-nb201 rlft-nb101 rlft-nb201 evaluate clean

PYTHON ?= python

help: ## Show this help message
	@echo "DGPO Companion — Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo ""

setup: ## Install dependencies and download data
	@echo "==> Running setup script..."
	PYTHON="$(PYTHON)" bash scripts/setup.sh

NB201_API ?= data/NASBench201/raw/NAS-Bench-201-v1_1-096897.pth
NB201_CACHE ?= data/NASBench201/raw/nb201_unified.pt

build-nb201-cache: ## Build unified NB201 cache (one-time, requires ~13GB RAM)
	@echo "==> Building unified NB201 cache..."
	$(PYTHON) scripts/build_nb201_cache.py --api-path "$(NB201_API)" --output "$(NB201_CACHE)"

pretrain-nb101: ## Pretrain diffusion model on NAS-Bench-101
	@echo "==> Starting NB101 pretraining..."
	$(PYTHON) src/train_pretrain.py --config configs/pretrain_nb101.yaml

pretrain-nb201: ## Pretrain diffusion model on NAS-Bench-201
	@echo "==> Starting NB201 pretraining..."
	$(PYTHON) src/train_pretrain.py --config configs/pretrain_nb201.yaml

PRETRAIN_NB101_CKPT ?= outputs/pretrain_nb101_seed42/checkpoints/best.pt
PRETRAIN_NB201_CKPT ?= outputs/pretrain_nb201_seed42/checkpoints/best.pt

rlft-nb101: ## RL fine-tune (DGPO) on NAS-Bench-101
	@echo "==> Starting NB101 RL fine-tuning..."
	$(PYTHON) src/train_rlft.py --config configs/rlft_nb101.yaml --resume "$(PRETRAIN_NB101_CKPT)"

rlft-nb201: ## RL fine-tune (DGPO) on NAS-Bench-201 (CIFAR-10)
	@echo "==> Starting NB201 RL fine-tuning..."
	$(PYTHON) src/train_rlft.py --config configs/rlft_nb201_cifar10.yaml --resume "$(PRETRAIN_NB201_CKPT)"

EVAL_CKPT ?= outputs/rlft_nb201_cifar10_seed42/checkpoints/best.pt
EVAL_CONFIG ?= configs/rlft_nb201_cifar10.yaml
EVAL_SAMPLES ?= 300

evaluate: ## Evaluate a trained checkpoint (set EVAL_CKPT, EVAL_CONFIG, EVAL_SAMPLES)
	@echo "==> Running evaluation..."
	$(PYTHON) src/evaluate.py --checkpoint "$(EVAL_CKPT)" --config "$(EVAL_CONFIG)" --n-samples $(EVAL_SAMPLES)

clean: ## Remove output artifacts
	@echo "==> Cleaning outputs..."
	rm -rf outputs/
