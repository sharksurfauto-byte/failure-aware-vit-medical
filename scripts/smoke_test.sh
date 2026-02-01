#!/bin/bash
# Smoke test - quick validation of training pipeline
# 2 epochs, small batch size, CNN model

set -e

echo "=========================================="
echo "SMOKE TEST: CNN Baseline"
echo "=========================================="
echo "Purpose: Validate pipeline (fast)"
echo "Config: 2 epochs, batch=16"
echo ""

python scripts/train.py \
  --model cnn \
  --epochs 2 \
  --batch-size 16 \
  --lr 3e-4 \
  --weight-decay 0.01 \
  --patience 5 \
  --device cuda \
  --checkpoint-dir checkpoints

echo ""
echo "[OK] Smoke test complete!"
echo "If this succeeded, you're ready for full training."
