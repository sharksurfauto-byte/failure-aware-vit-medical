#!/bin/bash
# Train CNN baseline with locked hyperparameters

set -e

echo "=========================================="
echo "TRAINING: CNN Baseline"
echo "=========================================="
echo "Config:"
echo "  - Epochs: 20 (max, early stopping @ 5)"
echo "  - Batch size: 64"
echo "  - Learning rate: 3e-4"
echo "  - Optimizer: AdamW (weight_decay=0.01)"
echo "  - Scheduler: CosineAnnealingLR"
echo ""

python scripts/train.py \
  --model cnn \
  --epochs 20 \
  --batch-size 64 \
  --lr 3e-4 \
  --weight-decay 0.01 \
  --patience 5 \
  --device cuda \
  --checkpoint-dir checkpoints

echo ""
echo "[OK] Training complete!"
echo "Checkpoint saved to: checkpoints/cnn_best.pt"
