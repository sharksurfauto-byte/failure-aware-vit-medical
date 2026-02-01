#!/bin/bash
# Train ViT baseline with locked hyperparameters

set -e

echo "=========================================="
echo "TRAINING: ViT Baseline"
echo "=========================================="
echo "Config:"
echo "  - Epochs: 20 (max, early stopping @ 5)"
echo "  - Batch size: 64 (fallback to 32 if OOM)"
echo "  - Learning rate: 3e-4"
echo "  - Optimizer: AdamW (weight_decay=0.01)"
echo "  - Scheduler: CosineAnnealingLR"
echo ""
echo "Model:"
echo "  - Patch size: 16x16"
echo "  - Embed dim: 384"
echo "  - Layers: 6"
echo "  - Heads: 6"
echo "  - Params: ~5M"
echo ""

python scripts/train.py \
  --model vit \
  --epochs 20 \
  --batch-size 64 \
  --lr 3e-4 \
  --weight-decay 0.01 \
  --patience 5 \
  --device cuda \
  --checkpoint-dir checkpoints

echo ""
echo "[OK] Training complete!"
echo "Checkpoint saved to: checkpoints/vit_best.pt"
