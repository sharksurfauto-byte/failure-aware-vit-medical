#!/bin/bash
# Colab Evaluation Script
# Run baseline evaluation on test_clean split

echo "============================================================"
echo "BASELINE MODEL EVALUATION"
echo "============================================================"

# Check if checkpoints are available in Drive
if [ ! -f "/content/drive/MyDrive/failure-aware-vit-medical/checkpoints/cnn_best.pt" ]; then
    echo "ERROR: CNN checkpoint not found in Drive"
    echo "Expected: /content/drive/MyDrive/failure-aware-vit-medical/checkpoints/cnn_best.pt"
    exit 1
fi

if [ ! -f "/content/drive/MyDrive/failure-aware-vit-medical/checkpoints/vit_best.pt" ]; then
    echo "ERROR: ViT checkpoint not found in Drive"
    echo "Expected: /content/drive/MyDrive/failure-aware-vit-medical/checkpoints/vit_best.pt"
    exit 1
fi

echo "✓ Checkpoints found in Drive"
echo ""

# Run evaluation
python scripts/evaluate_baselines.py \
  --cnn-checkpoint /content/drive/MyDrive/failure-aware-vit-medical/checkpoints/cnn_best.pt \
  --vit-checkpoint /content/drive/MyDrive/failure-aware-vit-medical/checkpoints/vit_best.pt \
  --batch-size 64 \
  --device cuda \
  --output /content/drive/MyDrive/failure-aware-vit-medical/evaluation_results.json

echo ""
echo "============================================================"
echo "Evaluation complete!"
echo "Results saved to Drive: evaluation_results.json"
echo "============================================================"
