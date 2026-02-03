#!/bin/bash
# Colab MC Dropout Evaluation Script

echo "============================================================"
echo "MC DROPOUT UNCERTAINTY ESTIMATION"
echo "============================================================"

# Check if checkpoint is available
if [ ! -f "/content/drive/MyDrive/failure-aware-vit-medical/checkpoints/vit_best.pt" ]; then
    echo "ERROR: ViT checkpoint not found in Drive"
    exit 1
fi

echo "✓ Checkpoint found in Drive"
echo ""

# Create output directory in Drive
mkdir -p /content/drive/MyDrive/failure-aware-vit-medical/mc_dropout_results

# Run MC Dropout evaluation
python scripts/mc_dropout_eval.py \
  --checkpoint /content/drive/MyDrive/failure-aware-vit-medical/checkpoints/vit_best.pt \
  --num-samples 20 \
  --batch-size 64 \
  --device cuda \
  --output-dir /content/drive/MyDrive/failure-aware-vit-medical/mc_dropout_results

echo ""
echo "============================================================"
echo "MC Dropout evaluation complete!"
echo "Results saved to Drive: mc_dropout_results/"
echo "============================================================"
