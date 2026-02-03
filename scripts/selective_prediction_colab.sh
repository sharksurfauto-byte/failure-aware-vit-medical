#!/bin/bash
# Colab Selective Prediction Analysis Script

echo "============================================================"
echo "SELECTIVE PREDICTION ANALYSIS"
echo "============================================================"

# Check if MC Dropout results exist
if [ ! -f "/content/drive/MyDrive/failure-aware-vit-medical/mc_dropout_results/mc_dropout_results.json" ]; then
    echo "ERROR: MC Dropout results not found"
    echo "Run mc_dropout_colab.sh first"
    exit 1
fi

echo "✓ MC Dropout results found"
echo ""

# Create output directory
mkdir -p /content/drive/MyDrive/failure-aware-vit-medical/selective_prediction_results

# Run selective prediction analysis
python scripts/selective_prediction_eval.py \
  --results /content/drive/MyDrive/failure-aware-vit-medical/mc_dropout_results/mc_dropout_results.json \
  --output-dir /content/drive/MyDrive/failure-aware-vit-medical/selective_prediction_results \
  --rejection-levels 10 15 20 25 30

echo ""
echo "============================================================"
echo "Analysis complete!"
echo "Results saved to Drive: selective_prediction_results/"
echo "============================================================"
