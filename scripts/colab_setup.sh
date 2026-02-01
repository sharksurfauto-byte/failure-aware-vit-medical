#!/bin/bash
# Colab setup script - one command to prepare environment
# Run after: git clone https://github.com/sharksurfauto-byte/failure-aware-vit-medical.git

set -e  # Exit on error

echo "=========================================="
echo "COLAB SETUP: Failure-Aware ViT Medical"
echo "=========================================="

# GPU check
echo ""
echo "=== Checking GPU availability ==="
python -c "import torch; assert torch.cuda.is_available(), 'ERROR: No GPU detected! Enable GPU runtime in Colab.'; print(f'GPU detected: {torch.cuda.get_device_name(0)}')"

# Install dependencies
echo ""
echo "=== Installing dependencies ==="
pip install -q -r requirements.txt
echo "[OK] Dependencies installed"

# Download dataset
echo ""
echo "=== Downloading dataset ==="
bash scripts/download_dataset.sh

# Generate splits
echo ""
echo "=== Generating train/val/test splits ==="
python scripts/create_splits.py || {
    echo "Error: Failed to create splits"
    exit 1
}

# Compute normalization stats
echo ""
echo "=== Computing normalization statistics ==="
python -c "from src.dataset import initialize_normalization_stats; initialize_normalization_stats()" || {
    echo "Error: Failed to compute normalization stats"
    exit 1
}

echo ""
echo "=========================================="
echo "[OK] Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Smoke test:  bash scripts/smoke_test.sh"
echo "  2. Train CNN:   bash scripts/train_cnn.sh"
echo "  3. Train ViT:   bash scripts/train_vit.sh"
echo ""
