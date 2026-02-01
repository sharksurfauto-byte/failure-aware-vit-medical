#!/bin/bash
# Package checkpoints for download from Colab
# Usage: bash scripts/download_checkpoints.sh

set -e

echo "=== Packaging checkpoints ==="

if [ ! -d "checkpoints" ]; then
    echo "Error: checkpoints/ directory not found"
    exit 1
fi

# Create archive
tar -czf checkpoints.tar.gz checkpoints/

echo "[OK] Checkpoints packaged to: checkpoints.tar.gz"
echo ""
echo "To download in Colab:"
echo "  from google.colab import files"
echo "  files.download('checkpoints.tar.gz')"
