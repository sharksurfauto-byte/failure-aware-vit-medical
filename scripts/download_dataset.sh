#!/bin/bash
# Download NIH Malaria Cell Images Dataset
# Usage: bash scripts/download_dataset.sh

set -e  # Exit on error

DATASET_URL="https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"
OUTPUT_DIR="data/raw"

echo "=== Downloading NIH Malaria Dataset ==="
echo "URL: $DATASET_URL"
echo "Output: $OUTPUT_DIR"

# Create output directory
mkdir -p $OUTPUT_DIR

# Download dataset
echo "Downloading... (this may take 2-4 minutes)"
wget -q --show-progress -O cell_images.zip $DATASET_URL || {
    echo "Error: Failed to download dataset"
    exit 1
}

# Extract
echo "Extracting..."
unzip -q cell_images.zip -d $OUTPUT_DIR || {
    echo "Error: Failed to extract dataset"
    exit 1
}

# Cleanup
rm cell_images.zip

echo "[OK] Dataset downloaded to $OUTPUT_DIR"
echo "Expected structure: $OUTPUT_DIR/cell_images/{Parasitized,Uninfected}"
