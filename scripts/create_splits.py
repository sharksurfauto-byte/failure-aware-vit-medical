"""
Create stratified train/val/test splits for NIH Malaria Dataset.

Split strategy:
- Train: 70% (~19,291 images)
- Validation: 15% (~4,134 images)
- Test: 15% (~4,134 images)
  - Clean test: 75% (~3,100 images)
  - Stress source: 25% (~1,034 images)

Output: data/splits.json with image paths and split assignments
"""

import json
import random
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np

# Reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Paths (project-root relative)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "cell_images"
OUTPUT_PATH = PROJECT_ROOT / "data" / "splits.json"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15  # Will be further split into clean (0.75) and stress (0.25)

# Small image threshold
SMALL_IMAGE_THRESHOLD = 80


def load_image_paths():
    """Load all image paths from both classes"""
    # Validate data directory exists
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"ERROR: Dataset not found at {DATA_ROOT}\n"
            f"Did you run 'bash scripts/download_dataset.sh'?"
        )
    
    parasitized_dir = DATA_ROOT / "Parasitized"
    uninfected_dir = DATA_ROOT / "Uninfected"
    
    if not parasitized_dir.exists() or not uninfected_dir.exists():
        raise FileNotFoundError(
            f"ERROR: Expected subdirectories not found:\n"
            f"  Parasitized: {parasitized_dir.exists()}\n"
            f"  Uninfected: {uninfected_dir.exists()}\n"
            f"Check dataset structure."
        )
    
    parasitized_paths = list(parasitized_dir.glob("*.png"))
    uninfected_paths = list(uninfected_dir.glob("*.png"))
    
    if len(parasitized_paths) == 0 or len(uninfected_paths) == 0:
        raise ValueError(
            f"ERROR: No images found!\n"
            f"  Parasitized: {len(parasitized_paths)}\n"
            f"  Uninfected: {len(uninfected_paths)}"
        )
    
    print(f"Loaded {len(parasitized_paths)} parasitized images")
    print(f"Loaded {len(uninfected_paths)} uninfected images")
    print(f"Total: {len(parasitized_paths) + len(uninfected_paths)}")
    
    return parasitized_paths, uninfected_paths


def identify_small_images(image_paths):
    """Identify images with min(H, W) < threshold"""
    small_images = []
    
    print(f"\nScanning for small images (min dimension < {SMALL_IMAGE_THRESHOLD}px)...")
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w, _ = img.shape
            if h < SMALL_IMAGE_THRESHOLD or w < SMALL_IMAGE_THRESHOLD:
                small_images.append(str(img_path))
    
    print(f"Found {len(small_images)} small images ({len(small_images)/len(image_paths)*100:.2f}%)")
    return small_images


def stratified_split(paths, train_ratio, val_ratio, test_ratio):
    """Create stratified splits maintaining class balance"""
    # Shuffle
    paths_copy = paths.copy()
    random.shuffle(paths_copy)
    
    # Calculate split indices
    n = len(paths_copy)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train = paths_copy[:train_end]
    val = paths_copy[train_end:val_end]
    test = paths_copy[val_end:]
    
    return train, val, test


def split_test_into_clean_and_stress(test_paths):
    """Split test set into clean (75%) and stress source (25%)"""
    test_copy = test_paths.copy()
    random.shuffle(test_copy)
    
    n = len(test_copy)
    clean_end = int(n * 0.75)
    
    clean_test = test_copy[:clean_end]
    stress_source = test_copy[clean_end:]
    
    return clean_test, stress_source


def create_splits():
    """Main split creation logic"""
    print("=" * 60)
    print("CREATING STRATIFIED SPLITS")
    print("=" * 60)
    
    # Load paths
    parasitized_paths, uninfected_paths = load_image_paths()
    
    # Identify small images (for tracking)
    all_paths = parasitized_paths + uninfected_paths
    small_images = identify_small_images(all_paths)
    
    # Create stratified splits per class
    print("\nCreating stratified splits...")
    para_train, para_val, para_test = stratified_split(
        parasitized_paths, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )
    uninf_train, uninf_val, uninf_test = stratified_split(
        uninfected_paths, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )
    
    # Combine classes
    train_set = para_train + uninf_train
    val_set = para_val + uninf_val
    test_set = para_test + uninf_test
    
    # Split test into clean and stress source
    clean_test, stress_source = split_test_into_clean_and_stress(test_set)
    
    # Convert to relative POSIX paths from PROJECT_ROOT for cross-platform compatibility
    # This ensures splits.json works on both Windows (local) and Linux (Colab)
    def make_relative_posix(p):
        """Convert absolute path to relative POSIX path from project root"""
        rel_path = Path(p).relative_to(PROJECT_ROOT)
        return rel_path.as_posix()
    
    train_set = [make_relative_posix(p) for p in train_set]
    val_set = [make_relative_posix(p) for p in val_set]
    clean_test = [make_relative_posix(p) for p in clean_test]
    stress_source = [make_relative_posix(p) for p in stress_source]
    small_images = [make_relative_posix(p) for p in small_images]
    
    # Print statistics
    print("\n" + "=" * 60)
    print("SPLIT STATISTICS")
    print("=" * 60)
    print(f"Train:        {len(train_set):>6} ({len(train_set)/len(all_paths)*100:.1f}%)")
    print(f"Validation:   {len(val_set):>6} ({len(val_set)/len(all_paths)*100:.1f}%)")
    print(f"Test (clean): {len(clean_test):>6} ({len(clean_test)/len(all_paths)*100:.1f}%)")
    print(f"Stress source:{len(stress_source):>6} ({len(stress_source)/len(all_paths)*100:.1f}%)")
    print(f"{'Total:':<14}{len(train_set) + len(val_set) + len(clean_test) + len(stress_source):>6}")
    
    # Verify no leakage
    all_splits = set(train_set + val_set + clean_test + stress_source)
    assert len(all_splits) == len(all_paths), "ERROR: Duplicate images across splits!"
    print("\n[OK] Verified: No data leakage between splits")
    
    # Count small images per split
    small_set = set(small_images)
    small_in_train = len([p for p in train_set if p in small_set])
    small_in_val = len([p for p in val_set if p in small_set])
    small_in_test = len([p for p in clean_test + stress_source if p in small_set])
    
    print(f"\nSmall images distribution:")
    print(f"  Train: {small_in_train}")
    print(f"  Val:   {small_in_val}")
    print(f"  Test:  {small_in_test}")
    
    # Create output structure
    splits_data = {
        "random_seed": RANDOM_SEED,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
            "test_clean_ratio": 0.75,
            "test_stress_ratio": 0.25
        },
        "splits": {
            "train": train_set,
            "val": val_set,
            "test_clean": clean_test,
            "stress_source": stress_source
        },
        "metadata": {
            "total_images": len(all_paths),
            "small_images": small_images,
            "small_image_threshold": SMALL_IMAGE_THRESHOLD
        }
    }
    
    # Save to JSON
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(splits_data, f, indent=2)
    
    print(f"\n[OK] Saved splits to {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    create_splits()
