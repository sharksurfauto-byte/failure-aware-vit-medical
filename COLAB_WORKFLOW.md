# Colab Execution Workflow

## Overview

This project uses **Google Colab as a remote GPU compute environment**. All code lives in **scripts and modules**, not notebooks. This document provides exact commands for reproducible training.

---

## Prerequisites

- **GitHub**: Code is version-controlled (this repo)
- **Colab**: Free tier (T4 GPU, 16GB VRAM)
- **Runtime**: Enable GPU in Colab: `Runtime` → `Change runtime type` → `T4 GPU`

---

## Quick Start (3 Commands)

```bash
# 1. Clone repo
!git clone https://github.com/sharksurfauto-byte/failure-aware-vit-medical.git
%cd failure-aware-vit-medical

# 2. Setup environment (installs deps, downloads dataset, generates splits)
!bash scripts/colab_setup.sh

# 3. Run smoke test (validates pipeline in 2 minutes)
!bash scripts/smoke_test.sh
```

**If smoke test passes**, you're ready for full training.

---

## Full Training Workflow

### 1. Train CNN Baseline

```bash
!bash scripts/train_cnn.sh
```

**Expected output**:
- Training time: ~20-30 minutes
- Checkpoint: `checkpoints/cnn_best.pt`
- Validation accuracy: >90%

### 2. Train ViT Baseline

```bash
!bash scripts/train_vit.sh
```

**Expected output**:
- Training time: ~30-45 minutes (slower than CNN)
- Checkpoint: `checkpoints/vit_best.pt`
- Validation accuracy: comparable to CNN

**If you get CUDA OOM error**:
Edit `scripts/train_vit.sh` and change `--batch-size 64` to `--batch-size 32`.

---

## Downloading Checkpoints

After training completes:

```bash
# Package checkpoints
!bash scripts/download_checkpoints.sh

# Download to local machine
from google.colab import files
files.download('checkpoints.tar.gz')
```

Extract locally:
```bash
tar -xzf checkpoints.tar.gz
```

---

## Training Configuration (Locked)

Both models use **identical training setup**:

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 0.01 |
| Batch size | 64 (CNN), 64→32 if OOM (ViT) |
| Max epochs | 20 |
| Early stopping | 5 epochs patience |
| LR scheduler | CosineAnnealingLR |

**Rationale**: Fair comparison requires identical training budget.

---

## Manual Execution (Advanced)

If you want finer control:

### Setup Steps (Expanded)

```bash
# Clone repo
!git clone https://github.com/sharksurfauto-byte/failure-aware-vit-medical.git
%cd failure-aware-vit-medical

# Verify GPU
!python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Install dependencies
!pip install -q -r requirements.txt

# Download dataset (~300MB, 2-4 minutes)
!bash scripts/download_dataset.sh

# Generate train/val/test splits
!python scripts/create_splits.py

# Compute normalization stats (train split only)
!python -c "from src.dataset import initialize_normalization_stats; initialize_normalization_stats()"
```

### Custom Training Arguments

```bash
!python scripts/train.py \
  --model vit \
  --epochs 10 \
  --batch-size 32 \
  --lr 1e-4 \
  --weight-decay 0.05 \
  --patience 3 \
  --device cuda \
  --checkpoint-dir custom_checkpoints
```

Run `!python scripts/train.py --help` for all options.

---

## Troubleshooting

### Error: "No GPU detected"

**Cause**: GPU runtime not enabled.

**Fix**:
1. In Colab: `Runtime` → `Change runtime type`
2. Select `T4 GPU`
3. Click `Save`
4. Re-run setup

### Error: "Dataset not found"

**Cause**: `download_dataset.sh` failed or was skipped.

**Fix**:
```bash
!bash scripts/download_dataset.sh
```

### Error: "CUDA out of memory"

**Cause**: Batch size too large for GPU.

**Fix for ViT**:
Edit `scripts/train_vit.sh`, line 25:
```bash
--batch-size 32  # Changed from 64
```

---

## Dataset Details

**Source**: NIH Malaria Cell Images  
**URL**: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets

**Splits** (deterministic, seed=42):
- Train: 70% (~19,291 images)
- Validation: 15% (~4,134 images)
- Test (clean): 11.25% (~3,100 images)
- Test (stress): 3.75% (~1,034 images)

**Preprocessing**:
1. Aspect-ratio preserving resize (shorter side → 224px)
2. Symmetric black padding to 224×224
3. Normalize with dataset-specific mean/std

---

## Reproducibility Checklist

✅ Code version-controlled in GitHub  
✅ `splits.json` committed (deterministic splits)  
✅ `normalization_stats.json` committed (fixed preprocessing)  
✅ Random seed locked (42)  
✅ Hyperparameters locked in training scripts  

**Result**: Anyone can reproduce exact results by:
1. Cloning this repo
2. Running `bash scripts/colab_setup.sh`
3. Running `bash scripts/train_cnn.sh` and `bash scripts/train_vit.sh`

---

## Development Workflow

**Local machine**:
- Write code
- Commit to Git
- Push to GitHub

**Colab**:
- Pull latest code
- Run training scripts
- Download checkpoints

**Never edit code in Colab notebooks**. All logic lives in `.py` files.

---

## Next Steps After Training

Once baselines are validated:
1. Evaluate models (accuracy + calibration)
2. Compare CNN vs ViT
3. Proceed to Stage 3: Uncertainty estimation

See main README.md for full project roadmap.
