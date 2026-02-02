# Colab Execution Workflow

## Overview

This project uses **Google Colab as a remote GPU compute environment**. All code lives in **scripts and modules**, not notebooks. This document provides exact commands for reproducible training.

---

## Prerequisites

- **GitHub**: Code is version-controlled (this repo)
- **Colab**: Free tier (T4 GPU, 16GB VRAM)
- **Runtime**: Enable GPU in Colab: `Runtime` → `Change runtime type` → `T4 GPU`

---

## Checkpoint Persistence (RECOMMENDED - Do This First)

By default, Colab runtime resets lose all data. **Mount Google Drive** to persist checkpoints across sessions.

### Step 1: Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Authorize** when prompted. You'll see: `/content/drive/MyDrive/`

### Step 2: Create Checkpoint Directory

```bash
!mkdir -p /content/drive/MyDrive/failure-aware-vit-medical/checkpoints
```

### Step 3: Use Drive for Checkpoints

When training, pass `--checkpoint-dir` pointing to Drive:

```bash
--checkpoint-dir /content/drive/MyDrive/failure-aware-vit-medical/checkpoints
```

**Benefits**:
- ✅ Checkpoints survive runtime crashes
- ✅ Resume training from saved state
- ✅ No need to download models manually
- ✅ Accessible from any Colab session

> [!IMPORTANT]
> **Always mount Drive before training**. If Colab crashes mid-training, your checkpoint is safe in Drive.

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

> [!TIP]
> **Mount Drive first** (see [Checkpoint Persistence](#checkpoint-persistence-recommended---do-this-first)) to save checkpoints permanently.

### 1. Train CNN Baseline

**Without Drive** (checkpoints lost on reset):
```bash
!bash scripts/train_cnn.sh
```

**With Drive** (recommended):
```bash
!python scripts/train.py \
  --model cnn \
  --epochs 20 \
  --batch-size 64 \
  --lr 3e-4 \
  --weight-decay 0.01 \
  --patience 5 \
  --device cuda \
  --checkpoint-dir /content/drive/MyDrive/failure-aware-vit-medical/checkpoints
```

**Expected output**:
- Training time: ~20-30 minutes
- Checkpoint: `checkpoints/cnn_best.pt` (or Drive path if specified)
- Validation accuracy: >90%

### 2. Train ViT Baseline

**With Drive** (recommended):
```bash
!python scripts/train.py \
  --model vit \
  --epochs 20 \
  --batch-size 64 \
  --lr 3e-4 \
  --weight-decay 0.01 \
  --patience 5 \
  --device cuda \
  --checkpoint-dir /content/drive/MyDrive/failure-aware-vit-medical/checkpoints
```

**Expected output**:
- Training time: ~30-45 minutes (slower than CNN)
- Checkpoint: `vit_best.pt` in Drive
- Validation accuracy: comparable to CNN

**If you get CUDA OOM error**:
Reduce batch size to 32:
```bash
--batch-size 32
```

### 3. Loading Saved Checkpoints (Resume or Inference)

```python
import torch
from src.models.cnn_baseline import CNNBaseline
from src.models.vit_baseline import ViTBaseline

# Load CNN
model = CNNBaseline(num_classes=2, dropout=0.3)
checkpoint = torch.load(
    "/content/drive/MyDrive/failure-aware-vit-medical/checkpoints/cnn_best.pt",
    map_location="cuda"
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load ViT
model = ViTBaseline(num_classes=2, embed_dim=384, depth=6, num_heads=6)
checkpoint = torch.load(
    "/content/drive/MyDrive/failure-aware-vit-medical/checkpoints/vit_best.pt",
    map_location="cuda"
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

**No retraining needed** - model ready for inference or evaluation.

---

## Downloading Checkpoints

### If Using Drive (Recommended)

**No download needed!** Checkpoints are already in:
```
/content/drive/MyDrive/failure-aware-vit-medical/checkpoints/
```

Access them from:
- Any Colab session (remount Drive)
- Google Drive web interface
- Local machine (install Google Drive Desktop)

### If Using Local Colab Storage

Only if you didn't use Drive, package and download:

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
