# Failure-Aware Vision Transformer for Malaria Diagnosis

Research project demonstrating uncertainty-aware medical image classification using Vision Transformers.

## Project Status: Stage 2 - Baseline Training

**Current Goal**: Validate CNN and ViT baselines on NIH Malaria dataset

## Quick Start (Colab)

```bash
!git clone https://github.com/sharksurfauto-byte/failure-aware-vit-medical.git
%cd failure-aware-vit-medical
!bash scripts/colab_setup.sh
!bash scripts/smoke_test.sh
```

See [COLAB_WORKFLOW.md](COLAB_WORKFLOW.md) for detailed instructions.

## Repository Structure

```
.
├── data/
│   ├── DATASET.md              # Dataset analysis and preprocessing decisions
│   ├── splits.json             # Train/val/test split assignments (reproducible)
│   ├── normalization_stats.json # Dataset normalization parameters
│   └── raw/                    # Downloaded dataset (not in Git)
├── src/
│   ├── dataset.py              # Malaria dataset + preprocessing pipeline
│   └── models/
│       ├── cnn_baseline.py     # CNN baseline (~1M params)
│       └── vit_baseline.py     # ViT baseline (~5M params)
├── scripts/
│   ├── download_dataset.sh     # Download NIH Malaria dataset
│   ├── colab_setup.sh          # One-command Colab environment setup
│   ├── create_splits.py        # Generate stratified splits
│   ├── train.py                # Training script for both models
│   ├── smoke_test.sh           # Fast pipeline validation
│   ├── train_cnn.sh            # CNN baseline training
│   ├── train_vit.sh            # ViT baseline training
│   └── download_checkpoints.sh # Package checkpoints for download
├── checkpoints/                # Saved model weights (not in Git)
├── requirements.txt            # Python dependencies
├── COLAB_WORKFLOW.md           # Step-by-step Colab execution guide
└── README.md                   # This file
```

## Dataset

**Source**: NIH Malaria Cell Images  
**Classes**: Parasitized (infected), Uninfected  
**Total**: 27,558 images (perfectly balanced)

**Preprocessing** (locked):
- Resize to 224×224 (aspect-ratio preserving + black padding)
- Normalize with dataset-specific mean/std
- Patch size: 16×16 (for ViT)

See [data/DATASET.md](data/DATASET.md) for full analysis.

## Models

### CNN Baseline
- 4 convolutional blocks
- ~1.1M parameters
- Strong feature extraction baseline

### ViT Baseline
- Patch-based transformer (16×16)
- 6 layers, 6 heads, 384 embedding dim
- ~5M parameters
- Enables attention-based explainability

## Training

**Hardware**: Google Colab (T4 GPU, 16GB VRAM)

**Hyperparameters** (identical for both):
- Optimizer: AdamW (lr=3e-4, weight_decay=0.01)
- Batch size: 64
- Max epochs: 20 (early stopping @ 5 patience)
- LR scheduler: CosineAnnealingLR

**Commands**:
```bash
bash scripts/train_cnn.sh  # CNN baseline
bash scripts/train_vit.sh  # ViT baseline
```

## Development Workflow

**Local**:
- Write code in `src/` and `scripts/`
- Commit to Git, push to GitHub

**Colab**:
- Clone repo, run `scripts/colab_setup.sh`
- Execute training via shell scripts
- Download checkpoints

**No notebook-based ML** - all logic is in Python scripts.

## Reproducibility

✅ Deterministic splits (seed=42, committed to Git)  
✅ Fixed preprocessing pipeline  
✅ Locked hyperparameters in training scripts  
✅ Version-controlled normalization stats  

Anyone can reproduce results by running:
```bash
bash scripts/colab_setup.sh
bash scripts/train_cnn.sh
bash scripts/train_vit.sh
```

## Project Roadmap

- [x] **Stage 1**: Dataset analysis and preprocessing
- [/] **Stage 2**: CNN and ViT baseline training (in progress)
- [ ] **Stage 3**: Uncertainty estimation (Bayesian ViT)
- [ ] **Stage 4**: Calibration and failure-awareness
- [ ] **Stage 5**: Attention-based explainability
- [ ] **Stage 6**: Stress testing on out-of-distribution data
- [ ] **Stage 7**: Evaluation and comparison
- [ ] **Stage 8**: Deployment-ready API

## License

Research project. Code for educational and research purposes.

## Citation

If you use this work, please cite:
```
@misc{failure-aware-vit-medical,
  author = {Sharma, [Your Name]},
  title = {Failure-Aware Vision Transformer for Medical Diagnosis},
  year = {2026},
  url = {https://github.com/sharksurfauto-byte/failure-aware-vit-medical}
}
```

## Contact

For questions or collaboration, open an issue on GitHub.
