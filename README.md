# Failure-Aware Vision Transformer for Medical Diagnosis

A human-in-the-loop medical image classification system that uses uncertainty estimation to prevent prediction failures from reaching patients.

## Problem Statement

Medical AI systems require more than high accuracy—they need to know when they're uncertain. **Overconfident failures** (incorrect predictions made with high confidence) pose significant safety risks in clinical settings. While Vision Transformers (ViTs) achieve strong performance on medical imaging tasks, they can exhibit dangerous confidence behavior on incorrect predictions.

This project addresses this gap by building a **failure-aware** system that:
- Detects when the model is likely to be wrong
- Automatically defers high-risk cases to human experts
- Quantifiably prevents the majority of failures from reaching patients

## System Overview

```
Medical Image (224×224)
        ↓
   Vision Transformer
        ↓
   MC Dropout (T=20)
        ↓
  Predictive Entropy
        ↓
    ┌───────────────┐
    │  Entropy ≤ τ  │  →  Auto-Predict (High Confidence)
    │  Entropy > τ  │  →  Flag for Expert Review (Uncertain)
    └───────────────┘
```

**Key Innovation**: By rejecting only 15% of the most uncertain predictions, the system achieves **99.20% accuracy** on automated cases while preventing **82.1% of all model errors**.

## Dataset & Experimental Setup

**Dataset**: NIH Malaria Cell Images Dataset
- 27,558 cell images (parasitized vs. uninfected)
- Deterministic stratified splits: 70% train / 15% val / 15% test
- Clean test set (11.3%) + stress test source (3.8%)

**Preprocessing**:
- Resize + pad to 224×224
- Dataset-specific normalization (pre-computed stats)
- Reproducible splits committed to version control

**Infrastructure**:
- Script-based training (no notebooks)
- Google Colab execution with T4 GPU
- Checkpoint persistence via Google Drive
- Fully reproducible pipeline

## Baseline Results (Stage 2)

Controlled comparison between CNN and ViT baselines:

| Model | Test Accuracy | ECE (Calibration) | Confidence on Failures |
|-------|--------------|-------------------|----------------------|
| CNN Baseline | 96.45% | 0.0085 | 77.40% |
| **ViT Baseline** | 96.36% | 0.0092 | **80.03%** ↑ |

**Key Finding**: ViT matches CNN accuracy but shows **higher confidence on incorrect predictions** (80% vs 77%), with a smaller confidence gap between correct and incorrect cases (17.3pp vs 19.4pp). This motivates explicit failure-awareness mechanisms.

## Failure-Aware Results (Stage 3)

### MC Dropout Uncertainty Estimation

Using Monte Carlo Dropout (T=20 forward passes), we measure predictive entropy to quantify model uncertainty.

**Entropy Separation**: 
- Correct predictions: Mean entropy = 0.0234
- Incorrect predictions: Mean entropy = 0.3843
- **Separation: 0.3609** (clear discriminative signal)

### Selective Prediction Performance

By deferring high-uncertainty cases to human experts, the system achieves substantial safety improvements:

| Rejection Strategy | Coverage | Accuracy (Auto) | Failures Prevented |
|-------------------|----------|-----------------|-------------------|
| None (baseline) | 100% | 96.23% | 0% |
| **Top 15% uncertain** | **85%** | **99.20%** | **82.1%** |
| Top 20% uncertain | 80% | 99.48% | 88.9% |
| Top 25% uncertain | 75% | 99.65% | 92.6% |

**Interpretation**: At the recommended operating point (15% rejection):
- **2,636 cases** auto-diagnosed with 99.20% accuracy
- **466 cases** deferred to clinicians for review
- **82.1% of all model failures** never reach patients

## Why This Matters

### Medical Deployment Context

Traditional ML: *"My model is 96% accurate."*  
**This system**: *"My model auto-diagnoses 85% of cases with 99% accuracy, and prevents 82% of errors by requesting expert review on uncertain cases."*

**Benefits**:
1. **Safety**: Most failures caught before clinical impact
2. **Efficiency**: Majority of cases automated with high confidence
3. **Trust**: System acknowledges limitations explicitly
4. **Scalability**: Reduces expert workload while maintaining safety

### Human-in-the-Loop Design

This is not a full-automation system. It's a **decision support** system that:
- Handles routine cases autonomously
- Escalates edge cases to experts
- Operates within a clinically meaningful risk tolerance

## Reproducibility

All experiments are fully reproducible:

- ✅ **Script-based workflow** (no notebook-only code)
- ✅ **Deterministic splits** (`data/splits.json` version-controlled)
- ✅ **Locked hyperparameters** (no tuning on test set)
- ✅ **Pre-computed normalization** (`data/normalization_stats.json`)
- ✅ **Colab execution guide** (`COLAB_WORKFLOW.md`)
- ✅ **Checkpoint persistence** (Google Drive integration)

### Quick Start (Google Colab)

```bash
# Clone repository
!git clone https://github.com/sharksurfauto-byte/failure-aware-vit-medical.git
%cd failure-aware-vit-medical

# Mount Google Drive for checkpoints
from google.colab import drive
drive.mount('/content/drive')

# Setup environment and download data
!bash scripts/colab_setup.sh

# Train baselines (or load from Drive)
!bash scripts/train_vit.sh --checkpoint-dir /content/drive/MyDrive/failure-aware-vit-medical/checkpoints

# Evaluate with MC Dropout
!bash scripts/mc_dropout_colab.sh

# Generate selective prediction analysis
!bash scripts/selective_prediction_colab.sh
```

## Project Structure

```
failure-aware-vit-medical/
├── app/                         # Deployment API
│   ├── main.py                  # FastAPI service
│   └── inference.py             # MC Dropout inference + decision logic
├── data/
│   ├── splits.json              # Deterministic train/val/test splits
│   └── normalization_stats.json # Dataset-specific preprocessing
├── src/
│   ├── models/
│   │   ├── cnn_baseline.py      # CNN baseline (ResNet-like)
│   │   └── vit_baseline.py      # Vision Transformer
│   └── dataset.py               # Malaria dataset class
├── scripts/
│   ├── train.py                 # Universal training script
│   ├── evaluate_baselines.py   # Stage 2 evaluation
│   ├── mc_dropout_eval.py       # MC Dropout uncertainty estimation
│   └── selective_prediction_eval.py  # Decision logic analysis
├── COLAB_WORKFLOW.md            # Detailed execution guide
└── RESULTS.md                   # Detailed experimental results
```

## Deployment

A minimal FastAPI service demonstrates deployment-ready behavior. The API exposes uncertainty-aware decision logic, automatically predicting low-risk cases while flagging high-uncertainty inputs for human review.

### Running the API

**Local**:
```bash
# Install dependencies
pip install -r requirements.txt

# Ensure checkpoint is available
# Place vit_best.pt in checkpoints/ directory

# Run API
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Docker** (optional):
```bash
docker build -t malaria-diagnosis-api .
docker run -p 8000:8000 -v /path/to/checkpoints:/app/checkpoints malaria-diagnosis-api
```

### API Endpoints

**POST /analyze** - Analyze malaria cell image
- Input: image file (multipart/form-data)
- Output: prediction, confidence, entropy, decision (AUTO/REVIEW)

**GET /health** - Health check

**GET /info** - API information and decision logic

### Example Usage

```python
import requests

response = requests.post(
    "http://localhost:8000/analyze",
    files={"image": open("cell_image.png", "rb")}
)

result = response.json()
# {
#   "prediction": "Parasitized",
#   "confidence": 0.9918,
#   "entropy": 0.0421,
#   "decision": "AUTO",
#   "threshold": 0.2015
# }
```

**Decision Logic**:
- `entropy ≤ 0.2015` → **AUTO** (safe to auto-predict, ~85% of cases)
- `entropy > 0.2015` → **REVIEW** (flag for expert review, ~15% of cases)

Visit `http://localhost:8000/docs` for interactive Swagger UI.

## Key Technologies

- **Model**: Vision Transformer (6 layers, 384 dim, ~5M params)
- **Uncertainty**: Monte Carlo Dropout (20 forward passes)
- **Metric**: Predictive Entropy
- **Framework**: PyTorch
- **API**: FastAPI + Uvicorn
- **Compute**: Google Colab (T4 GPU)

## Results Summary

See [RESULTS.md](RESULTS.md) for detailed tables, plots, and analysis.

**Bottom Line**:
- ✅ ViT baseline: 96.36% accuracy (matches CNN)
- ✅ MC Dropout: 82.1% failure detection at 15% rejection
- ✅ Selective prediction: 99.20% accuracy on 85% coverage
- ✅ No retraining required (inference-only uncertainty)

## Future Work

Potential extensions (not required for core functionality):

1. **Stress Testing**: Evaluate decision logic on corrupted/OOD data
2. **Mutual Information**: Decompose uncertainty into epistemic/aleatoric
3. **Deep Ensembles**: Compare with ensemble-based uncertainty
4. **Clinical Validation**: Partner with medical experts for real-world evaluation

## Citation

If you use this work, please cite:

```
Failure-Aware Vision Transformer for Medical Diagnosis
GitHub: https://github.com/sharksurfauto-byte/failure-aware-vit-medical
```


---

**Author**: Built as a demonstration of failure-aware medical AI systems with quantifiable safety guarantees.

**Contact**: Available via GitHub Issues
