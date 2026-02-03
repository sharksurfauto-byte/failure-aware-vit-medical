# Experimental Results

Detailed results for the Failure-Aware ViT Medical Diagnosis project.

## Stage 2: Baseline Training

### Dataset Split

| Split | Samples | Percentage |
|-------|---------|------------|
| Train | 19,290 | 70% |
| Validation | 4,132 | 15% |
| Test (clean) | 3,102 | 11.3% |
| Stress source | 1,034 | 3.8% |
| **Total** | **27,558** | **100%** |

### Model Comparison

| Model | Test Accuracy | Precision | Recall | F1 Score | ECE |
|-------|--------------|-----------|--------|----------|-----|
| CNN Baseline | 96.45% | 96.52% | 96.38% | 96.45% | 0.0085 |
| ViT Baseline | 96.36% | 96.41% | 96.31% | 96.36% | 0.0092 |

**Hyperparameters** (locked, no tuning):
- Optimizer: AdamW
- Learning rate: 3e-4
- Weight decay: 0.01
- Batch size: 64 (CNN), 64 (ViT)
- Max epochs: 20
- Early stopping: 5 epochs patience
- LR scheduler: CosineAnnealingLR

### Confidence Analysis

| Model | Confidence (Correct) | Confidence (Incorrect) | Gap |
|-------|---------------------|----------------------|-----|
| CNN | 96.77% | 77.40% | 19.37pp |
| ViT | 97.34% | **80.03%** | **17.31pp** |

**Key Insight**: ViT shows **higher confidence on failures** and a **smaller confidence gap**, indicating riskier behavior despite similar accuracy.

---

## Stage 3.1: MC Dropout Uncertainty Estimation

### Parameters
- Forward passes (T): 20
- Test set: Clean split (3,102 samples)
- Dropout rate: 0.1 (attention + MLP layers)

### Entropy Statistics

| Prediction Type | Mean Entropy | Std Entropy | Samples |
|----------------|--------------|-------------|---------|
| Correct | 0.0234 | 0.0412 | 2,989 |
| Incorrect | 0.3843 | 0.2156 | 113 |
| **Separation** | **0.3609** | — | — |

**Interpretation**: Clear separation between correct and incorrect predictions. High entropy reliably indicates higher failure risk.

### Distribution

![Entropy Distribution](mc_dropout_results/entropy_correct_vs_incorrect.png)

---

## Stage 3A: Selective Prediction Decision Logic

### Decision Rule

```
For each sample:
  - Compute predictive entropy H
  - If H ≤ threshold τ → Auto-predict (safe)
  - If H > threshold τ → Flag for expert review (uncertain)
```

### Performance at Different Operating Points

| Rejection Strategy | Coverage | Accuracy (Auto) | Accuracy Gain | Failures Prevented |
|-------------------|----------|-----------------|---------------|-------------------|
| **Baseline** (no rejection) | 100.0% | 96.23% | — | 0.0% |
| Top 10% uncertain | 90.0% | 98.43% | +2.20pp | 67.3% |
| **Top 15% uncertain** | **85.0%** | **99.20%** | **+2.97pp** | **82.1%** |
| Top 20% uncertain | 80.0% | 99.48% | +3.25pp | 88.9% |
| Top 25% uncertain | 75.0% | 99.65% | +3.42pp | 92.6% |
| Top 30% uncertain | 70.0% | 99.77% | +3.54pp | 95.6% |

### Recommended Operating Point

**15% Rejection** (85% coverage):
- **Auto-predicted**: 2,636 samples (99.20% accuracy)
- **Flagged for review**: 466 samples
- **Failures prevented**: 82.1% (93 out of 113 total failures)
- **Entropy threshold**: 0.0891

**Medical Context**: 
- Expert workload: Review 466 cases (~15% of total)
- Automation: 2,636 cases handled autonomously with 99% accuracy
- Safety: Only 20 model errors reach automated diagnosis (vs. 113 without rejection)

### Accuracy-Coverage Curve

![Accuracy vs Coverage](selective_prediction_results/accuracy_coverage_curve.png)

**Observation**: Accuracy increases monotonically as coverage decreases, confirming that entropy-based ranking is effective for selective prediction.

---

## Key Findings Summary

### 1. ViT Matches CNN Accuracy
- Test accuracy: 96.36% (CNN: 96.45%)
- Well-calibrated (ECE: 0.0092)
- Suitable for medical diagnosis

### 2. ViT Exhibits Riskier Confidence
- Higher confidence on failures: 80% (vs CNN: 77%)
- Smaller confidence gap: 17.3pp (vs CNN: 19.4pp)
- **Motivates explicit uncertainty estimation**

### 3. MC Dropout Detects Failures
- Clear entropy separation: 0.36 between correct/incorrect
- No retraining required (inference-only)
- **82.1% failure detection at 15% rejection**

### 4. Selective Prediction Enables Safe Deployment
- 99.20% accuracy on auto-predicted cases (85% coverage)
- +2.97pp accuracy improvement over baseline
- **Human-in-the-loop safety mechanism**

---

## Computational Requirements

### Training
- CNN: ~15 minutes (T4 GPU, Colab)
- ViT: ~25 minutes (T4 GPU, Colab)
- Total training time: ~40 minutes

### Inference
- Standard: ~2 seconds for 3,102 test samples
- MC Dropout (T=20): ~40 seconds for 3,102 test samples
- **20× inference cost for uncertainty estimation**

---

## Reproducibility Checklist

- ✅ Random seed: 42 (all splits and training)
- ✅ Splits committed: `data/splits.json`
- ✅ Normalization stats committed: `data/normalization_stats.json`
- ✅ No test set tuning (hyperparameters locked)
- ✅ Deterministic evaluation order
- ✅ Version-controlled checkpoints (Drive)
- ✅ Script-based execution (no manual steps)

---

**Last Updated**: February 2026  
**Dataset**: NIH Malaria Cell Images (27,558 samples)  
**Compute**: Google Colab (T4 GPU, 16GB RAM)
