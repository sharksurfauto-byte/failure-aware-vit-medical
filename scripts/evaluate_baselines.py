#!/usr/bin/env python3
"""
Baseline Model Evaluation Script

Evaluates CNN and ViT baselines on test_clean split.
Computes: Accuracy, Precision, Recall, F1, ECE, confidence distributions.

Usage:
    python scripts/evaluate_baselines.py \
        --cnn-checkpoint checkpoints/cnn_best.pt \
        --vit-checkpoint checkpoints/vit_best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import MalariaDataset
from src.models.cnn_baseline import CNNBaseline
from src.models.vit_baseline import ViTBaseline


def compute_ece(confidences, predictions, labels, n_bins=15):
    """
    Compute Expected Calibration Error (ECE)
    
    Args:
        confidences: Array of confidence scores (max softmax probabilities)
        predictions: Array of predicted class labels
        labels: Array of true class labels
        n_bins: Number of bins for calibration
    
    Returns:
        ECE score (lower is better, 0 = perfect calibration)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this confidence bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Accuracy of predictions in this bin
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
            # Average confidence in this bin
            avg_confidence_in_bin = confidences[in_bin].mean()
            # ECE contribution from this bin
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def evaluate_model(model, test_loader, device, model_name):
    """
    Evaluate a model on test set
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_confidences = []
    
    print(f"\nEvaluating {model_name}...")
    
    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc=f"{model_name} evaluation"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            
            # Get predictions and confidences
            confidences, predictions = torch.max(probs, dim=1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    # Compute ECE
    ece = compute_ece(all_confidences, all_preds, all_labels)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Confidence statistics
    conf_mean = all_confidences.mean()
    conf_std = all_confidences.std()
    
    # Confidence by correctness
    correct_mask = all_preds == all_labels
    conf_correct = all_confidences[correct_mask].mean() if correct_mask.any() else 0.0
    conf_incorrect = all_confidences[~correct_mask].mean() if (~correct_mask).any() else 0.0
    
    results = {
        'model': model_name,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'ece': float(ece),
        'confusion_matrix': cm.tolist(),
        'confidence': {
            'mean': float(conf_mean),
            'std': float(conf_std),
            'correct': float(conf_correct),
            'incorrect': float(conf_incorrect)
        }
    }
    
    return results


def load_model(checkpoint_path, model_type, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if model_type == 'cnn':
        model = CNNBaseline(num_classes=2, dropout=0.3)
    elif model_type == 'vit':
        model = ViTBaseline(
            num_classes=2,
            embed_dim=384,
            depth=6,
            num_heads=6,
            patch_size=16
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded {model_type.upper()} from epoch {checkpoint['epoch']}")
    print(f"  Val accuracy: {checkpoint['val_acc']:.2f}%")
    
    return model


def print_comparison_table(results_cnn, results_vit):
    """Print formatted comparison table"""
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON (Test Clean Split)")
    print("=" * 70)
    
    print(f"\n{'Metric':<20} {'CNN':<15} {'ViT':<15} {'Winner':<10}")
    print("-" * 70)
    
    metrics = [
        ('Accuracy', 'accuracy', '%', True),
        ('Precision', 'precision', '%', True),
        ('Recall', 'recall', '%', True),
        ('F1 Score', 'f1', '%', True),
        ('ECE', 'ece', '', False),  # Lower is better
        ('Avg Confidence', 'confidence.mean', '%', False),
    ]
    
    for metric_name, key, suffix, higher_better in metrics:
        # Handle nested keys (e.g., 'confidence.mean')
        cnn_val = results_cnn
        vit_val = results_vit
        for k in key.split('.'):
            cnn_val = cnn_val[k]
            vit_val = vit_val[k]
        
        # Format values
        if suffix == '%':
            cnn_str = f"{cnn_val*100:.2f}%"
            vit_str = f"{vit_val*100:.2f}%"
        else:
            cnn_str = f"{cnn_val:.4f}"
            vit_str = f"{vit_val:.4f}"
        
        # Determine winner
        if higher_better:
            winner = "CNN" if cnn_val > vit_val else "ViT"
        else:
            winner = "CNN" if cnn_val < vit_val else "ViT"
        
        print(f"{metric_name:<20} {cnn_str:<15} {vit_str:<15} {winner:<10}")
    
    print("\n" + "=" * 70)
    print("CONFIDENCE ANALYSIS")
    print("=" * 70)
    
    for model_name, results in [('CNN', results_cnn), ('ViT', results_vit)]:
        print(f"\n{model_name}:")
        print(f"  Confidence (correct predictions): {results['confidence']['correct']*100:.2f}%")
        print(f"  Confidence (incorrect predictions): {results['confidence']['incorrect']*100:.2f}%")
        conf_gap = (results['confidence']['correct'] - results['confidence']['incorrect']) * 100
        print(f"  Confidence gap: {conf_gap:.2f}pp")
    
    print("\n" + "=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate baseline models')
    parser.add_argument('--cnn-checkpoint', type=str, required=True,
                        help='Path to CNN checkpoint')
    parser.add_argument('--vit-checkpoint', type=str, required=True,
                        help='Path to ViT checkpoint')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for evaluation (default: 64)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cuda, cpu, or auto (default: auto)')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='Output JSON file for results (default: evaluation_results.json)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = MalariaDataset(
        split='test_clean',
        splits_path=str(PROJECT_ROOT / 'data' / 'splits.json')
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    print(f"Test set size: {len(test_dataset)}")
    
    # Load models
    print("\n" + "=" * 70)
    print("LOADING MODELS")
    print("=" * 70)
    
    cnn_model = load_model(args.cnn_checkpoint, 'cnn', device)
    vit_model = load_model(args.vit_checkpoint, 'vit', device)
    
    # Evaluate models
    print("\n" + "=" * 70)
    print("RUNNING EVALUATION")
    print("=" * 70)
    
    results_cnn = evaluate_model(cnn_model, test_loader, device, 'CNN')
    results_vit = evaluate_model(vit_model, test_loader, device, 'ViT')
    
    # Print comparison
    print_comparison_table(results_cnn, results_vit)
    
    # Save results
    output_path = PROJECT_ROOT / args.output
    results = {
        'cnn': results_cnn,
        'vit': results_vit,
        'test_set_size': len(test_dataset)
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")


if __name__ == '__main__':
    main()
