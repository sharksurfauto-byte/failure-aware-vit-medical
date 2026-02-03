#!/usr/bin/env python3
"""
Monte Carlo Dropout Evaluation for ViT

Implements MC Dropout inference to estimate predictive uncertainty.
Computes entropy-based failure detection metrics.

Usage:
    python scripts/mc_dropout_eval.py \
        --checkpoint /path/to/vit_best.pt \
        --num-samples 20 \
        --output-dir mc_dropout_results
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import MalariaDataset
from src.models.vit_baseline import ViTBaseline


def enable_mc_dropout(model):
    """
    Enable dropout layers during inference while keeping other layers in eval mode.
    
    Args:
        model: PyTorch model with Dropout layers
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def predictive_entropy(probs):
    """
    Compute predictive entropy: H(p) = -sum(p * log(p))
    
    Args:
        probs: Tensor of shape [N, C] with predicted probabilities
    
    Returns:
        Tensor of shape [N] with entropy values
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    return -torch.sum(probs * torch.log(probs + eps), dim=1)


def mc_dropout_inference(model, test_loader, device, num_samples=20):
    """
    Run MC Dropout inference on test set.
    
    Args:
        model: ViT model
        test_loader: DataLoader for test set
        device: torch device
        num_samples: Number of forward passes (T)
    
    Returns:
        Dictionary with predictions, uncertainties, and targets
    """
    model.eval()
    enable_mc_dropout(model)
    
    all_probs = []
    all_targets = []
    
    print(f"\nRunning MC Dropout with T={num_samples} forward passes...")
    
    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc="MC Dropout Inference"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Collect T forward passes
            probs_T = []
            for _ in range(num_samples):
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                probs_T.append(probs.unsqueeze(0))
            
            # Stack and compute mean prediction
            probs_T = torch.cat(probs_T, dim=0)  # [T, B, C]
            probs_mean = probs_T.mean(dim=0)      # [B, C]
            
            all_probs.append(probs_mean.cpu())
            all_targets.append(labels.cpu())
    
    # Concatenate all batches
    all_probs = torch.cat(all_probs, dim=0)      # [N, C]
    all_targets = torch.cat(all_targets, dim=0)  # [N]
    
    # Compute metrics
    confidence = all_probs.max(dim=1).values
    entropy = predictive_entropy(all_probs)
    predictions = all_probs.argmax(dim=1)
    correct = (predictions == all_targets)
    
    results = {
        'confidence': confidence.numpy(),
        'entropy': entropy.numpy(),
        'predictions': predictions.numpy(),
        'targets': all_targets.numpy(),
        'correct': correct.numpy(),
        'num_samples': num_samples
    }
    
    return results


def plot_entropy_distribution(results, output_dir):
    """
    Plot entropy distribution for correct vs incorrect predictions.
    """
    correct_mask = results['correct']
    entropy = results['entropy']
    
    plt.figure(figsize=(10, 6))
    plt.hist(entropy[correct_mask], bins=30, alpha=0.6, label='Correct', color='green')
    plt.hist(entropy[~correct_mask], bins=30, alpha=0.6, label='Incorrect', color='red')
    plt.xlabel('Predictive Entropy', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Predictive Entropy: Correct vs Incorrect Predictions', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / 'entropy_correct_vs_incorrect.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"✓ Saved plot: {output_path}")


def compute_selective_prediction_metrics(results):
    """
    Compute selective prediction metrics at different uncertainty thresholds.
    
    Returns:
        List of dictionaries with coverage, accuracy, and failure detection rates
    """
    entropy = results['entropy']
    correct = results['correct']
    
    # Define rejection percentiles (top X% most uncertain)
    rejection_percentiles = [10, 15, 20, 25, 30]
    
    metrics = []
    
    for reject_pct in rejection_percentiles:
        # Compute threshold (higher entropy = more uncertain)
        threshold = np.percentile(entropy, 100 - reject_pct)
        
        # Keep samples below threshold (low uncertainty)
        keep_mask = entropy <= threshold
        
        # Compute metrics
        coverage = keep_mask.mean()
        accuracy_kept = correct[keep_mask].mean() if keep_mask.any() else 0.0
        
        # Failure detection: what % of failures were in rejected samples?
        failures = ~correct
        failures_rejected = failures[~keep_mask].sum()
        total_failures = failures.sum()
        failure_detection_rate = failures_rejected / total_failures if total_failures > 0 else 0.0
        
        metrics.append({
            'rejection_pct': reject_pct,
            'coverage': float(coverage),
            'accuracy_on_kept': float(accuracy_kept),
            'failures_flagged_pct': float(failure_detection_rate * 100)
        })
    
    return metrics


def print_results_table(results, selective_metrics):
    """
    Print formatted results table.
    """
    print("\n" + "=" * 70)
    print("MC DROPOUT UNCERTAINTY ESTIMATION RESULTS")
    print("=" * 70)
    
    # Overall metrics
    accuracy = results['correct'].mean()
    num_samples = len(results['correct'])
    num_correct = results['correct'].sum()
    num_failures = (~results['correct']).sum()
    
    print(f"\nOverall Performance:")
    print(f"  Test Accuracy: {accuracy * 100:.2f}%")
    print(f"  Correct: {num_correct} / {num_samples}")
    print(f"  Failures: {num_failures}")
    
    # Entropy statistics
    entropy_correct = results['entropy'][results['correct']]
    entropy_incorrect = results['entropy'][~results['correct']]
    
    print(f"\nEntropy Statistics:")
    print(f"  Correct predictions:")
    print(f"    Mean: {entropy_correct.mean():.4f}")
    print(f"    Std:  {entropy_correct.std():.4f}")
    print(f"  Incorrect predictions:")
    print(f"    Mean: {entropy_incorrect.mean():.4f}")
    print(f"    Std:  {entropy_incorrect.std():.4f}")
    print(f"  Separation: {entropy_incorrect.mean() - entropy_correct.mean():.4f}")
    
    # Selective prediction table
    print("\n" + "=" * 70)
    print("SELECTIVE PREDICTION (Uncertainty-Based Rejection)")
    print("=" * 70)
    print(f"\n{'Reject %':<12} {'Coverage':<12} {'Accuracy':<15} {'Failures Flagged':<20}")
    print("-" * 70)
    
    for m in selective_metrics:
        print(f"{m['rejection_pct']:>7}%     "
              f"{m['coverage']*100:>6.1f}%      "
              f"{m['accuracy_on_kept']*100:>9.2f}%       "
              f"{m['failures_flagged_pct']:>9.1f}%")
    
    print("\n" + "=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description='MC Dropout evaluation for ViT')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to ViT checkpoint')
    parser.add_argument('--num-samples', type=int, default=20,
                        help='Number of MC Dropout samples (T) (default: 20)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for evaluation (default: 64)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cuda, cpu, or auto (default: auto)')
    parser.add_argument('--output-dir', type=str, default='mc_dropout_results',
                        help='Output directory for results (default: mc_dropout_results)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load normalization stats
    print("\nLoading normalization stats...")
    norm_stats_path = PROJECT_ROOT / 'data' / 'normalization_stats.json'
    with open(norm_stats_path, 'r') as f:
        norm_stats = json.load(f)
    
    MalariaDataset.MEAN = norm_stats['mean']
    MalariaDataset.STD = norm_stats['std']
    
    # Load test dataset
    print("Loading test dataset...")
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
    
    # Load ViT model
    print("\nLoading ViT model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = ViTBaseline(
        num_classes=2,
        embed_dim=384,
        depth=6,
        num_heads=6,
        patch_size=16
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Val accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Run MC Dropout inference
    results = mc_dropout_inference(model, test_loader, device, args.num_samples)
    
    # Compute selective prediction metrics
    selective_metrics = compute_selective_prediction_metrics(results)
    
    # Print results
    print_results_table(results, selective_metrics)
    
    # Save raw results
    save_results = {
        'confidence': results['confidence'].tolist(),
        'entropy': results['entropy'].tolist(),
        'correct': results['correct'].tolist(),
        'num_samples': results['num_samples'],
        'selective_prediction': selective_metrics
    }
    
    results_path = output_dir / 'mc_dropout_results.json'
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")
    
    # Generate plots
    plot_entropy_distribution(results, output_dir)
    
    print(f"\n✅ MC Dropout evaluation complete!")
    print(f"   Output directory: {output_dir}")


if __name__ == '__main__':
    main()
