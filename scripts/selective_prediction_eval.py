#!/usr/bin/env python3
"""
Selective Prediction Evaluation

Analyzes MC Dropout results to formalize decision rules:
- Auto-predict when entropy is low (safe)
- Flag for review when entropy is high (risky)

Computes coverage-accuracy trade-offs at different uncertainty thresholds.

Usage:
    python scripts/selective_prediction_eval.py \
        --results mc_dropout_results/mc_dropout_results.json \
        --output-dir selective_prediction_results
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def compute_selective_metrics(entropy, correct, rejection_percentiles):
    """
    Compute selective prediction metrics at different rejection thresholds.
    
    Args:
        entropy: Array of predictive entropy values
        correct: Boolean array indicating correct predictions
        rejection_percentiles: List of percentiles to reject (e.g., [10, 15, 20])
    
    Returns:
        List of metric dictionaries
    """
    metrics = []
    
    # Baseline (no rejection)
    baseline = {
        'rejection_pct': 0,
        'threshold': 0.0,
        'coverage': 1.0,
        'accuracy_auto': float(correct.mean()),
        'failures_flagged_pct': 0.0,
        'num_auto': len(correct),
        'num_flagged': 0
    }
    metrics.append(baseline)
    
    # Compute for each rejection threshold
    for reject_pct in rejection_percentiles:
        # Compute entropy threshold (high entropy = uncertain)
        threshold = np.percentile(entropy, 100 - reject_pct)
        
        # Auto-predict: samples with entropy BELOW threshold
        auto_mask = entropy <= threshold
        
        # Metrics
        coverage = auto_mask.mean()
        accuracy_auto = correct[auto_mask].mean() if auto_mask.any() else 0.0
        
        # Failure flagging rate
        failures = ~correct
        failures_flagged = failures[~auto_mask].sum()
        total_failures = failures.sum()
        failure_flag_rate = failures_flagged / total_failures if total_failures > 0 else 0.0
        
        metrics.append({
            'rejection_pct': reject_pct,
            'threshold': float(threshold),
            'coverage': float(coverage),
            'accuracy_auto': float(accuracy_auto),
            'failures_flagged_pct': float(failure_flag_rate * 100),
            'num_auto': int(auto_mask.sum()),
            'num_flagged': int((~auto_mask).sum())
        })
    
    return metrics


def plot_accuracy_coverage_curve(metrics, output_path):
    """
    Plot accuracy vs coverage curve for selective prediction.
    """
    coverages = [m['coverage'] * 100 for m in metrics]
    accuracies = [m['accuracy_auto'] * 100 for m in metrics]
    
    plt.figure(figsize=(10, 6))
    plt.plot(coverages, accuracies, 'o-', linewidth=2, markersize=8, label='MC Dropout Selective Prediction')
    
    # Highlight key points
    baseline = metrics[0]
    plt.scatter([baseline['coverage'] * 100], [baseline['accuracy_auto'] * 100], 
                color='red', s=150, zorder=5, label='Baseline (No Rejection)', marker='*')
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Labels
    plt.xlabel('Coverage (% of samples auto-predicted)', fontsize=12)
    plt.ylabel('Accuracy on Auto-predicted Samples (%)', fontsize=12)
    plt.title('Selective Prediction: Accuracy vs Coverage Trade-off', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    
    # Set axis limits for clarity
    plt.xlim(75, 105)
    y_min = min(accuracies) - 1
    y_max = max(accuracies) + 1
    plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"✓ Saved plot: {output_path}")


def print_decision_table(metrics):
    """
    Print formatted decision table.
    """
    print("\n" + "=" * 90)
    print("SELECTIVE PREDICTION: DECISION LOGIC ANALYSIS")
    print("=" * 90)
    
    print("\nDecision Rule:")
    print("  • If entropy ≤ threshold → Auto-predict (safe)")
    print("  • If entropy > threshold → Flag for human review (risky)")
    
    print("\n" + "-" * 90)
    print(f"{'Rejection':<18} {'Threshold':<12} {'Coverage':<12} {'Accuracy':<15} {'Failures':<20}")
    print(f"{'Strategy':<18} {'(entropy)':<12} {'(auto)':<12} {'(auto-pred)':<15} {'Flagged':<20}")
    print("-" * 90)
    
    for m in metrics:
        if m['rejection_pct'] == 0:
            strategy = "None (baseline)"
            threshold_str = "N/A"
        else:
            strategy = f"Top {m['rejection_pct']}% uncertain"
            threshold_str = f"{m['threshold']:.4f}"
        
        print(f"{strategy:<18} {threshold_str:<12} "
              f"{m['coverage']*100:>6.1f}%      "
              f"{m['accuracy_auto']*100:>9.2f}%       "
              f"{m['failures_flagged_pct']:>9.1f}%")
    
    print("=" * 90)
    
    # Key insights
    print("\nKey Insights:")
    
    # Find best trade-off (e.g., 15% rejection)
    if len(metrics) > 2:
        best_idx = 2  # Usually 15% rejection
        best = metrics[best_idx]
        
        accuracy_gain = (best['accuracy_auto'] - metrics[0]['accuracy_auto']) * 100
        coverage_loss = (metrics[0]['coverage'] - best['coverage']) * 100
        
        print(f"  • At {best['rejection_pct']}% rejection:")
        print(f"    - Accuracy improves by {accuracy_gain:.2f} percentage points")
        print(f"    - Only {coverage_loss:.1f}% of predictions need review")
        print(f"    - Captures {best['failures_flagged_pct']:.1f}% of all failures")
        
        # Medical deployment context
        print(f"\n  • Medical Deployment Context:")
        print(f"    - {best['num_auto']} samples: Auto-diagnose with {best['accuracy_auto']*100:.2f}% accuracy")
        print(f"    - {best['num_flagged']} samples: Send to expert clinician")
        print(f"    - Result: {best['failures_flagged_pct']:.1f}% of errors prevented from reaching patients")
    
    print("\n" + "=" * 90)


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze selective prediction performance')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to mc_dropout_results.json')
    parser.add_argument('--output-dir', type=str, default='selective_prediction_results',
                        help='Output directory for results (default: selective_prediction_results)')
    parser.add_argument('--rejection-levels', type=int, nargs='+', default=[10, 15, 20, 25, 30],
                        help='Rejection percentiles to evaluate (default: 10 15 20 25 30)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading MC Dropout results from: {args.results}")
    
    # Load MC Dropout results
    with open(args.results, 'r') as f:
        mc_results = json.load(f)
    
    entropy = np.array(mc_results['entropy'])
    correct = np.array(mc_results['correct'])
    
    print(f"Loaded {len(entropy)} samples")
    print(f"Overall accuracy: {correct.mean() * 100:.2f}%")
    
    # Compute selective prediction metrics
    print(f"\nComputing metrics at rejection levels: {args.rejection_levels}")
    metrics = compute_selective_metrics(entropy, correct, args.rejection_levels)
    
    # Print decision table
    print_decision_table(metrics)
    
    # Save metrics
    output_path = output_dir / 'selective_prediction_table.json'
    with open(output_path, 'w') as f:
        json.dump({
            'metrics': metrics,
            'rejection_levels': args.rejection_levels,
            'total_samples': len(entropy),
            'baseline_accuracy': float(correct.mean())
        }, f, indent=2)
    
    print(f"\n✓ Saved metrics to: {output_path}")
    
    # Plot accuracy-coverage curve
    plot_path = output_dir / 'accuracy_coverage_curve.png'
    plot_accuracy_coverage_curve(metrics, plot_path)
    
    print(f"\n✅ Selective prediction analysis complete!")
    print(f"   Output directory: {output_dir}")
    
    # Final recommendation
    print("\n" + "=" * 90)
    print("DEPLOYMENT RECOMMENDATION")
    print("=" * 90)
    
    # Find optimal threshold (balancing coverage and safety)
    optimal = metrics[2] if len(metrics) > 2 else metrics[1]  # Default to 15% rejection
    
    print(f"\nRecommended Configuration:")
    print(f"  • Entropy threshold: {optimal['threshold']:.4f}")
    print(f"  • Auto-prediction coverage: {optimal['coverage']*100:.1f}%")
    print(f"  • Expected accuracy (auto): {optimal['accuracy_auto']*100:.2f}%")
    print(f"  • Failure prevention rate: {optimal['failures_flagged_pct']:.1f}%")
    print(f"\nInterpretation:")
    print(f"  The system will automatically diagnose ~{optimal['num_auto']} cases with high confidence,")
    print(f"  while flagging ~{optimal['num_flagged']} uncertain cases for expert review.")
    print(f"  This prevents {optimal['failures_flagged_pct']:.1f}% of prediction errors from reaching patients.")
    print("=" * 90 + "\n")


if __name__ == '__main__':
    main()
