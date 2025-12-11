#!/usr/bin/env python3
"""
Comprehensive Analysis Script - Generates all tables and figures from the paper

This script computes:
1. Per-class AUROC and AUPRC metrics
2. Species with lowest AUPRC (Table)
3. Most confused class pairs (Table)
4. Calibration curves for all predictors
5. Method comparison bar charts
6. All outputs saved to results/
"""

import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Metrics
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

# TorchCP
from torchcp.classification.score import APS, LAC
try:
    from torchcp.classification.score import EntmaxScore
    ENTMAX_AVAILABLE = True
except ImportError:
    ENTMAX_AVAILABLE = False

from torchcp.classification.predictor import SplitPredictor, ClassConditionalPredictor
try:
    from torchcp.classification.predictor import RC3PPredictor
    RC3P_AVAILABLE = True
except ImportError:
    RC3P_AVAILABLE = False

# Import helpers
from helpers import create_dataloaders, create_resnet50_model


def compute_per_class_metrics(model, test_loader, class_names, device='cuda'):
    """Compute per-class AUROC and AUPRC."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Computing per-class metrics"):
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    n_classes = len(class_names)

    # Compute per-class metrics
    results = []
    for i in range(n_classes):
        # Binary labels for this class
        y_true = (all_labels == i).astype(int)
        y_score = all_probs[:, i]

        try:
            auroc = roc_auc_score(y_true, y_score)
            auprc = average_precision_score(y_true, y_score)
        except:
            auroc = 0.0
            auprc = 0.0

        results.append({
            'class': class_names[i],
            'auroc': auroc,
            'auprc': auprc
        })

    return pd.DataFrame(results)


def get_confusion_pairs(model, test_loader, class_names, device='cuda'):
    """Get most confused class pairs."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Computing confusion matrix"):
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Find most confused pairs
    confusion_pairs = []
    for true_idx in range(len(class_names)):
        # Count for this true class
        true_count = (all_labels == true_idx).sum()
        if true_count == 0:
            continue

        for pred_idx in range(len(class_names)):
            if true_idx == pred_idx:
                continue

            error_count = cm[true_idx, pred_idx]
            if error_count == 0:
                continue

            error_rate = error_count / true_count

            confusion_pairs.append({
                'true_class': class_names[true_idx],
                'pred_class': class_names[pred_idx],
                'error_rate': error_rate,
                'error_count': error_count
            })

    # Sort by error rate
    confusion_df = pd.DataFrame(confusion_pairs)
    confusion_df = confusion_df.sort_values('error_rate', ascending=False)

    return confusion_df


def evaluate_conformal_with_alphas(predictor, eval_loader, alphas, device='cuda'):
    """Evaluate conformal predictor at multiple alpha levels."""
    coverages = []
    set_sizes = []

    # Store results for each alpha
    all_pred_sets = []

    for alpha in alphas:
        # Re-calibrate at this alpha (assumes predictor can be re-calibrated)
        # For simplicity, we'll just evaluate at different thresholds
        alpha_coverages = []
        alpha_set_sizes = []

        with torch.no_grad():
            for images, labels in eval_loader:
                images = images.to(device)
                labels = labels.cpu().numpy()

                # Get prediction sets
                pred_sets = predictor.predict(images)
                pred_sets = pred_sets.cpu().numpy()

                # Compute coverage and set size
                for i in range(len(labels)):
                    pred_set_indices = np.where(pred_sets[i] == 1)[0]
                    alpha_set_sizes.append(len(pred_set_indices))
                    alpha_coverages.append(int(labels[i] in pred_set_indices))

        coverages.append(np.mean(alpha_coverages))
        set_sizes.append(np.mean(alpha_set_sizes))

    return coverages, set_sizes


def plot_calibration_curves(model, cal_loader, eval_loader, output_dir, device='cuda'):
    """Generate calibration curves for all predictors."""
    output_dir = Path(output_dir) / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Alpha values to test
    alphas = np.linspace(0.01, 0.20, 20)
    target_coverages = 1 - alphas

    # Score functions
    score_configs = [
        ('APS', APS()),
        ('LAC', LAC()),
    ]
    if ENTMAX_AVAILABLE:
        score_configs.append(('EntmaxScore', EntmaxScore()))

    # Predictor configs
    predictor_configs = [
        ('SplitPredictor', SplitPredictor),
        ('ClassConditionalPredictor', ClassConditionalPredictor),
    ]
    if RC3P_AVAILABLE:
        predictor_configs.append(('RC3PPredictor', RC3PPredictor))

    for pred_name, pred_cls in predictor_configs:
        print(f"\nGenerating calibration curves for {pred_name}...")

        fig, ax = plt.subplots(figsize=(8, 6))

        for score_name, score_fn in score_configs:
            coverages = []

            for alpha in tqdm(alphas, desc=f"{pred_name} + {score_name}", leave=False):
                try:
                    predictor = pred_cls(score_fn, model)
                    predictor.calibrate(cal_loader, alpha=alpha)

                    # Evaluate coverage
                    correct = 0
                    total = 0

                    with torch.no_grad():
                        for images, labels in eval_loader:
                            images = images.to(device)
                            labels = labels.cpu().numpy()

                            pred_sets = predictor.predict(images)
                            pred_sets = pred_sets.cpu().numpy()

                            for i in range(len(labels)):
                                pred_set_indices = np.where(pred_sets[i] == 1)[0]
                                if labels[i] in pred_set_indices:
                                    correct += 1
                                total += 1

                    coverage = correct / total if total > 0 else 0
                    coverages.append(coverage)

                except Exception as e:
                    coverages.append(np.nan)

            # Plot
            ax.plot(target_coverages, coverages, marker='o', label=score_name, linewidth=2)

        # Add diagonal
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Ideal')

        ax.set_xlabel('Target Coverage $(1-\\alpha)$', fontsize=13, fontweight='bold')
        ax.set_ylabel('Empirical Coverage', fontsize=13, fontweight='bold')
        ax.set_title(f'Calibration Curve: {pred_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.79, 1.01])
        ax.set_ylim([0.79, 1.01])

        plt.tight_layout()
        save_path = output_dir / f'calibration_curve_{pred_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved to {save_path}")


def plot_method_comparison(results, alpha, output_dir):
    """Create bar chart comparing methods at fixed alpha."""
    output_dir = Path(output_dir) / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([{
        'predictor': r['predictor'],
        'score': r['score'],
        'avg_set_size': r['avg_set_size']
    } for r in results])

    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by predictor
    predictors = df['predictor'].unique()
    scores = df['score'].unique()
    x = np.arange(len(predictors))
    width = 0.25

    for i, score in enumerate(scores):
        score_data = df[df['score'] == score]
        values = [score_data[score_data['predictor'] == p]['avg_set_size'].values[0]
                  if len(score_data[score_data['predictor'] == p]) > 0 else 0
                  for p in predictors]

        ax.bar(x + i*width, values, width, label=score)

    ax.set_ylabel('Average Prediction Set Size', fontsize=13, fontweight='bold')
    ax.set_xlabel('Calibration Method', fontsize=13, fontweight='bold')
    ax.set_title(f'Method Comparison at $\\alpha = {alpha}$ ({(1-alpha)*100:.0f}% coverage)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(predictors, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / f'method_comparison_alpha_{alpha}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nMethod comparison chart saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive analysis for paper')

    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth')
    parser.add_argument('--data-dir', type=str, default='./CUB_200_2011')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--results-dir', type=str, default='./results')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("\n" + "=" * 70)
    print("Loading Data")
    print("=" * 70)

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    test_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    _, test_loader = create_dataloaders(
        root_dir=args.data_dir,
        train_transform=test_transform,
        test_transform=test_transform,
        batch_size=args.batch_size,
        use_bbox=True
    )

    test_dataset = test_loader.dataset
    class_names = test_dataset.class_names

    # Split for calibration
    cal_size = int(len(test_dataset) * 0.5)
    eval_size = len(test_dataset) - cal_size

    cal_dataset, eval_dataset = random_split(
        test_dataset, [cal_size, eval_size],
        generator=torch.Generator().manual_seed(42)
    )

    cal_loader = DataLoader(cal_dataset, batch_size=args.batch_size, shuffle=False)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    print("\n" + "=" * 70)
    print("Loading Model")
    print("=" * 70)

    model = create_resnet50_model(num_classes=200, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 1. Per-class metrics
    print("\n" + "=" * 70)
    print("Computing Per-Class Metrics")
    print("=" * 70)

    per_class_df = compute_per_class_metrics(model, test_loader, class_names, device)

    # Save and display lowest AUPRC
    per_class_df_sorted = per_class_df.sort_values('auprc')
    lowest_10 = per_class_df_sorted.head(10)

    print("\n=== Species with Lowest AUPRC ===")
    print(lowest_10.to_string(index=False))

    lowest_10.to_csv(results_dir / 'lowest_auprc_species.csv', index=False)

    # Generate LaTeX table
    latex_table = lowest_10[['class', 'auroc', 'auprc']].to_latex(
        index=False,
        float_format='%.3f',
        column_format='lcc',
        caption='Species with the lowest per-class AUPRC.',
        label='tab:low-auprc-species'
    )
    with open(results_dir / 'lowest_auprc_table.tex', 'w') as f:
        f.write(latex_table)

    # 2. Confusion analysis
    print("\n" + "=" * 70)
    print("Computing Confusion Pairs")
    print("=" * 70)

    confusion_df = get_confusion_pairs(model, test_loader, class_names, device)
    top_10_confusion = confusion_df.head(10)

    print("\n=== Top 10 Most Confused Class Pairs ===")
    print(top_10_confusion.to_string(index=False))

    top_10_confusion.to_csv(results_dir / 'top_confused_pairs.csv', index=False)

    # Generate LaTeX table
    latex_confusion = top_10_confusion[['true_class', 'pred_class', 'error_rate']].copy()
    latex_confusion['pair'] = latex_confusion['true_class'] + ' $\\rightarrow$ ' + latex_confusion['pred_class']
    latex_confusion['error_rate'] = (latex_confusion['error_rate'] * 100).round(2)

    latex_table = latex_confusion[['pair', 'error_rate']].to_latex(
        index=False,
        float_format='%.2f',
        column_format='lc',
        caption='Top 10 most confused class pairs.',
        label='tab:confused-classes',
        header=['True $\\rightarrow$ Predicted', 'Error Rate (\\%)']
    )
    with open(results_dir / 'confusion_pairs_table.tex', 'w') as f:
        f.write(latex_table)

    # 3. Calibration curves
    print("\n" + "=" * 70)
    print("Generating Calibration Curves")
    print("=" * 70)

    plot_calibration_curves(model, cal_loader, eval_loader, results_dir, device)

    # 4. Method comparison from test.py results
    print("\n" + "=" * 70)
    print("Method Comparison Chart")
    print("=" * 70)

    # Check if conformal_results.csv exists from test.py
    conformal_csv = results_dir / 'conformal_results.csv'
    if conformal_csv.exists():
        conformal_df = pd.read_csv(conformal_csv)
        results_list = conformal_df.to_dict('records')
        plot_method_comparison(results_list, args.alpha, results_dir)
    else:
        print("  Warning: conformal_results.csv not found. Run test.py first.")

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print(f"\nAll outputs saved to: {results_dir}/")


if __name__ == '__main__':
    main()
