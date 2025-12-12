#!/usr/bin/env python3
"""
Explainability Testing for Conformal Prediction on CUB-200-2011

This script evaluates explainability methods (SHAP, GradCAM) in the context
of conformal prediction sets, analyzing uncertainty types and visual features.
"""

import os
import sys
import argparse
from typing import Optional, Dict, List, Tuple
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from scipy.spatial.distance import cosine

from torchvision import transforms, models
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import MetricCollection

from torchcp.classification.score import APS, LAC, EntmaxScore
from torchcp.classification.predictor import SplitPredictor, ClassConditionalPredictor, RC3PPredictor

# Import all helpers from unified module
from helpers import (
    # Dataset utilities
    Cub2011Dataset,
    create_dataloaders,
    # Explainability methods
    explain_predictions_with_shap,
    explain_predictions_with_gradcam,
    simple_differential,
    visualize_all,
    # Evaluation classes
    ConformalExplainabilityEvaluator,
    DifferentialHeatmapAnalyzer,
    # Analysis functions
    evaluate_on_conformal_sets,
    generate_differential_insights,
    # Visualization functions
    plot_comprehensive_evaluation,
    create_differential_visualization,
    create_uncertainty_landscape_plot,
    generate_summary_report
)


# ==============================================================================
# Model Definition
# ==============================================================================

class LitResNet(pl.LightningModule):
    """ResNet50 model for bird species classification."""

    def __init__(
        self,
        num_classes: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        pretrained: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        if pretrained:
            for param in self.model.parameters():
                param.requires_grad = False
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        self.criterion = nn.CrossEntropyLoss()

        common_metrics = {
            "Accuracy": torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes, average="micro"
            ),
            "Precision": torchmetrics.Precision(
                task="multiclass", num_classes=num_classes, average="macro"
            ),
            "Recall": torchmetrics.Recall(
                task="multiclass", num_classes=num_classes, average="macro"
            ),
            "F1Score": torchmetrics.F1Score(
                task="multiclass", num_classes=num_classes, average="macro"
            ),
            "AveragePrecision": torchmetrics.AveragePrecision(
                task="multiclass", num_classes=num_classes, average="macro"
            ),
        }

        metrics = MetricCollection(common_metrics)
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_metrics(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for name, metric in self.train_metrics.items():
            self.log(name, metric, on_step=False, on_epoch=True,
                    prog_bar=("Accuracy" in name or "F1Score" in name))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_metrics(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for name, metric in self.val_metrics.items():
            self.log(name, metric, on_step=False, on_epoch=True,
                    prog_bar=("Accuracy" in name or "F1Score" in name))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]


# ==============================================================================
# Robust Explainability Evaluator
# ==============================================================================

class RobustExplainabilityEvaluator:
    """Evaluator for explainability methods with robust metrics."""

    def __init__(self, grid_size: int = 14):
        """
        Args:
            grid_size: Grid size for spatial comparison (14x14 for 448x448 images)
        """
        self.grid_size = grid_size

    def _to_grid(self, attr_map: torch.Tensor) -> torch.Tensor:
        """Downsamples attribution map to grid with binary thresholding."""
        grid = F.adaptive_avg_pool2d(attr_map, (self.grid_size, self.grid_size))
        threshold = grid.mean()
        return (grid > threshold).float()

    def calculate_grid_iou(self, map_a: torch.Tensor, map_b: torch.Tensor) -> float:
        """Calculate IoU between two attribution maps."""
        grid_a = self._to_grid(map_a)
        grid_b = self._to_grid(map_b)
        intersection = (grid_a * grid_b).sum()
        union = torch.max(grid_a, grid_b).sum()
        if union == 0:
            return 0.0
        return (intersection / union).item()

    def calculate_gini(self, attr_map: torch.Tensor) -> float:
        """Calculate Gini coefficient (focus/sparsity measure)."""
        flat = attr_map.flatten().cpu().numpy()
        flat = np.sort(flat)
        n = len(flat)
        if n == 0 or np.sum(flat) == 0:
            return 0.0
        index = np.arange(1, n + 1)
        return (np.sum((2 * index - n - 1) * flat)) / (n * np.sum(flat))

    def calculate_bbox_energy(
        self, attr_map: torch.Tensor, bbox: List[int], image_size: int = 448
    ) -> float:
        """Calculate percentage of attribution within bounding box."""
        mask = torch.zeros_like(attr_map)
        x, y, w, h = [int(c) for c in bbox]
        x = max(0, min(x, image_size))
        y = max(0, min(y, image_size))
        w = max(0, min(w, image_size - x))
        h = max(0, min(h, image_size - y))
        mask[:, :, y:y+h, x:x+w] = 1.0
        total_energy = attr_map.sum()
        if total_energy == 0:
            return 0.0
        energy_in_box = (attr_map * mask).sum()
        return (energy_in_box / total_energy).item()


# ==============================================================================
# Conformal Prediction Utilities
# ==============================================================================

def setup_conformal_predictor(
    model: nn.Module,
    cal_loader: DataLoader,
    alpha: float = 0.05,
    predictor_cls = SplitPredictor,
    score_fn = None,
    device: str = 'cuda'
) -> object:
    """Set up and calibrate conformal predictor."""
    if score_fn is None:
        score_fn = LAC()

    model = model.to(device)
    model.eval()
    predictor = predictor_cls(score_fn, model)
    predictor.calibrate(cal_loader, alpha=alpha)
    return predictor


def get_prediction_set(
    predictor: object,
    image: torch.Tensor,
    device: str = 'cuda'
) -> np.ndarray:
    """Get conformal prediction set for an image."""
    with torch.no_grad():
        pred_mask = predictor.predict(image.unsqueeze(0).to(device))
        prediction_set = torch.nonzero(pred_mask[0]).flatten().cpu().numpy()
    return prediction_set


# ==============================================================================
# Explanation Wrapper Functions
# ==============================================================================

def create_explanation_wrappers(model: nn.Module, device: str = 'cuda'):
    """Create wrapper functions for SHAP and GradCAM explanations."""

    def shap_wrapper(image: torch.Tensor, class_idx: int) -> torch.Tensor:
        """Generate smoothed SHAP explanation."""
        idx_tensor = torch.tensor([class_idx], device=device) if isinstance(class_idx, int) else class_idx

        results = explain_predictions_with_shap(
            model=model,
            input_tensor=image,
            class_indices=idx_tensor,
            method='gradient',
            n_samples=50,
            device=device
        )

        # Normalize: Abs -> Grayscale -> Smooth -> 0-1
        attr = torch.abs(results['attributions'][int(class_idx)])
        attr = torch.sum(attr, dim=1, keepdim=True)
        attr = F.avg_pool2d(attr, kernel_size=9, stride=1, padding=4)
        attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
        return attr

    def gradcam_wrapper(image: torch.Tensor, class_idx: int) -> torch.Tensor:
        """Generate GradCAM explanation."""
        idx_tensor = torch.tensor([class_idx], device=device) if isinstance(class_idx, int) else class_idx

        # For regular ResNet50, the target layer is model.layer4[-1] (not model.model.layer4[-1])
        results = explain_predictions_with_gradcam(
            model=model,
            input_tensor=image,
            class_indices=idx_tensor,
            target_layer=model.layer4[-1],  # Updated for regular PyTorch model
            device=device
        )

        # Normalize: Mean -> 0-1
        attr = torch.mean(results['attributions'][int(class_idx)], dim=1, keepdim=True)
        attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
        return attr

    return shap_wrapper, gradcam_wrapper


# ==============================================================================
# Evaluation Functions
# ==============================================================================

def run_robust_evaluation(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    conformal_predictor: object,
    n_samples: int = 50,
    device: str = 'cuda'
) -> Dict:
    """Run comprehensive evaluation on conformal prediction sets."""
    print(f"Starting evaluation on {n_samples} samples...")

    shap_fn, gradcam_fn = create_explanation_wrappers(model, device=device)
    evaluator = RobustExplainabilityEvaluator(grid_size=14)

    results = defaultdict(list)
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    for idx in tqdm(indices, desc="Evaluating samples"):
        image, label = dataset[idx][:2]

        # Get prediction set
        prediction_set = get_prediction_set(conformal_predictor, image, device)

        # Skip if no uncertainty
        if len(prediction_set) <= 1:
            continue

        # Limit to top 3 classes for speed
        classes_to_test = prediction_set[:3]
        sample_maps = {'shap': {}, 'gradcam': {}}

        for class_idx in classes_to_test:
            idx_int = int(class_idx)

            try:
                s_map = shap_fn(image, idx_int)
                g_map = gradcam_fn(image, idx_int)

                sample_maps['shap'][idx_int] = s_map
                sample_maps['gradcam'][idx_int] = g_map

                # Spatial agreement
                agreement = evaluator.calculate_grid_iou(s_map, g_map)
                results['spatial_agreement'].append(agreement)

                # Focus scores
                results['shap_gini'].append(evaluator.calculate_gini(s_map))
                results['gradcam_gini'].append(evaluator.calculate_gini(g_map))

            except Exception as e:
                print(f"Error processing class {idx_int}: {e}")
                continue

        # Inter-class overlap analysis
        keys = list(sample_maps['shap'].keys())
        if len(keys) >= 2:
            overlap = evaluator.calculate_grid_iou(
                sample_maps['shap'][keys[0]],
                sample_maps['shap'][keys[1]]
            )
            results['inter_class_overlap'].append(overlap)
            results['set_size'].append(len(prediction_set))

    return results


def analyze_confused_species(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    class_names: List[str],
    confused_pairs: List[Tuple[str, str, float]],
    shap_fn,
    gradcam_fn,
    evaluator: ConformalExplainabilityEvaluator,
    device: str = 'cuda',
    output_dir: str = '.'
):
    """Analyze most confused species pairs."""
    print("=" * 80)
    print("ANALYSIS OF MOST CONFUSED SPECIES PAIRS")
    print("=" * 80)

    for true_name, pred_name, error_rate in confused_pairs:
        print(f"\n\nAnalyzing: {true_name} → {pred_name} (Error Rate: {error_rate:.1%})")
        print("-" * 60)

        # Get class indices
        try:
            true_idx = class_names.index(true_name.replace('_', ' '))
            pred_idx = class_names.index(pred_name.replace('_', ' '))
        except ValueError:
            print(f"  → Could not find classes in dataset, skipping")
            continue

        # Find examples of this confusion
        confusion_examples = []
        for idx in range(min(100, len(dataset))):
            if len(confusion_examples) >= 3:
                break

            image, label = dataset[idx][:2]
            if label != true_idx:
                continue

            with torch.no_grad():
                logits = model(image.unsqueeze(0).to(device))
                probs = torch.softmax(logits, dim=1)[0]

                if probs[pred_idx] > 0.1:
                    confusion_examples.append({
                        'idx': idx,
                        'image': image,
                        'true_prob': probs[true_idx].item(),
                        'pred_prob': probs[pred_idx].item()
                    })

        if not confusion_examples:
            print("  No examples found in this sample")
            continue

        # Analyze first example
        example = confusion_examples[0]
        shap_true = shap_fn(example['image'], true_idx)
        shap_pred = shap_fn(example['image'], pred_idx)

        # Compute similarity
        shap_sim = 1 - cosine(
            shap_true.flatten().cpu().numpy(),
            shap_pred.flatten().cpu().numpy()
        )

        print(f"\n  Explanation Similarity Analysis:")
        print(f"    SHAP similarity: {shap_sim:.3f}")
        print(f"    True class confidence: {example['true_prob']:.3f}")
        print(f"    Confused class confidence: {example['pred_prob']:.3f}")

        if shap_sim > 0.7:
            print("    → High similarity: Aleatoric uncertainty (visual similarity)")
        elif shap_sim < 0.3:
            print("    → Low similarity: Epistemic uncertainty (spurious correlations)")
        else:
            print("    → Mixed similarity: Partial feature overlap")


def plot_uncertainty_analysis(
    results: Dict,
    save_path: str = 'uncertainty_vs_overlap.png'
):
    """Plot relationship between set size and inter-class overlap."""
    if 'set_size' not in results or 'inter_class_overlap' not in results:
        print("Insufficient data for uncertainty analysis plot")
        return

    plt.figure(figsize=(10, 6))
    sns.regplot(
        x=results['set_size'],
        y=results['inter_class_overlap'],
        scatter_kws={'alpha': 0.6},
        line_kws={'color': 'red'}
    )

    plt.xlabel('Conformal Set Size (Uncertainty)', fontsize=12, fontweight='bold')
    plt.ylabel('Inter-Class Feature Overlap (IoU)', fontsize=12, fontweight='bold')
    plt.title('Visual Ambiguity Drives Uncertainty', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Uncertainty analysis plot saved to {save_path}")


def generate_latex_table(results: Dict, save_path: str = 'results_table.tex'):
    """Generate LaTeX table of evaluation metrics."""
    metrics_data = []

    # Focus Score
    if 'shap_gini' in results and results['shap_gini']:
        shap_focus = results['shap_gini']
        grad_focus = results['gradcam_gini']
        metrics_data.append([
            'Focus Score (Gini) $\\uparrow$',
            f"{np.mean(shap_focus):.3f} $\\pm$ {np.std(shap_focus):.3f}",
            f"{np.mean(grad_focus):.3f} $\\pm$ {np.std(grad_focus):.3f}",
            '\\textbf{SHAP}' if np.mean(shap_focus) > np.mean(grad_focus) else '\\textbf{GradCAM}'
        ])

    # Spatial Agreement
    if 'spatial_agreement' in results and results['spatial_agreement']:
        spatial = results['spatial_agreement']
        metrics_data.append([
            'Spatial Agreement (IoU)',
            '-',
            '-',
            f"{np.mean(spatial):.3f} $\\pm$ {np.std(spatial):.3f}"
        ])

    # Inter-class Overlap
    if 'inter_class_overlap' in results and results['inter_class_overlap']:
        overlap = results['inter_class_overlap']
        metrics_data.append([
            'Inter-Class Overlap',
            '-',
            '-',
            f"{np.mean(overlap):.3f} $\\pm$ {np.std(overlap):.3f}"
        ])

    df = pd.DataFrame(metrics_data, columns=['Metric', 'SHAP', 'GradCAM', 'Value'])
    latex_table = df.to_latex(index=False, escape=False, column_format='l|cc|c')

    with open(save_path, 'w') as f:
        f.write(latex_table)

    print(f"✓ LaTeX table saved to {save_path}")
    print("\n" + latex_table)


def print_summary_insights(results: Dict):
    """Print summary of key insights."""
    print("\n" + "=" * 80)
    print("KEY INSIGHTS SUMMARY")
    print("=" * 80)

    if 'set_size' in results and results['set_size']:
        avg_set_size = np.mean(results['set_size'])
        print(f"\n1. Average Conformal Set Size: {avg_set_size:.2f}")

    if 'spatial_agreement' in results and results['spatial_agreement']:
        agreement = np.mean(results['spatial_agreement'])
        print(f"\n2. SHAP-GradCAM Spatial Agreement: {agreement:.3f}")
        if agreement > 0.5:
            print("   → High agreement: Methods capture similar features")
        else:
            print("   → Low agreement: Methods capture different aspects")

    if 'shap_gini' in results and results['shap_gini']:
        shap_focus = np.mean(results['shap_gini'])
        grad_focus = np.mean(results['gradcam_gini'])
        print(f"\n3. Focus Scores (Gini Coefficient):")
        print(f"   SHAP: {shap_focus:.3f}")
        print(f"   GradCAM: {grad_focus:.3f}")
        if shap_focus > grad_focus:
            print("   → SHAP produces more focused explanations")
        else:
            print("   → GradCAM produces more focused explanations")

    if 'inter_class_overlap' in results and results['inter_class_overlap']:
        overlap = np.mean(results['inter_class_overlap'])
        print(f"\n4. Inter-Class Feature Overlap: {overlap:.3f}")
        if overlap > 0.5:
            print("   → High overlap suggests aleatoric uncertainty (visual similarity)")
        else:
            print("   → Low overlap suggests distinct visual features")

    print("\n" + "=" * 80)


def save_detailed_metrics(results: Dict, output_dir: str):
    """Save detailed metrics to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save all metrics to CSV
    metrics_df = pd.DataFrame({
        'metric': ['spatial_agreement', 'shap_gini', 'gradcam_gini', 'inter_class_overlap', 'set_size'],
        'mean': [
            np.mean(results.get('spatial_agreement', [0])),
            np.mean(results.get('shap_gini', [0])),
            np.mean(results.get('gradcam_gini', [0])),
            np.mean(results.get('inter_class_overlap', [0])),
            np.mean(results.get('set_size', [0]))
        ],
        'std': [
            np.std(results.get('spatial_agreement', [0])),
            np.std(results.get('shap_gini', [0])),
            np.std(results.get('gradcam_gini', [0])),
            np.std(results.get('inter_class_overlap', [0])),
            np.std(results.get('set_size', [0]))
        ],
        'min': [
            np.min(results.get('spatial_agreement', [0])),
            np.min(results.get('shap_gini', [0])),
            np.min(results.get('gradcam_gini', [0])),
            np.min(results.get('inter_class_overlap', [0])),
            np.min(results.get('set_size', [0]))
        ],
        'max': [
            np.max(results.get('spatial_agreement', [0])),
            np.max(results.get('shap_gini', [0])),
            np.max(results.get('gradcam_gini', [0])),
            np.max(results.get('inter_class_overlap', [0])),
            np.max(results.get('set_size', [0]))
        ]
    })

    metrics_path = os.path.join(output_dir, 'explainability_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✓ Detailed metrics saved to {metrics_path}")


def plot_additional_visualizations(results: Dict, output_dir: str):
    """Create additional visualization plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Distribution comparison (SHAP vs GradCAM Gini)
    if 'shap_gini' in results and 'gradcam_gini' in results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram
        axes[0].hist(results['shap_gini'], bins=30, alpha=0.6, label='SHAP', color='blue', edgecolor='black')
        axes[0].hist(results['gradcam_gini'], bins=30, alpha=0.6, label='GradCAM', color='red', edgecolor='black')
        axes[0].set_xlabel('Gini Coefficient (Focus Score)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title('Focus Score Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        # Box plot
        data_to_plot = [results['shap_gini'], results['gradcam_gini']]
        bp = axes[1].boxplot(data_to_plot, labels=['SHAP', 'GradCAM'], patch_artist=True)
        bp['boxes'][0].set_facecolor('blue')
        bp['boxes'][1].set_facecolor('red')
        axes[1].set_ylabel('Gini Coefficient', fontsize=12, fontweight='bold')
        axes[1].set_title('Focus Score Comparison', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(output_dir, 'focus_score_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Focus score comparison saved to {save_path}")

    # Plot 2: Spatial agreement distribution
    if 'spatial_agreement' in results and results['spatial_agreement']:
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(results['spatial_agreement'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(x=np.mean(results['spatial_agreement']), color='red', linestyle='--',
                   linewidth=2, label=f"Mean: {np.mean(results['spatial_agreement']):.3f}")
        ax.set_xlabel('Spatial Agreement (IoU)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('SHAP-GradCAM Spatial Agreement Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(output_dir, 'spatial_agreement_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Spatial agreement distribution saved to {save_path}")

    # Plot 3: Comprehensive metrics summary
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = []
    values = []
    errors = []

    if 'spatial_agreement' in results and results['spatial_agreement']:
        metrics.append('Spatial\nAgreement')
        values.append(np.mean(results['spatial_agreement']))
        errors.append(np.std(results['spatial_agreement']))

    if 'shap_gini' in results and results['shap_gini']:
        metrics.append('SHAP\nFocus')
        values.append(np.mean(results['shap_gini']))
        errors.append(np.std(results['shap_gini']))

    if 'gradcam_gini' in results and results['gradcam_gini']:
        metrics.append('GradCAM\nFocus')
        values.append(np.mean(results['gradcam_gini']))
        errors.append(np.std(results['gradcam_gini']))

    if 'inter_class_overlap' in results and results['inter_class_overlap']:
        metrics.append('Inter-Class\nOverlap')
        values.append(np.mean(results['inter_class_overlap']))
        errors.append(np.std(results['inter_class_overlap']))

    if metrics:
        bars = ax.bar(metrics, values, yerr=errors, capsize=10, color='steelblue',
                      edgecolor='black', alpha=0.7, error_kw={'linewidth': 2})
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Explainability Metrics Summary', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(output_dir, 'metrics_summary.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Metrics summary saved to {save_path}")


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate explainability methods on conformal prediction sets'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./best-epoch=17.ckpt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--cub-root',
        type=str,
        default='./CUB_200_2011/',
        help='Path to CUB-200-2011 dataset'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for data loading'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=50,
        help='Number of samples to evaluate'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Conformal prediction significance level'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Directory for output files'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=448,
        help='Input image size'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Define transforms
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    # Load data
    print("Loading dataset...")
    train_loader, test_loader = create_dataloaders(
        root_dir=args.cub_root,
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=args.batch_size,
        use_bbox=True
    )

    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset

    # Split test set into calibration and evaluation
    cal_size = int(len(test_dataset) * 0.5)
    test_size = len(test_dataset) - cal_size

    cal_dataset, eval_dataset = random_split(
        test_dataset,
        [cal_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    cal_loader = DataLoader(cal_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train set size: {len(train_dataset)}")
    print(f"Calibration set size: {len(cal_dataset)}")
    print(f"Evaluation set size: {len(eval_dataset)}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")

    # Use the same model loading as test.py and analysis.py
    from helpers import create_resnet50_model

    model = create_resnet50_model(num_classes=200, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("Model loaded successfully!")

    # Setup conformal predictor
    print("Setting up conformal predictor...")
    conformal_predictor = setup_conformal_predictor(
        model=model,
        cal_loader=cal_loader,
        alpha=args.alpha,
        device=device
    )

    # Run evaluation
    print("\n" + "=" * 80)
    print("RUNNING EVALUATION")
    print("=" * 80)

    results = run_robust_evaluation(
        model=model,
        dataset=eval_dataset,
        conformal_predictor=conformal_predictor,
        n_samples=args.n_samples,
        device=device
    )

    # Generate visualizations and reports
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS AND REPORTS")
    print("=" * 80)

    print("\n1. Uncertainty analysis plot...")
    plot_uncertainty_analysis(
        results,
        save_path=os.path.join(args.output_dir, 'uncertainty_vs_overlap.png')
    )

    print("\n2. LaTeX table...")
    generate_latex_table(
        results,
        save_path=os.path.join(args.output_dir, 'results_table.tex')
    )

    print("\n3. Detailed metrics CSV...")
    save_detailed_metrics(results, args.output_dir)

    print("\n4. Additional visualizations...")
    plot_additional_visualizations(results, args.output_dir)

    # Print summary
    print_summary_insights(results)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"\nOutput files saved to: {args.output_dir}/")
    print("  - uncertainty_vs_overlap.png")
    print("  - results_table.tex")
    print("  - explainability_metrics.csv")
    print("  - focus_score_comparison.png")
    print("  - spatial_agreement_distribution.png")
    print("  - metrics_summary.png")


if __name__ == '__main__':
    main()
