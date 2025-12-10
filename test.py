#!/usr/bin/env python3
"""
Test Script for Fine-tuned Model with Explainability and Metrics

This script:
1. Computes AUROC and AUPRC metrics
2. Evaluates different conformal predictors (Split, Class-Conditional, RC3P)
3. Compares score functions (LAC, APS, Entmax)
4. Visualizes samples with explainability methods
5. Saves all results and charts to results/
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
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# Metrics
from sklearn.metrics import roc_auc_score, average_precision_score

# TorchCP for conformal prediction
try:
    from torchcp.classification.score import APS, LAC
    try:
        from torchcp.classification.score import EntmaxScore
        ENTMAX_AVAILABLE = True
    except ImportError:
        ENTMAX_AVAILABLE = False
        print("Warning: EntmaxScore not available in this torchcp version")

    from torchcp.classification.predictor import SplitPredictor, ClassConditionalPredictor
    try:
        from torchcp.classification.predictor import RC3PPredictor
        RC3P_AVAILABLE = True
    except ImportError:
        RC3P_AVAILABLE = False
        print("Warning: RC3PPredictor not available in this torchcp version")

    TORCHCP_AVAILABLE = True
except ImportError:
    print("Error: torchcp not available. Install with: pip install torchcp")
    TORCHCP_AVAILABLE = False
    exit(1)

# Import helpers
from helpers import (
    create_dataloaders,
    create_resnet50_model,
    explain_predictions_with_shap,
    explain_predictions_with_gradcam,
    simple_differential
)


def denormalize_image(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize image tensor for visualization."""
    mean_tensor = torch.tensor(mean).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std).view(1, 3, 1, 1)
    denorm = image_tensor * std_tensor + mean_tensor
    return torch.clamp(denorm, 0, 1)


def compute_classification_metrics(model, test_loader, device='cuda'):
    """Compute AUROC and AUPRC metrics."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Computing metrics"):
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)

    # Convert labels to one-hot
    n_classes = all_probs.shape[1]
    all_labels_onehot = np.eye(n_classes)[all_labels]

    # Compute metrics
    auroc = roc_auc_score(all_labels_onehot, all_probs, average='macro', multi_class='ovr')
    auprc = average_precision_score(all_labels_onehot, all_probs, average='macro')

    return auroc, auprc


def evaluate_conformal_predictor(predictor, eval_loader, device='cuda'):
    """Evaluate conformal predictor and return coverage and average set size."""
    coverages = []
    set_sizes = []

    with torch.no_grad():
        for images, labels in tqdm(eval_loader, desc="Evaluating conformal", leave=False):
            images = images.to(device)
            labels = labels.cpu().numpy()

            # Get prediction sets
            pred_sets = predictor.predict(images)
            pred_sets = pred_sets.cpu().numpy()

            # Compute coverage and set size
            for i in range(len(labels)):
                pred_set_indices = np.where(pred_sets[i] == 1)[0]
                set_sizes.append(len(pred_set_indices))
                coverages.append(int(labels[i] in pred_set_indices))

    coverage = np.mean(coverages)
    avg_set_size = np.mean(set_sizes)

    return coverage, avg_set_size, set_sizes


def run_conformal_experiments(model, cal_loader, eval_loader, alpha=0.05, device='cuda'):
    """Run comprehensive conformal prediction experiments."""
    results = []

    # Score functions to test
    score_configs = [
        ('APS', APS()),
        ('LAC', LAC()),
    ]

    if ENTMAX_AVAILABLE:
        score_configs.append(('Entmax', EntmaxScore()))

    # Predictor classes to test
    predictor_configs = [
        ('Split', SplitPredictor),
        ('Class-Conditional', ClassConditionalPredictor),
    ]

    if RC3P_AVAILABLE:
        predictor_configs.append(('RC3P', RC3PPredictor))

    print("\n" + "=" * 70)
    print("Running Conformal Prediction Experiments")
    print("=" * 70)

    for predictor_name, predictor_cls in predictor_configs:
        for score_name, score_fn in score_configs:
            print(f"\n{predictor_name} + {score_name}:")

            try:
                # Create and calibrate predictor
                predictor = predictor_cls(score_fn, model)
                predictor.calibrate(cal_loader, alpha=alpha)

                # Evaluate
                coverage, avg_set_size, set_sizes = evaluate_conformal_predictor(
                    predictor, eval_loader, device
                )

                print(f"  Coverage: {coverage:.4f}")
                print(f"  Avg Set Size: {avg_set_size:.2f}")

                results.append({
                    'predictor': predictor_name,
                    'score': score_name,
                    'coverage': coverage,
                    'avg_set_size': avg_set_size,
                    'set_sizes': set_sizes
                })

            except Exception as e:
                print(f"  Error: {e}")
                continue

    return results


def plot_conformal_results(results, save_dir, alpha=0.05):
    """Create visualizations for conformal prediction results."""
    df = pd.DataFrame([{
        'predictor': r['predictor'],
        'score': r['score'],
        'coverage': r['coverage'],
        'avg_set_size': r['avg_set_size']
    } for r in results])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Coverage by predictor and score
    pivot_coverage = df.pivot(index='predictor', columns='score', values='coverage')
    ax1 = axes[0, 0]
    pivot_coverage.plot(kind='bar', ax=ax1, rot=45, width=0.8)
    ax1.axhline(y=1-alpha, color='red', linestyle='--', linewidth=2, label=f'Target ({1-alpha:.0%})')
    ax1.set_ylabel('Coverage', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Predictor', fontsize=12, fontweight='bold')
    ax1.set_title('Coverage by Predictor and Score Function', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Average set size by predictor and score
    pivot_size = df.pivot(index='predictor', columns='score', values='avg_set_size')
    ax2 = axes[0, 1]
    pivot_size.plot(kind='bar', ax=ax2, rot=45, width=0.8)
    ax2.set_ylabel('Average Set Size', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Predictor', fontsize=12, fontweight='bold')
    ax2.set_title('Average Set Size by Predictor and Score', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Heatmap of set sizes
    ax3 = axes[1, 0]
    pivot_size_heatmap = df.pivot(index='predictor', columns='score', values='avg_set_size')
    sns.heatmap(pivot_size_heatmap, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax3,
                cbar_kws={'label': 'Avg Set Size'})
    ax3.set_title('Set Size Heatmap', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Predictor', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Score Function', fontsize=12, fontweight='bold')

    # Plot 4: Set size distribution for best predictor
    ax4 = axes[1, 1]
    best_result = min(results, key=lambda x: x['avg_set_size'])
    ax4.hist(best_result['set_sizes'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax4.axvline(x=best_result['avg_set_size'], color='red', linestyle='--',
                linewidth=2, label=f"Mean: {best_result['avg_set_size']:.2f}")
    ax4.set_xlabel('Set Size', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title(f"Set Size Distribution\n{best_result['predictor']} + {best_result['score']}",
                  fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = Path(save_dir) / 'conformal_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nConformal analysis chart saved to {save_path}")


def visualize_sample_with_explanations(
    image,
    true_label,
    prediction_set,
    model,
    class_names,
    device='cuda',
    save_path=None
):
    """Visualize a sample with model predictions and explainability methods."""
    model.eval()

    # Get model predictions
    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
        probs = F.softmax(logits, dim=1)[0]

    # Denormalize image for visualization
    img_for_viz = denormalize_image(image.unsqueeze(0)).squeeze()
    img_np = img_for_viz.permute(1, 2, 0).cpu().numpy()

    # Limit to top 3 classes in prediction set
    classes_to_explain = prediction_set[:min(3, len(prediction_set))]
    n_classes = len(classes_to_explain)

    # Create figure
    fig, axes = plt.subplots(n_classes + 1, 4, figsize=(16, 4 * (n_classes + 1)))
    if n_classes == 0:
        axes = axes.reshape(1, -1)

    # Row 0: Show original image and prediction info
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # Show prediction set info
    pred_text = f"True: {class_names[true_label]}\n\n"
    pred_text += f"Set (n={len(prediction_set)}):\n"
    for i, cls_idx in enumerate(classes_to_explain):
        prob = probs[cls_idx].item()
        pred_text += f"{i+1}. {class_names[cls_idx]}: {prob:.3f}\n"

    axes[0, 1].text(0.1, 0.5, pred_text, fontsize=10, va='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[0, 1].axis('off')
    axes[0, 2].axis('off')
    axes[0, 3].axis('off')

    # Generate explanations for each class
    class_indices_tensor = torch.tensor(classes_to_explain, device=device)

    # SHAP explanations
    shap_results = explain_predictions_with_shap(
        model=model,
        input_tensor=image,
        class_indices=class_indices_tensor,
        method='gradient',
        n_samples=50,
        device=device,
        denormalize=False
    )

    # GradCAM explanations
    gradcam_results = explain_predictions_with_gradcam(
        model=model,
        input_tensor=image,
        class_indices=class_indices_tensor,
        device=device,
        denormalize=False
    )

    # Differential explanations
    shap_differential = simple_differential(shap_results, brightness=3.0)
    gradcam_differential = simple_differential(gradcam_results, brightness=3.0)

    # Visualize each class
    for row_idx, cls_idx in enumerate(classes_to_explain, start=1):
        cls_name = class_names[cls_idx]
        prob = probs[cls_idx].item()

        # Column 0: SHAP
        shap_attr = shap_results['attributions'][cls_idx].squeeze().numpy()
        shap_heatmap = np.abs(shap_attr).sum(axis=0)
        shap_heatmap = (shap_heatmap - shap_heatmap.min()) / (shap_heatmap.max() - shap_heatmap.min() + 1e-10)
        shap_smooth = gaussian_filter(shap_heatmap, sigma=2)

        axes[row_idx, 0].imshow(img_np)
        im0 = axes[row_idx, 0].imshow(shap_smooth, cmap='jet', alpha=0.6)
        axes[row_idx, 0].set_title(f'SHAP\n{cls_name} ({prob:.3f})', fontsize=10, fontweight='bold')
        axes[row_idx, 0].axis('off')
        plt.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)

        # Column 1: GradCAM
        gradcam_attr = gradcam_results['attributions'][cls_idx].squeeze().numpy()
        gradcam_heatmap = np.mean(gradcam_attr, axis=0)
        gradcam_heatmap = (gradcam_heatmap - gradcam_heatmap.min()) / (gradcam_heatmap.max() - gradcam_heatmap.min() + 1e-10)
        gradcam_smooth = gaussian_filter(gradcam_heatmap, sigma=2)

        axes[row_idx, 1].imshow(img_np)
        im1 = axes[row_idx, 1].imshow(gradcam_smooth, cmap='jet', alpha=0.6)
        axes[row_idx, 1].set_title(f'GradCAM\n{cls_name} ({prob:.3f})', fontsize=10, fontweight='bold')
        axes[row_idx, 1].axis('off')
        plt.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

        # Column 2: SHAP Differential
        shap_diff = shap_differential[cls_idx]
        shap_diff_smooth = gaussian_filter(shap_diff, sigma=2)

        axes[row_idx, 2].imshow(img_np)
        im2 = axes[row_idx, 2].imshow(shap_diff_smooth, cmap='jet', alpha=0.6)
        axes[row_idx, 2].set_title(f'SHAP Differential\n(Unique Features)', fontsize=10, fontweight='bold')
        axes[row_idx, 2].axis('off')
        plt.colorbar(im2, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)

        # Column 3: GradCAM Differential
        gradcam_diff = gradcam_differential[cls_idx]
        gradcam_diff_smooth = gaussian_filter(gradcam_diff, sigma=2)

        axes[row_idx, 3].imshow(img_np)
        im3 = axes[row_idx, 3].imshow(gradcam_diff_smooth, cmap='jet', alpha=0.6)
        axes[row_idx, 3].set_title(f'GradCAM Differential\n(Unique Features)', fontsize=10, fontweight='bold')
        axes[row_idx, 3].axis('off')
        plt.colorbar(im3, ax=axes[row_idx, 3], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Test fine-tuned model with metrics and explainability'
    )

    # Model parameters
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth',
                        help='Path to model checkpoint')

    # Data parameters
    parser.add_argument('--data-dir', type=str, default='./CUB_200_2011',
                        help='Path to CUB-200-2011 dataset')
    parser.add_argument('--image-size', type=int, default=448,
                        help='Image size (default: 448)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')

    # Conformal prediction parameters
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Conformal prediction alpha (default: 0.05 for 95%% coverage)')

    # Test parameters
    parser.add_argument('--n-samples', type=int, default=10,
                        help='Number of test samples to visualize (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    # Output parameters
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='Directory to save results')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ========================================================================
    # Load Data
    # ========================================================================

    print("\n" + "=" * 70)
    print("Loading Data")
    print("=" * 70)

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
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

    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of classes: {len(class_names)}")

    # Split test set for calibration and evaluation
    cal_size = int(len(test_dataset) * 0.5)
    eval_size = len(test_dataset) - cal_size

    cal_dataset, eval_dataset = random_split(
        test_dataset,
        [cal_size, eval_size],
        generator=torch.Generator().manual_seed(42)
    )

    cal_loader = DataLoader(cal_dataset, batch_size=args.batch_size, shuffle=False)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Calibration set: {len(cal_dataset)}")
    print(f"Evaluation set: {len(eval_dataset)}")

    # ========================================================================
    # Load Model
    # ========================================================================

    print("\n" + "=" * 70)
    print("Loading Model")
    print("=" * 70)

    model = create_resnet50_model(num_classes=200, pretrained=False)

    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    if 'test_acc' in checkpoint:
        print(f"Checkpoint test accuracy: {checkpoint['test_acc']*100:.2f}%")

    # ========================================================================
    # Compute Classification Metrics
    # ========================================================================

    print("\n" + "=" * 70)
    print("Computing Classification Metrics")
    print("=" * 70)

    auroc, auprc = compute_classification_metrics(model, test_loader, device)

    print(f"\nAUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")

    # ========================================================================
    # Run Conformal Prediction Experiments
    # ========================================================================

    conformal_results = run_conformal_experiments(
        model=model,
        cal_loader=cal_loader,
        eval_loader=eval_loader,
        alpha=args.alpha,
        device=device
    )

    # ========================================================================
    # Save Numerical Results
    # ========================================================================

    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)

    # Save metrics to CSV
    metrics_data = {
        'AUROC': [auroc],
        'AUPRC': [auprc]
    }

    for result in conformal_results:
        key = f"{result['predictor']}_{result['score']}"
        metrics_data[f'{key}_Coverage'] = [result['coverage']]
        metrics_data[f'{key}_AvgSetSize'] = [result['avg_set_size']]

    metrics_df = pd.DataFrame(metrics_data)
    metrics_csv_path = results_dir / 'metrics.csv'
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nMetrics saved to {metrics_csv_path}")

    # Save detailed conformal results
    conformal_df = pd.DataFrame([{
        'Predictor': r['predictor'],
        'Score': r['score'],
        'Coverage': r['coverage'],
        'Avg_Set_Size': r['avg_set_size']
    } for r in conformal_results])

    conformal_csv_path = results_dir / 'conformal_results.csv'
    conformal_df.to_csv(conformal_csv_path, index=False)
    print(f"Conformal results saved to {conformal_csv_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("CONFORMAL PREDICTION RESULTS")
    print("=" * 70)
    print(conformal_df.to_string(index=False))

    # ========================================================================
    # Create Visualizations
    # ========================================================================

    plot_conformal_results(conformal_results, results_dir, args.alpha)

    # ========================================================================
    # Visualize Sample Predictions with Explainability
    # ========================================================================

    print("\n" + "=" * 70)
    print(f"Visualizing {args.n_samples} Sample Predictions")
    print("=" * 70)

    # Use best predictor for sample visualizations
    best_result = min(conformal_results, key=lambda x: x['avg_set_size'])
    print(f"\nUsing best predictor: {best_result['predictor']} + {best_result['score']}")

    # Recreate best predictor
    score_fn = LAC() if best_result['score'] == 'LAC' else APS()
    if best_result['predictor'] == 'Split':
        predictor = SplitPredictor(score_fn, model)
    elif best_result['predictor'] == 'Class-Conditional':
        predictor = ClassConditionalPredictor(score_fn, model)
    else:
        predictor = RC3PPredictor(score_fn, model)

    predictor.calibrate(cal_loader, alpha=args.alpha)

    # Select random samples
    sample_indices = np.random.choice(
        len(eval_dataset),
        min(args.n_samples, len(eval_dataset)),
        replace=False
    )

    visualized_count = 0
    for i, idx in enumerate(tqdm(sample_indices, desc="Processing samples")):
        image, true_label = eval_dataset[idx]

        # Get prediction set
        with torch.no_grad():
            pred_mask = predictor.predict(image.unsqueeze(0).to(device))
            prediction_set = torch.nonzero(pred_mask[0]).flatten().cpu().numpy()

        # Skip if only one class (no uncertainty)
        if len(prediction_set) <= 1:
            continue

        # Visualize with explanations
        save_path = results_dir / f'sample_{visualized_count+1:03d}.png'
        visualize_sample_with_explanations(
            image=image,
            true_label=true_label,
            prediction_set=prediction_set,
            model=model,
            class_names=class_names,
            device=device,
            save_path=save_path
        )

        visualized_count += 1
        if visualized_count >= args.n_samples:
            break

    print("\n" + "=" * 70)
    print("Testing Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {results_dir}")
    print(f"  - metrics.csv: Classification and conformal metrics")
    print(f"  - conformal_results.csv: Detailed conformal results")
    print(f"  - conformal_analysis.png: Comparison charts")
    print(f"  - {visualized_count} sample visualizations")


if __name__ == '__main__':
    main()
