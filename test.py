#!/usr/bin/env python3
"""
Test Script for Fine-tuned Model with Explainability

This script loads the fine-tuned ResNet-50 model and demonstrates:
1. Conformal prediction sets
2. SHAP explainability
3. GradCAM explainability
4. Differential explainability

Results are saved to the results/ folder.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# TorchCP for conformal prediction
try:
    from torchcp.classification.score import APS
    from torchcp.classification.predictor import SplitPredictor
    TORCHCP_AVAILABLE = True
except ImportError:
    print("Warning: torchcp not available. Install with: pip install torchcp")
    TORCHCP_AVAILABLE = False

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


def setup_conformal_predictor(model, cal_loader, alpha=0.05, device='cuda'):
    """Set up and calibrate conformal predictor."""
    if not TORCHCP_AVAILABLE:
        print("Warning: Conformal prediction not available without torchcp")
        return None

    model = model.to(device)
    model.eval()

    score_fn = APS()
    predictor = SplitPredictor(score_fn, model)

    print(f"Calibrating conformal predictor (alpha={alpha})...")
    predictor.calibrate(cal_loader, alpha=alpha)

    return predictor


def get_prediction_set(predictor, image, device='cuda'):
    """Get conformal prediction set for an image."""
    if predictor is None:
        # Fallback: return top-3 predictions
        with torch.no_grad():
            logits = predictor.model(image.unsqueeze(0).to(device))
            probs = F.softmax(logits, dim=1)[0]
            top_k = torch.topk(probs, k=min(3, len(probs)))
            return top_k.indices.cpu().numpy()

    with torch.no_grad():
        pred_mask = predictor.predict(image.unsqueeze(0).to(device))
        prediction_set = torch.nonzero(pred_mask[0]).flatten().cpu().numpy()

    return prediction_set


def visualize_sample_with_explanations(
    image,
    true_label,
    prediction_set,
    model,
    class_names,
    device='cuda',
    save_path=None
):
    """
    Visualize a sample with model predictions and explainability methods.

    Shows:
    - Original image
    - Model predictions with confidence
    - SHAP explanations
    - GradCAM explanations
    - Differential explanations
    """
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
    pred_text = f"True Label: {class_names[true_label]}\n\n"
    pred_text += f"Prediction Set (size={len(prediction_set)}):\n"
    for i, cls_idx in enumerate(classes_to_explain):
        prob = probs[cls_idx].item()
        pred_text += f"{i+1}. {class_names[cls_idx]}: {prob:.3f}\n"

    axes[0, 1].text(0.1, 0.5, pred_text, fontsize=10, va='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[0, 1].axis('off')
    axes[0, 2].axis('off')
    axes[0, 3].axis('off')

    # Generate explanations for each class
    print(f"Generating explanations for {n_classes} classes...")

    class_indices_tensor = torch.tensor(classes_to_explain, device=device)

    # SHAP explanations
    print("  Computing SHAP...")
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
    print("  Computing GradCAM...")
    gradcam_results = explain_predictions_with_gradcam(
        model=model,
        input_tensor=image,
        class_indices=class_indices_tensor,
        device=device,
        denormalize=False
    )

    # Differential explanations
    print("  Computing differential...")
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
        print(f"Saved visualization to {save_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Test fine-tuned model with explainability visualization'
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
                        help='Batch size for calibration (default: 32)')

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

    # We only need test data
    _, test_loader = create_dataloaders(
        root_dir=args.data_dir,
        train_transform=test_transform,  # Not used
        test_transform=test_transform,
        batch_size=args.batch_size,
        use_bbox=True
    )

    test_dataset = test_loader.dataset
    class_names = test_dataset.class_names

    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of classes: {len(class_names)}")

    # Split test set for calibration
    cal_size = int(len(test_dataset) * 0.5)
    eval_size = len(test_dataset) - cal_size

    cal_dataset, eval_dataset = random_split(
        test_dataset,
        [cal_size, eval_size],
        generator=torch.Generator().manual_seed(42)
    )

    cal_loader = DataLoader(cal_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Calibration set size: {len(cal_dataset)}")
    print(f"Evaluation set size: {len(eval_dataset)}")

    # ========================================================================
    # Load Model
    # ========================================================================

    print("\n" + "=" * 70)
    print("Loading Model")
    print("=" * 70)

    # Create model
    model = create_resnet50_model(num_classes=200, pretrained=False)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    if 'test_acc' in checkpoint:
        print(f"Checkpoint test accuracy: {checkpoint['test_acc']*100:.2f}%")

    # ========================================================================
    # Setup Conformal Predictor
    # ========================================================================

    print("\n" + "=" * 70)
    print("Setting up Conformal Predictor")
    print("=" * 70)

    predictor = setup_conformal_predictor(
        model=model,
        cal_loader=cal_loader,
        alpha=args.alpha,
        device=device
    )

    if predictor:
        print(f"Conformal predictor calibrated!")
        print(f"Target coverage: {1-args.alpha:.1%}")

    # ========================================================================
    # Test and Visualize Samples
    # ========================================================================

    print("\n" + "=" * 70)
    print(f"Testing and Visualizing {args.n_samples} Samples")
    print("=" * 70)

    # Select random samples
    sample_indices = np.random.choice(
        len(eval_dataset),
        min(args.n_samples, len(eval_dataset)),
        replace=False
    )

    for i, idx in enumerate(tqdm(sample_indices, desc="Processing samples")):
        print(f"\nSample {i+1}/{len(sample_indices)} (index={idx})")

        image, true_label = eval_dataset[idx]

        # Get prediction set
        prediction_set = get_prediction_set(predictor, image, device)

        print(f"  True label: {class_names[true_label]}")
        print(f"  Prediction set size: {len(prediction_set)}")

        # Skip if only one class (no uncertainty)
        if len(prediction_set) <= 1:
            print("  Skipping (no uncertainty, single prediction)")
            continue

        # Visualize with explanations
        save_path = results_dir / f'sample_{i+1:03d}_idx{idx}.png'
        visualize_sample_with_explanations(
            image=image,
            true_label=true_label,
            prediction_set=prediction_set,
            model=model,
            class_names=class_names,
            device=device,
            save_path=save_path
        )

    print("\n" + "=" * 70)
    print("Testing Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {results_dir}")
    print(f"Generated {len(list(results_dir.glob('*.png')))} visualizations")


if __name__ == '__main__':
    main()
