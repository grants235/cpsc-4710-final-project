#!/usr/bin/env python3
"""
Generate case study visualizations for top confused class pairs.
Creates detailed SHAP and GradCAM comparisons for misclassified examples.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import torchvision.transforms as transforms
from tqdm import tqdm

from helpers import (
    CUBDataset,
    create_resnet50_model,
    explain_predictions_with_shap,
    explain_predictions_with_gradcam,
    simple_differential
)

# Constants
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 448
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

def load_class_names():
    """Load CUB-200-2011 class names."""
    with open('CUB_200_2011/classes.txt', 'r') as f:
        classes = [line.strip().split(' ')[1].split('.')[1].replace('_', ' ')
                  for line in f.readlines()]
    return classes

def denormalize(tensor, mean=MEAN, std=STD):
    """Denormalize image tensor for visualization."""
    img = tensor.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(img, 0, 1)

def find_confused_examples(model, test_loader, class_names, top_n=4):
    """Find actual misclassified examples for top confused pairs."""
    model.eval()

    # Track predictions
    all_data = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Finding confused examples")):
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)

            for i in range(len(labels)):
                all_data.append({
                    'batch_idx': batch_idx,
                    'in_batch_idx': i,
                    'true_class': labels[i].item(),
                    'pred_class': predicted[i].item()
                })

    # Read top confused pairs
    confused_df = pd.read_csv('results/top_confused_pairs.csv')

    # Find examples for top pairs
    confused_examples = {}
    for _, row in confused_df.head(top_n).iterrows():
        true_class_name = row['true_class'].replace('_', ' ')
        pred_class_name = row['pred_class'].replace('_', ' ')

        true_idx = class_names.index(true_class_name)
        pred_idx = class_names.index(pred_class_name)

        # Find misclassified examples
        examples = [d for d in all_data
                   if d['true_class'] == true_idx and d['pred_class'] == pred_idx]

        if examples:
            # Take first 2 examples
            confused_examples[f"{true_class_name}_to_{pred_class_name}"] = {
                'examples': examples[:2],
                'true_idx': true_idx,
                'pred_idx': pred_idx,
                'true_name': true_class_name,
                'pred_name': pred_class_name
            }

    return confused_examples

def create_comparison_figure(image, true_idx, pred_idx, class_names,
                            shap_results, gradcam_results,
                            shap_diff, gradcam_diff, save_path):
    """Create side-by-side comparison figure."""

    # Denormalize image for display
    image_np = denormalize(image).permute(1, 2, 0).cpu().numpy()

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    # Title
    fig.suptitle(f"{class_names[true_idx]} → {class_names[pred_idx]}",
                fontsize=16, fontweight='bold', y=0.98)

    # Row 1: GradCAM
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Original Image", fontsize=11)
    axes[0, 0].axis('off')

    im1 = axes[0, 1].imshow(gradcam_results['attributions'][true_idx], cmap='hot')
    axes[0, 1].set_title(f"GradCAM: True Class\n({class_names[true_idx]})", fontsize=11)
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[0, 2].imshow(gradcam_results['attributions'][pred_idx], cmap='hot')
    axes[0, 2].set_title(f"GradCAM: Predicted\n({class_names[pred_idx]})", fontsize=11)
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im3 = axes[0, 3].imshow(gradcam_diff[true_idx], cmap='hot')
    axes[0, 3].set_title("GradCAM Differential\n(Unique to True Class)", fontsize=11)
    axes[0, 3].axis('off')
    plt.colorbar(im3, ax=axes[0, 3], fraction=0.046, pad=0.04)

    # Row 2: SHAP
    axes[1, 0].imshow(image_np)
    axes[1, 0].set_title("Original Image", fontsize=11)
    axes[1, 0].axis('off')

    im4 = axes[1, 1].imshow(shap_results['attributions'][true_idx], cmap='hot')
    axes[1, 1].set_title(f"SHAP: True Class\n({class_names[true_idx]})", fontsize=11)
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im5 = axes[1, 2].imshow(shap_results['attributions'][pred_idx], cmap='hot')
    axes[1, 2].set_title(f"SHAP: Predicted\n({class_names[pred_idx]})", fontsize=11)
    axes[1, 2].axis('off')
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

    im6 = axes[1, 3].imshow(shap_diff[true_idx], cmap='hot')
    axes[1, 3].set_title("SHAP Differential\n(Unique to True Class)", fontsize=11)
    axes[1, 3].axis('off')
    plt.colorbar(im6, ax=axes[1, 3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def main():
    print("=" * 70)
    print("CASE STUDY VISUALIZATION GENERATOR")
    print("=" * 70)

    # Load model
    print("\n1. Loading model...")
    model = create_resnet50_model(num_classes=200, pretrained=False)
    checkpoint = torch.load('best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    # Load class names
    class_names = load_class_names()

    # Load test dataset
    print("2. Loading test dataset...")
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    test_dataset = CUBDataset(
        root_dir='CUB_200_2011',
        split='test',
        transform=test_transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    # Find confused examples
    print("3. Finding confused examples...")
    confused_examples = find_confused_examples(model, test_loader, class_names, top_n=4)

    print(f"\nFound {len(confused_examples)} confused pairs with examples")

    # Generate visualizations
    print("\n4. Generating case study visualizations...")
    for pair_name, pair_data in confused_examples.items():
        print(f"\n  Processing: {pair_data['true_name']} → {pair_data['pred_name']}")

        true_idx = pair_data['true_idx']
        pred_idx = pair_data['pred_idx']

        for ex_num, example in enumerate(pair_data['examples'], 1):
            # Get image from dataset
            global_idx = example['batch_idx'] * 32 + example['in_batch_idx']
            image, _ = test_dataset[global_idx]
            input_tensor = image.unsqueeze(0).to(DEVICE)

            # Get both true and predicted class indices
            class_indices = torch.tensor([true_idx, pred_idx], device=DEVICE)

            # Generate SHAP explanations
            shap_results = explain_predictions_with_shap(
                model, input_tensor, class_indices,
                method='gradient', n_samples=50, device=DEVICE
            )

            # Generate GradCAM explanations
            gradcam_results = explain_predictions_with_gradcam(
                model, input_tensor, class_indices,
                target_layer=model.layer4[-1], device=DEVICE
            )

            # Compute differentials
            shap_diff = simple_differential(shap_results)
            gradcam_diff = simple_differential(gradcam_results)

            # Save figure
            save_path = RESULTS_DIR / f"case_study_{pair_name}_example_{ex_num}.png"
            create_comparison_figure(
                image, true_idx, pred_idx, class_names,
                shap_results, gradcam_results,
                shap_diff, gradcam_diff, save_path
            )

    print("\n" + "=" * 70)
    print("COMPLETE! All case study visualizations saved to results/")
    print("=" * 70)

if __name__ == '__main__':
    main()
