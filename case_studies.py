#!/usr/bin/env python3
"""
CASE STUDY VISUALIZATION GENERATOR (Multi-row Version)

This script:
  1. Loads a trained ResNet-50 model from checkpoints/best_model.pth
  2. Loads the CUB-200-2011 test set
  3. Reads results/top_confused_pairs.csv to get the top confused class pairs
  4. Finds actual misclassified examples (true -> predicted) for those pairs
  5. For each example, generates a multi-row visualization:

     Row 0: Original image + text summary (true / predicted, probs)
     Row 1+: For each class in {true_class, predicted_class}:
             [ SHAP overlay | GradCAM overlay | SHAP differential | GradCAM differential ]

Figures are saved to results/case_study_<true>_to_<pred>_example_<k>.png
"""

from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from torchvision import transforms
from tqdm import tqdm

from helpers import (
    Cub2011Dataset,
    create_resnet50_model,
    explain_predictions_with_shap,
    explain_predictions_with_gradcam,
    simple_differential,
)

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 448
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def load_class_names() -> List[str]:
    """Load CUB-200-2011 class names as a list indexed by class id (0..199)."""
    with open("CUB_200_2011/classes.txt", "r") as f:
        classes = [
            line.strip().split(" ")[1].split(".")[1].replace("_", " ")
            for line in f.readlines()
        ]
    return classes


def denormalize(t: torch.Tensor, mean=MEAN, std=STD) -> torch.Tensor:
    """Denormalize image tensor for visualization. Input: (3,H,W) or (1,3,H,W)."""
    if t.dim() == 3:
        t = t.unsqueeze(0)
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)
    out = t * std_t + mean_t
    return torch.clamp(out, 0, 1).squeeze(0)


def find_confused_examples(
    model: torch.nn.Module,
    test_loader,
    class_names: List[str],
    top_n: int = 4,
) -> Dict[str, Dict]:
    """
    Find actual misclassified examples for the top confused pairs.

    Returns:
        dict mapping pair_name -> {
            'examples': [ {batch_idx, in_batch_idx, true_class, pred_class}, ... ],
            'true_idx': int,
            'pred_idx': int,
            'true_name': str,
            'pred_name': str
        }
    """
    model.eval()
    all_data = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(
            tqdm(test_loader, desc="Finding confused examples")
        ):
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)

            for i in range(len(labels)):
                all_data.append(
                    {
                        "batch_idx": batch_idx,
                        "in_batch_idx": i,
                        "true_class": labels[i].item(),
                        "pred_class": predicted[i].item(),
                    }
                )

    # Read top confused pairs
    confused_df = pd.read_csv("results/top_confused_pairs.csv")

    confused_examples: Dict[str, Dict] = {}
    for _, row in confused_df.head(top_n).iterrows():
        true_class_name = row["true_class"].replace("_", " ")
        pred_class_name = row["pred_class"].replace("_", " ")

        true_idx = class_names.index(true_class_name)
        pred_idx = class_names.index(pred_class_name)

        # Misclassified examples for this pair
        examples = [
            d
            for d in all_data
            if d["true_class"] == true_idx and d["pred_class"] == pred_idx
        ]

        if examples:
            confused_examples[f"{true_class_name}_to_{pred_class_name}"] = {
                "examples": examples[:2],  # take first 2
                "true_idx": true_idx,
                "pred_idx": pred_idx,
                "true_name": true_class_name,
                "pred_name": pred_class_name,
            }

    return confused_examples


# ----------------------------------------------------------------------
# Visualization for a single confused example
# ----------------------------------------------------------------------

def visualize_confused_example_multirow(
    image: torch.Tensor,
    true_idx: int,
    pred_idx: int,
    model: torch.nn.Module,
    class_names: List[str],
    device: torch.device,
    save_path: Path,
):
    """
    Create a multi-row visualization for a single misclassified example.

    Layout:
      Row 0: Original image + text (true / predicted / probabilities)
      Row 1+: For each class in [true_idx, pred_idx]:
              [ SHAP | GradCAM | SHAP Differential | GradCAM Differential ]
    """
    model.eval()

    # Get model predictions
    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
        probs = F.softmax(logits, dim=1)[0]

    # Denormalize image
    img_for_viz = denormalize(image)
    img_np = img_for_viz.permute(1, 2, 0).cpu().numpy()

    # Classes to explain: true and predicted
    classes_to_explain = [true_idx, pred_idx]
    n_classes = len(classes_to_explain)

    # Create figure: rows = n_classes + 1, cols = 4
    fig, axes = plt.subplots(n_classes + 1, 4, figsize=(16, 4 * (n_classes + 1)))
    if n_classes + 1 == 1:
        axes = axes.reshape(1, -1)

    # ------------------------------------------------------------------
    # Row 0: Original image + summary text
    # ------------------------------------------------------------------
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    true_name = class_names[true_idx]
    pred_name = class_names[pred_idx]

    pred_text = f"True:      {true_name}\nPredicted: {pred_name}\n\n"
    pred_text += "Shown classes:\n"
    for i, cls_idx in enumerate(classes_to_explain):
        prob = probs[cls_idx].item()
        pred_text += f"{i+1}. {class_names[cls_idx]}: {prob:.3f}\n"

    axes[0, 1].text(
        0.05,
        0.5,
        pred_text,
        fontsize=10,
        va="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    axes[0, 1].axis("off")
    axes[0, 2].axis("off")
    axes[0, 3].axis("off")

    # ------------------------------------------------------------------
    # Explanations for the two classes (SHAP, GradCAM, differentials)
    # ------------------------------------------------------------------
    class_indices_tensor = torch.tensor(classes_to_explain, device=device)

    # SHAP explanations
    shap_results = explain_predictions_with_shap(
        model=model,
        input_tensor=image,
        class_indices=class_indices_tensor,
        method="gradient",
        n_samples=50,
        device=device,
        denormalize=False,
    )

    # GradCAM explanations
    gradcam_results = explain_predictions_with_gradcam(
        model=model,
        input_tensor=image,
        class_indices=class_indices_tensor,
        device=device,
        denormalize=False,
    )

    # Differential explanations (use your helper)
    shap_differential = simple_differential(shap_results, brightness=3.0)
    gradcam_differential = simple_differential(gradcam_results, brightness=3.0)

    # One row per class
    for row_idx, cls_idx in enumerate(classes_to_explain, start=1):
        cls_name = class_names[cls_idx]
        prob = probs[cls_idx].item()

        # ---------------- Column 0: SHAP overlay ----------------
        shap_attr = shap_results["attributions"][cls_idx].squeeze().numpy()
        # (C,H,W) → sum over channels, normalize, smooth
        shap_heatmap = np.abs(shap_attr).sum(axis=0)
        shap_heatmap = (shap_heatmap - shap_heatmap.min()) / (
            shap_heatmap.max() - shap_heatmap.min() + 1e-10
        )
        shap_smooth = gaussian_filter(shap_heatmap, sigma=2)

        axes[row_idx, 0].imshow(img_np)
        im0 = axes[row_idx, 0].imshow(shap_smooth, cmap="jet", alpha=0.6)
        axes[row_idx, 0].set_title(
            f"SHAP\n{cls_name} ({prob:.3f})", fontsize=10, fontweight="bold"
        )
        axes[row_idx, 0].axis("off")
        plt.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)

        # ---------------- Column 1: GradCAM overlay ----------------
        gradcam_attr = gradcam_results["attributions"][cls_idx].squeeze().numpy()
        # (C,H,W) → mean over channels, normalize, smooth
        gradcam_heatmap = np.mean(gradcam_attr, axis=0)
        gradcam_heatmap = (gradcam_heatmap - gradcam_heatmap.min()) / (
            gradcam_heatmap.max() - gradcam_heatmap.min() + 1e-10
        )
        gradcam_smooth = gaussian_filter(gradcam_heatmap, sigma=2)

        axes[row_idx, 1].imshow(img_np)
        im1 = axes[row_idx, 1].imshow(gradcam_smooth, cmap="jet", alpha=0.6)
        axes[row_idx, 1].set_title(
            f"GradCAM\n{cls_name} ({prob:.3f})", fontsize=10, fontweight="bold"
        )
        axes[row_idx, 1].axis("off")
        plt.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

        # ---------------- Column 2: SHAP differential ----------------
        shap_diff = shap_differential[cls_idx]
        shap_diff_smooth = gaussian_filter(shap_diff, sigma=2)

        axes[row_idx, 2].imshow(img_np)
        im2 = axes[row_idx, 2].imshow(shap_diff_smooth, cmap="jet", alpha=0.6)
        axes[row_idx, 2].set_title(
            "SHAP Differential\n(Unique Features)", fontsize=10, fontweight="bold"
        )
        axes[row_idx, 2].axis("off")
        plt.colorbar(im2, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)

        # ---------------- Column 3: GradCAM differential ----------------
        gradcam_diff = gradcam_differential[cls_idx]
        gradcam_diff_smooth = gaussian_filter(gradcam_diff, sigma=2)

        axes[row_idx, 3].imshow(img_np)
        im3 = axes[row_idx, 3].imshow(gradcam_diff_smooth, cmap="jet", alpha=0.6)
        axes[row_idx, 3].set_title(
            "GradCAM Differential\n(Unique Features)", fontsize=10, fontweight="bold"
        )
        axes[row_idx, 3].axis("off")
        plt.colorbar(im3, ax=axes[row_idx, 3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("CASE STUDY VISUALIZATION GENERATOR (Multi-row)")
    print("=" * 70)

    # --------------------------------------------------------------
    # 1. Load model
    # --------------------------------------------------------------
    print("\n1. Loading model...")
    model = create_resnet50_model(num_classes=200, pretrained=False)
    checkpoint_path = "checkpoints/best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")

    # --------------------------------------------------------------
    # 2. Load class names and test dataset
    # --------------------------------------------------------------
    print("\n2. Loading test dataset...")
    class_names = load_class_names()

    test_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )

    test_dataset = Cub2011Dataset(
        root_dir="CUB_200_2011",
        train=False,
        transform=test_transform,
        use_bbox=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
    )

    # --------------------------------------------------------------
    # 3. Find confused examples
    # --------------------------------------------------------------
    print("\n3. Finding confused examples...")
    confused_examples = find_confused_examples(
        model, test_loader, class_names, top_n=4
    )
    print(f"\nFound {len(confused_examples)} confused pairs with examples")

    # --------------------------------------------------------------
    # 4. Generate visualizations
    # --------------------------------------------------------------
    print("\n4. Generating case study visualizations...")

    for pair_name, pair_data in confused_examples.items():
        true_idx = pair_data["true_idx"]
        pred_idx = pair_data["pred_idx"]
        true_name = pair_data["true_name"]
        pred_name = pair_data["pred_name"]

        print(f"\n  Processing pair: {true_name} → {pred_name}")

        for ex_num, example in enumerate(pair_data["examples"], start=1):
            global_idx = example["batch_idx"] * 32 + example["in_batch_idx"]
            image, _ = test_dataset[global_idx]

            save_path = RESULTS_DIR / f"case_study_{pair_name}_example_{ex_num}.png"

            visualize_confused_example_multirow(
                image=image,
                true_idx=true_idx,
                pred_idx=pred_idx,
                model=model,
                class_names=class_names,
                device=DEVICE,
                save_path=save_path,
            )

    print("\n" + "=" * 70)
    print("COMPLETE! All case study visualizations saved to results/")
    print("=" * 70)


if __name__ == "__main__":
    main()
