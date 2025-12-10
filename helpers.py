#!/usr/bin/env python3
"""
Helper Functions and Classes for Explainability Testing

This module consolidates all helper functions, classes, and utilities used for
evaluating explainability methods on conformal prediction sets.

Sections:
    1. Dataset Utilities
    2. Explainability Methods (SHAP, GradCAM)
    3. Evaluation Classes
    4. Visualization Functions
    5. Analysis Functions
"""

import os
import warnings
from typing import Optional, List, Tuple, Dict, Callable
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

from PIL import Image
from scipy import stats
from scipy.ndimage import gaussian_filter, label, find_objects
from scipy.spatial.distance import cosine, pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import jaccard_score
from sklearn.decomposition import PCA
from tqdm import tqdm

from captum.attr import (
    KernelShap,
    GradientShap,
    IntegratedGradients,
    GuidedBackprop,
    Saliency,
    LayerGradCam
)
from captum.attr import visualization as viz

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ==============================================================================
# 1. DATASET UTILITIES
# ==============================================================================

class Cub2011Dataset(Dataset):
    """Dataset class for CUB-200-2011 bird species classification."""

    def __init__(
        self,
        root_dir: str,
        transform=None,
        train: bool = True,
        use_bbox: bool = True
    ):
        """
        Initialize CUB-200-2011 dataset.

        Args:
            root_dir: Directory with all the CUB_200_2011 data
            transform: Optional transform to be applied on a sample
            train: If True, creates training set, else creates test set
            use_bbox: If True, crops the image to the bounding box before transforms
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.use_bbox = use_bbox

        # Read images.txt (file_id, file_path)
        images_df = pd.read_csv(
            os.path.join(self.root_dir, 'images.txt'),
            sep=' ', names=['img_id', 'img_path']
        )

        # Read image_class_labels.txt (file_id, class_id)
        labels_df = pd.read_csv(
            os.path.join(self.root_dir, 'image_class_labels.txt'),
            sep=' ', names=['img_id', 'class_id']
        )

        # Read train_test_split.txt (file_id, is_train)
        split_df = pd.read_csv(
            os.path.join(self.root_dir, 'train_test_split.txt'),
            sep=' ', names=['img_id', 'is_train']
        )

        # Read classes.txt (class_id, class_name)
        classes_df = pd.read_csv(
            os.path.join(self.root_dir, 'classes.txt'),
            sep=' ', names=['class_id', 'class_name']
        )

        # Read bounding_boxes.txt
        bboxes_df = pd.read_csv(
            os.path.join(self.root_dir, 'bounding_boxes.txt'),
            sep=' ', names=['img_id', 'x', 'y', 'width', 'height']
        )

        # Create mapping from class_id to class_name
        self.class_names = {
            row.class_id - 1: row.class_name.split('.')[-1]
            for row in classes_df.itertuples()
        }

        # Merge dataframes
        data_df = images_df.merge(labels_df, on='img_id')
        data_df = data_df.merge(split_df, on='img_id')
        data_df = data_df.merge(bboxes_df, on='img_id')

        # Filter for train or test
        if self.train:
            self.data = data_df[data_df['is_train'] == 1]
        else:
            self.data = data_df[data_df['is_train'] == 0]

        # Create samples list: (full_image_path, class_label_index, bbox)
        self.samples = []
        for row in self.data.itertuples():
            img_path = os.path.join(self.root_dir, 'images', row.img_path)
            class_label_index = row.class_id - 1  # Convert to 0-indexed
            bbox = (row.x, row.y, row.width, row.height)
            self.samples.append((img_path, class_label_index, bbox))

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Fetch the sample at the given index.

        Returns:
            tuple: (image, label) where image is the transformed image
                   and label is the 0-indexed integer class label
        """
        img_path, label, bbox = self.samples[idx]

        # Load image and convert to RGB
        image = Image.open(img_path).convert('RGB')

        # Apply bounding box crop if requested
        if self.use_bbox:
            left = int(bbox[0])
            upper = int(bbox[1])
            right = int(bbox[0] + bbox[2])
            lower = int(bbox[1] + bbox[3])
            crop_box = (left, upper, right, lower)
            image = image.crop(crop_box)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label


def create_dataloaders(
    root_dir: str,
    train_transform,
    test_transform,
    batch_size: int = 32,
    use_bbox: bool = True,
    train_loader_kwargs: Optional[Dict] = None,
    test_loader_kwargs: Optional[Dict] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create and return training and testing DataLoaders for CUB-200-2011.

    Args:
        root_dir: Path to the CUB_200_2011 dataset root directory
        train_transform: Transformations to apply to the training set
        test_transform: Transformations to apply to the test set
        batch_size: Number of samples per batch
        use_bbox: Whether to crop images to their bounding boxes
        train_loader_kwargs: Additional keyword arguments for training DataLoader
        test_loader_kwargs: Additional keyword arguments for test DataLoader

    Returns:
        tuple: (train_loader, test_loader)
    """
    if train_loader_kwargs is None:
        train_loader_kwargs = {}
    if test_loader_kwargs is None:
        test_loader_kwargs = {}

    # Create datasets
    train_dataset = Cub2011Dataset(
        root_dir=root_dir,
        transform=train_transform,
        train=True,
        use_bbox=use_bbox
    )

    test_dataset = Cub2011Dataset(
        root_dir=root_dir,
        transform=test_transform,
        train=False,
        use_bbox=use_bbox
    )

    # Create dataloaders
    train_shuffle = train_loader_kwargs.pop('shuffle', True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        **train_loader_kwargs
    )

    test_shuffle = test_loader_kwargs.pop('shuffle', False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=test_shuffle,
        **test_loader_kwargs
    )

    return train_loader, test_loader


# ==============================================================================
# 2. EXPLAINABILITY METHODS (SHAP, GradCAM)
# ==============================================================================

def explain_predictions_with_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_indices: torch.Tensor,
    target_layer=None,
    device: str = 'cuda',
    denormalize: bool = True,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> Dict:
    """
    Generate GradCAM attributions using Captum.

    Args:
        model: The trained PyTorch model
        input_tensor: Input image (3, H, W) or (1, 3, H, W)
        class_indices: Tensor of class indices to explain
        target_layer: Which layer (default: model.layer4[-1] for ResNet)
        device: 'cuda' or 'cpu'
        denormalize: Whether to denormalize the image for visualization
        mean: Mean values used for normalization
        std: Standard deviation values used for normalization

    Returns:
        Dictionary containing attributions, predictions, and input image
    """
    # Setup
    model.eval()
    model = model.to(device)

    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad = True
    class_indices = class_indices.to(device)

    # Default layer for ResNet
    if target_layer is None:
        target_layer = model.layer4[-1]

    # Get predictions
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)

    _, _, H, W = input_tensor.shape

    # Create GradCAM explainer
    gradcam = LayerGradCam(model, target_layer)

    attributions_dict = {}
    predictions_dict = {}

    print(f"Computing GradCAM for {len(class_indices)} classes...")

    # Generate CAM for each class
    for class_idx in class_indices:
        class_idx_item = class_idx.item()

        # Get prediction score
        pred_score = probabilities[0, class_idx_item].item()
        predictions_dict[class_idx_item] = pred_score

        # Compute GradCAM using Captum
        attribution = gradcam.attribute(
            input_tensor,
            target=class_idx_item,
            relu_attributions=True
        )

        # Upsample to input image size
        attribution_upsampled = F.interpolate(
            attribution,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        # Normalize to [0, 1]
        attr_np = attribution_upsampled.squeeze().detach().cpu().numpy()
        if attr_np.max() > 0:
            attr_np = attr_np / attr_np.max()

        # Convert to tensor format [1, 3, H, W] to match SHAP
        cam_tensor = torch.from_numpy(attr_np).float()
        cam_tensor = cam_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

        attributions_dict[class_idx_item] = cam_tensor

        print(f"  Class {class_idx_item}: Score = {pred_score:.4f}")

    # Denormalize input
    input_for_viz = input_tensor.cpu().detach()
    if denormalize:
        mean_tensor = torch.tensor(mean).view(1, 3, 1, 1)
        std_tensor = torch.tensor(std).view(1, 3, 1, 1)
        input_for_viz = input_for_viz * std_tensor + mean_tensor
        input_for_viz = torch.clamp(input_for_viz, 0, 1)

    return {
        'attributions': attributions_dict,
        'predictions': predictions_dict,
        'input_image': input_for_viz,
        'class_indices': class_indices.cpu()
    }


def explain_predictions_with_shap(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_indices: torch.Tensor,
    method: str = 'gradient',
    n_samples: int = 50,
    baseline: Optional[torch.Tensor] = None,
    device: str = 'cuda',
    denormalize: bool = True,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> Dict:
    """
    Compute SHAP attributions for each class using Captum.

    Args:
        model: The trained PyTorch model
        input_tensor: Input image tensor of shape (1, 3, H, W) or (3, H, W)
        class_indices: Tensor of class indices to explain
        method: Type of SHAP ('gradient', 'kernel', or 'integrated_gradients')
        n_samples: Number of samples for KernelShap or GradientShap
        baseline: Baseline for comparison (if None, uses black image)
        device: Device to run computations on
        denormalize: Whether to denormalize the image for visualization
        mean: Mean values used for normalization
        std: Standard deviation values used for normalization

    Returns:
        Dictionary containing attributions, predictions, and input image
    """
    # Setup
    model.eval()
    model = model.to(device)

    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)
    class_indices = class_indices.to(device)

    # Create baseline if not provided
    if baseline is None:
        baseline = torch.zeros_like(input_tensor).to(device)
    else:
        baseline = baseline.to(device)
        if baseline.dim() == 3:
            baseline = baseline.unsqueeze(0)

    # Get predictions for all classes
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)

    # Initialize the SHAP method
    if method.lower() == 'gradient':
        explainer = GradientShap(model)
    elif method.lower() == 'kernel':
        explainer = KernelShap(model)
    elif method.lower() == 'integrated_gradients':
        explainer = IntegratedGradients(model)
    else:
        raise ValueError(
            f"Unknown method: {method}. Choose 'gradient', 'kernel', or 'integrated_gradients'"
        )

    # Compute attributions for each class
    attributions_dict = {}
    predictions_dict = {}

    print(f"Computing {method} attributions for {len(class_indices)} classes...")

    for class_idx in class_indices:
        class_idx_item = class_idx.item()

        # Get prediction score
        pred_score = probabilities[0, class_idx_item].item()
        predictions_dict[class_idx_item] = pred_score

        # Compute attribution
        if method.lower() == 'gradient':
            baselines = baseline.repeat(n_samples, 1, 1, 1)
            baselines = baselines + torch.randn_like(baselines) * 0.1

            attribution = explainer.attribute(
                input_tensor,
                baselines=baselines,
                target=class_idx_item,
                n_samples=n_samples
            )
        elif method.lower() == 'kernel':
            attribution = explainer.attribute(
                input_tensor,
                target=class_idx_item,
                n_samples=n_samples,
                baselines=baseline
            )
        elif method.lower() == 'integrated_gradients':
            attribution = explainer.attribute(
                input_tensor,
                baselines=baseline,
                target=class_idx_item,
                n_steps=n_samples
            )

        attributions_dict[class_idx_item] = attribution.cpu().detach()

        print(f"  Class {class_idx_item}: Score = {pred_score:.4f}")

    # Denormalize input image for visualization
    input_for_viz = input_tensor.cpu().detach()
    if denormalize:
        mean_tensor = torch.tensor(mean).view(1, 3, 1, 1)
        std_tensor = torch.tensor(std).view(1, 3, 1, 1)
        input_for_viz = input_for_viz * std_tensor + mean_tensor
        input_for_viz = torch.clamp(input_for_viz, 0, 1)

    return {
        'attributions': attributions_dict,
        'predictions': predictions_dict,
        'input_image': input_for_viz,
        'class_indices': class_indices.cpu()
    }


def simple_differential(results: Dict, brightness: float = 3.0) -> Dict:
    """
    Compute simple differential: subtract others, clip negatives, renormalize.

    Args:
        results: Dictionary containing attributions from SHAP or GradCAM
        brightness: Brightness multiplier for the differential maps

    Returns:
        Dictionary mapping class_idx to differential heatmap
    """
    attributions = results['attributions']
    differential = {}

    for class_idx, attr in attributions.items():
        my_heatmap = np.abs(attr.squeeze().numpy()).sum(axis=0)

        other_heatmaps = []
        for other_idx, other_attr in attributions.items():
            if other_idx != class_idx:
                other_heatmap = np.abs(other_attr.squeeze().numpy()).sum(axis=0)
                other_heatmaps.append(other_heatmap)

        avg_others = np.mean(other_heatmaps, axis=0)
        diff = my_heatmap - avg_others
        diff = np.maximum(diff, 0)

        if diff.max() > 0:
            diff = diff / diff.max()

        diff = np.minimum(diff * brightness, 1.0)
        differential[class_idx] = diff

    return differential


def visualize_all(
    results: Dict,
    differential: Dict,
    class_names: Dict,
    alpha: float = 0.7
):
    """
    Visualize: [Original] | [Heatmap] | [Overlay] | [Differential].

    Args:
        results: Dictionary containing attributions and input image
        differential: Dictionary containing differential heatmaps
        class_names: Dictionary mapping class indices to names
        alpha: Alpha value for overlay transparency
    """
    input_np = results['input_image'].squeeze().permute(1, 2, 0).numpy()

    for class_idx in results['attributions'].keys():
        # Get attribution heatmap
        attr = results['attributions'][class_idx].squeeze().numpy()
        attr = np.abs(attr).sum(axis=0)
        attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-10)
        attr_smooth = gaussian_filter(attr, sigma=2)

        # Get differential heatmap
        diff_smooth = gaussian_filter(differential[class_idx], sigma=2)

        # Get class name
        pred_score = results['predictions'][class_idx]
        if class_names and class_idx in class_names:
            name = class_names[class_idx].replace('_', ' ')
            name = ' '.join(word.capitalize() for word in name.split())
            title = f"{name} (Score: {pred_score:.3f})"
        else:
            title = f"Class {class_idx} (Score: {pred_score:.3f})"

        # Create figure with 4 panels
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # 1. Original image
        axes[0].imshow(input_np)
        axes[0].set_title('Original', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # 2. Heatmap only
        im1 = axes[1].imshow(attr_smooth, cmap='jet')
        axes[1].set_title('Attribution Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # 3. Overlay
        axes[2].imshow(input_np)
        im2 = axes[2].imshow(attr_smooth, cmap='jet', alpha=alpha)
        axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        # 4. Differential (unique features)
        axes[3].imshow(input_np)
        im3 = axes[3].imshow(diff_smooth, cmap='jet', alpha=alpha)
        axes[3].set_title('Unique Features', fontsize=12, fontweight='bold')
        axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()


# ==============================================================================
# 3. EVALUATION CLASSES
# ==============================================================================

class ConformalExplainabilityEvaluator:
    """
    Evaluator for XAI methods designed for conformal prediction sets.
    Includes metrics for uncertainty explanation quality and differential analysis.
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize the evaluator.

        Args:
            model: Trained model to evaluate
            device: Device to run computations on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def get_prediction_set_explanations(
        self,
        image: torch.Tensor,
        prediction_set: List[int],
        explain_fn: Callable
    ) -> Dict:
        """
        Generate explanations for each class in the conformal prediction set.

        Args:
            image: Input image tensor
            prediction_set: List of class indices in the conformal prediction set
            explain_fn: Function that generates attribution maps

        Returns:
            Dictionary mapping class_idx to attribution_map
        """
        explanations = {}
        for class_idx in prediction_set:
            attr_map = explain_fn(image, class_idx)
            explanations[class_idx] = attr_map
        return explanations

    def compute_differential_heatmap(
        self,
        explanations: Dict,
        target_class: Optional[int] = None,
        normalize: bool = True
    ) -> Dict:
        """
        Compute differential heatmaps showing unique features for each class.

        Args:
            explanations: Dictionary of {class_idx: attribution_map}
            target_class: Specific class to compute differential for (None for all)
            normalize: Whether to normalize the differential maps

        Returns:
            Dictionary of differential heatmaps for each class
        """
        differential_maps = {}

        all_maps = list(explanations.values())
        avg_attribution = torch.stack(all_maps).mean(dim=0)

        classes_to_process = [target_class] if target_class else explanations.keys()

        for class_idx in classes_to_process:
            # Differential = class-specific - average of others
            other_attrs = [
                attr for idx, attr in explanations.items()
                if idx != class_idx
            ]
            if other_attrs:
                avg_others = torch.stack(other_attrs).mean(dim=0)
                differential = explanations[class_idx] - avg_others
            else:
                differential = explanations[class_idx] - avg_attribution

            if normalize:
                diff_min = differential.min()
                diff_max = differential.max()
                if diff_max > diff_min:
                    differential = (differential - diff_min) / (diff_max - diff_min)

            differential_maps[class_idx] = differential

        return differential_maps

    def uncertainty_focus_score(self, differential_maps: Dict) -> float:
        """
        Measure how focused/dispersed the differential explanations are.

        Higher score = more focused on specific features (aleatoric uncertainty)
        Lower score = more dispersed (epistemic uncertainty)

        Args:
            differential_maps: Dictionary of differential heatmaps

        Returns:
            Average focus score across all classes
        """
        focus_scores = []

        for diff_map in differential_maps.values():
            # Compute entropy as a measure of dispersion
            flat_map = diff_map.flatten()
            flat_map = flat_map[flat_map > 0]

            if len(flat_map) > 0:
                # Normalize to probability distribution
                prob_dist = flat_map / flat_map.sum()
                # Compute entropy (lower = more focused)
                entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-10))

                # Convert to focus score (inverse of entropy)
                focus = 1.0 / (1.0 + entropy.item())
                focus_scores.append(focus)

        return np.mean(focus_scores) if focus_scores else 0.0

    def inter_class_explanation_similarity(self, explanations: Dict) -> float:
        """
        Compute pairwise similarities between explanations for different classes.

        High similarity suggests model confusion between classes.

        Args:
            explanations: Dictionary of {class_idx: attribution_map}

        Returns:
            Average similarity score
        """
        class_indices = list(explanations.keys())
        if len(class_indices) < 2:
            return 0.0

        similarities = []
        for i in range(len(class_indices)):
            for j in range(i + 1, len(class_indices)):
                attr1 = explanations[class_indices[i]].flatten().cpu().numpy()
                attr2 = explanations[class_indices[j]].flatten().cpu().numpy()

                # Cosine similarity
                sim = 1 - cosine(attr1, attr2)
                similarities.append(sim)

        return np.mean(similarities)

    def spatial_consistency_score(
        self,
        shap_attr: torch.Tensor,
        gradcam_attr: torch.Tensor,
        threshold: float = 0.5
    ) -> float:
        """
        Measure spatial consistency between SHAP and GradCAM explanations.

        Uses IoU of top-k% regions.

        Args:
            shap_attr: SHAP attribution map
            gradcam_attr: GradCAM attribution map
            threshold: Quantile threshold for binarization

        Returns:
            IoU score between the two maps
        """
        # Threshold both maps to get binary masks
        shap_thresh = torch.quantile(shap_attr.flatten(), threshold)
        gradcam_thresh = torch.quantile(gradcam_attr.flatten(), threshold)

        shap_mask = (shap_attr > shap_thresh).float()
        gradcam_mask = (gradcam_attr > gradcam_thresh).float()

        # Compute IoU
        intersection = (shap_mask * gradcam_mask).sum()
        union = torch.maximum(shap_mask, gradcam_mask).sum()

        iou = (intersection / (union + 1e-10)).item()
        return iou

    def uncertainty_type_classification(
        self,
        explanations: Dict,
        differential_maps: Dict,
        focus_threshold: float = 0.5,
        similarity_threshold: float = 0.7
    ) -> str:
        """
        Classify uncertainty type based on explanation patterns.

        Args:
            explanations: Dictionary of attributions
            differential_maps: Dictionary of differential heatmaps
            focus_threshold: Threshold for focus classification
            similarity_threshold: Threshold for similarity classification

        Returns:
            String: 'aleatoric', 'epistemic', or 'mixed'
        """
        focus = self.uncertainty_focus_score(differential_maps)
        similarity = self.inter_class_explanation_similarity(explanations)

        if focus > focus_threshold and similarity > similarity_threshold:
            return 'aleatoric'
        elif focus < focus_threshold and similarity < similarity_threshold:
            return 'epistemic'
        else:
            return 'mixed'


class DifferentialHeatmapAnalyzer:
    """Advanced analysis of differential heatmaps for uncertainty explanation."""

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize the analyzer.

        Args:
            model: Trained model to analyze
            device: Device to run computations on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def compute_feature_importance_regions(
        self,
        heatmap: torch.Tensor,
        n_regions: int = 5
    ) -> List[Dict]:
        """
        Identify the most important regions in a heatmap.

        Args:
            heatmap: Attribution heatmap tensor
            n_regions: Number of top regions to return

        Returns:
            List of dictionaries containing region information
        """
        # Threshold to get important regions
        threshold = torch.quantile(heatmap.flatten(), 0.8)
        binary_map = (heatmap > threshold).cpu().numpy().astype(int)

        # Find connected components
        labeled_array, num_features = label(binary_map)

        regions = []
        for i in range(1, num_features + 1):
            slice_y, slice_x = find_objects(labeled_array == i)[0]
            region_map = heatmap[slice_y, slice_x]
            importance = region_map.sum().item()

            regions.append({
                'y': slice_y.start,
                'x': slice_x.start,
                'height': slice_y.stop - slice_y.start,
                'width': slice_x.stop - slice_x.start,
                'importance': importance
            })

        # Sort by importance and return top n
        regions = sorted(regions, key=lambda x: x['importance'], reverse=True)
        return regions[:n_regions]

    def analyze_feature_overlap(self, explanations_dict: Dict) -> Dict:
        """
        Analyze which features are shared vs unique across classes.

        Args:
            explanations_dict: Dictionary of {class_idx: attribution_map}

        Returns:
            Dictionary with 'shared' and 'unique' feature maps
        """
        all_maps = list(explanations_dict.values())

        # Find shared features (present in all explanations)
        shared_features = torch.stack(all_maps).min(dim=0)[0]

        # Find unique features for each class
        unique_features = {}
        for class_idx, attr_map in explanations_dict.items():
            others = [
                m for idx, m in explanations_dict.items()
                if idx != class_idx
            ]
            if others:
                max_others = torch.stack(others).max(dim=0)[0]
                unique = F.relu(attr_map - max_others)
                unique_features[class_idx] = unique
            else:
                unique_features[class_idx] = attr_map

        return {
            'shared': shared_features,
            'unique': unique_features
        }

    def compute_explanation_clustering(self, all_explanations: List[Dict]) -> Optional[Dict]:
        """
        Cluster explanations to find groups of similar uncertainty patterns.

        Args:
            all_explanations: List of dictionaries containing explanations

        Returns:
            Dictionary with clustering results or None if insufficient data
        """
        # Flatten all explanations
        flattened = []
        labels = []

        for exp_dict in all_explanations:
            for class_idx, attr_map in exp_dict.items():
                flat = attr_map.flatten().cpu().numpy()
                flattened.append(flat)
                labels.append(class_idx)

        if len(flattened) < 2:
            return None

        # Compute distance matrix
        distances = pdist(flattened, metric='cosine')

        # Perform hierarchical clustering
        linkage_matrix = linkage(distances, method='ward')

        return {
            'linkage': linkage_matrix,
            'labels': labels,
            'distances': squareform(distances)
        }

    def generate_uncertainty_signature(self, differential_maps: Dict) -> Dict:
        """
        Create a 'signature' of the uncertainty pattern.

        Args:
            differential_maps: Dictionary of differential heatmaps

        Returns:
            Dictionary with uncertainty characteristics
        """
        signatures = []

        for diff_map in differential_maps.values():
            # Spatial entropy (how spread out is the attention)
            flat = diff_map.flatten()
            flat_pos = flat[flat > 0]
            if len(flat_pos) > 0:
                prob = flat_pos / flat_pos.sum()
                entropy = -torch.sum(prob * torch.log(prob + 1e-10)).item()
            else:
                entropy = 0

            # Peak prominence (how strong are the peaks)
            peak = diff_map.max().item()
            mean = diff_map.mean().item()
            prominence = peak / (mean + 1e-10) if mean > 0 else 0

            # Coverage (what fraction of image is important)
            coverage = (diff_map > diff_map.mean()).float().mean().item()

            signatures.append({
                'entropy': entropy,
                'prominence': prominence,
                'coverage': coverage
            })

        # Aggregate signature
        avg_signature = {
            'entropy': np.mean([s['entropy'] for s in signatures]),
            'prominence': np.mean([s['prominence'] for s in signatures]),
            'coverage': np.mean([s['coverage'] for s in signatures])
        }

        # Classify uncertainty pattern
        if avg_signature['entropy'] < 2 and avg_signature['prominence'] > 5:
            pattern = 'focused_specific'
        elif avg_signature['entropy'] > 4 and avg_signature['coverage'] > 0.3:
            pattern = 'diffuse_global'
        elif avg_signature['prominence'] > 3 and avg_signature['coverage'] < 0.2:
            pattern = 'sparse_peaks'
        else:
            pattern = 'mixed'

        avg_signature['pattern'] = pattern

        return avg_signature

    def create_attention_flow_visualization(
        self,
        image: torch.Tensor,
        explanations_dict: Dict,
        class_names: Dict,
        save_path: Optional[str] = None
    ):
        """
        Visualize how attention 'flows' between different classes in prediction set.

        Args:
            image: Input image tensor
            explanations_dict: Dictionary of {class_idx: attribution_map}
            class_names: Dictionary mapping class indices to names
            save_path: Path to save the visualization

        Returns:
            Matplotlib figure
        """
        n_classes = len(explanations_dict)

        fig, axes = plt.subplots(2, n_classes + 1, figsize=(5 * (n_classes + 1), 10))

        # Display original image
        if image.shape[0] == 3:
            img_display = image.permute(1, 2, 0).cpu().numpy()
            img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
        else:
            img_display = image.squeeze().cpu().numpy()

        axes[0, 0].imshow(img_display, cmap='gray' if len(img_display.shape) == 2 else None)
        axes[0, 0].set_title('Input Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        axes[1, 0].axis('off')

        # Analyze feature overlap
        overlap_analysis = self.analyze_feature_overlap(explanations_dict)

        # Show shared features
        axes[1, 0].imshow(
            overlap_analysis['shared'].cpu().numpy(),
            cmap='Greens',
            alpha=0.7
        )
        axes[1, 0].set_title('Shared Features\n(Common to all)', fontsize=10)
        axes[1, 0].axis('off')

        # For each class
        for idx, (class_idx, attr_map) in enumerate(explanations_dict.items()):
            col = idx + 1

            # Top row: Standard attribution
            axes[0, col].imshow(
                img_display,
                alpha=0.3,
                cmap='gray' if len(img_display.shape) == 2 else None
            )
            axes[0, col].imshow(attr_map.cpu().numpy(), cmap='hot', alpha=0.7)
            axes[0, col].set_title(f'{class_names[class_idx]}', fontsize=10, fontweight='bold')
            axes[0, col].axis('off')

            # Bottom row: Unique features
            axes[1, col].imshow(
                img_display,
                alpha=0.3,
                cmap='gray' if len(img_display.shape) == 2 else None
            )
            unique_map = overlap_analysis['unique'][class_idx]
            axes[1, col].imshow(unique_map.cpu().numpy(), cmap='RdBu_r', alpha=0.7)
            axes[1, col].set_title('Unique Features', fontsize=10)
            axes[1, col].axis('off')

            # Add importance regions
            regions = self.compute_feature_importance_regions(attr_map, n_regions=3)
            for region in regions[:2]:
                rect = Rectangle(
                    (region['x'], region['y']),
                    region['width'],
                    region['height'],
                    linewidth=2,
                    edgecolor='green',
                    facecolor='none'
                )
                axes[0, col].add_patch(rect)

        plt.suptitle(
            'Attention Flow Analysis Across Conformal Prediction Set',
            fontsize=14,
            fontweight='bold'
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def compute_pairwise_explanation_distances(
        self,
        explanations_list: List[Dict],
        class_pairs: List[Tuple[int, int]]
    ) -> Dict:
        """
        Compute distances between explanations for specific class pairs.

        Args:
            explanations_list: List of explanation dictionaries
            class_pairs: List of (class1, class2) tuples

        Returns:
            Dictionary with distance statistics for each pair
        """
        distance_matrix = defaultdict(list)

        for explanations in explanations_list:
            for class1, class2 in class_pairs:
                if class1 in explanations and class2 in explanations:
                    attr1 = explanations[class1].flatten().cpu().numpy()
                    attr2 = explanations[class2].flatten().cpu().numpy()

                    # Compute various distance metrics
                    cosine_dist = 1 - np.dot(attr1, attr2) / (
                        np.linalg.norm(attr1) * np.linalg.norm(attr2)
                    )
                    l2_dist = np.linalg.norm(attr1 - attr2)

                    distance_matrix[(class1, class2)].append({
                        'cosine': cosine_dist,
                        'l2': l2_dist
                    })

        # Aggregate distances
        summary = {}
        for pair, distances in distance_matrix.items():
            summary[pair] = {
                'mean_cosine': np.mean([d['cosine'] for d in distances]),
                'std_cosine': np.std([d['cosine'] for d in distances]),
                'mean_l2': np.mean([d['l2'] for d in distances]),
                'std_l2': np.std([d['l2'] for d in distances]),
            }

        return summary


# ==============================================================================
# 4. VISUALIZATION FUNCTIONS
# ==============================================================================

def create_differential_visualization(
    image: torch.Tensor,
    explanations: Dict,
    differential_maps: Dict,
    class_names: Dict,
    save_path: Optional[str] = None
):
    """
    Create visualization showing original explanations and differential heatmaps.

    Args:
        image: Input image tensor
        explanations: Dictionary of attributions
        differential_maps: Dictionary of differential heatmaps
        class_names: Dictionary mapping class indices to names
        save_path: Path to save the visualization

    Returns:
        Matplotlib figure
    """
    n_classes = len(explanations)
    fig, axes = plt.subplots(3, n_classes, figsize=(5 * n_classes, 15))

    if n_classes == 1:
        axes = axes.reshape(-1, 1)

    # Prepare original image
    if image.shape[0] == 3:
        img_display = image.permute(1, 2, 0).cpu().numpy()
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
    else:
        img_display = image.squeeze().cpu().numpy()

    for idx, (class_idx, attr_map) in enumerate(explanations.items()):
        # Row 1: Original image
        axes[0, idx].imshow(img_display, cmap='gray' if len(img_display.shape) == 2 else None)
        axes[0, idx].set_title(f'{class_names[class_idx]}', fontsize=10, fontweight='bold')
        axes[0, idx].axis('off')

        # Row 2: Standard attribution
        im1 = axes[1, idx].imshow(attr_map.cpu().numpy(), cmap='hot', alpha=0.8)
        axes[1, idx].imshow(
            img_display,
            alpha=0.3,
            cmap='gray' if len(img_display.shape) == 2 else None
        )
        axes[1, idx].set_title('Standard Attribution', fontsize=9)
        axes[1, idx].axis('off')
        plt.colorbar(im1, ax=axes[1, idx], fraction=0.046, pad=0.04)

        # Row 3: Differential attribution
        diff_map = differential_maps[class_idx]
        im2 = axes[2, idx].imshow(diff_map.cpu().numpy(), cmap='RdBu_r', alpha=0.8)
        axes[2, idx].imshow(
            img_display,
            alpha=0.3,
            cmap='gray' if len(img_display.shape) == 2 else None
        )
        axes[2, idx].set_title('Differential Attribution\n(Unique Features)', fontsize=9)
        axes[2, idx].axis('off')
        plt.colorbar(im2, ax=axes[2, idx], fraction=0.046, pad=0.04)

    plt.suptitle(
        'Differential Heatmap Analysis for Conformal Prediction Set',
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return fig


def plot_comprehensive_evaluation(results: Dict, save_prefix: str = 'evaluation'):
    """
    Create comprehensive visualization of evaluation results.

    Args:
        results: Dictionary containing evaluation metrics
        save_prefix: Prefix for saved files

    Returns:
        Matplotlib figure
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    colors = ['#3498db', '#e74c3c']

    # 1. Deletion and Insertion AUC comparison
    ax1 = fig.add_subplot(gs[0, 0])
    if 'deletion_auc' in results['shap']:
        data = [results['shap']['deletion_auc'], results['gradcam']['deletion_auc']]
        bp = ax1.boxplot(data, labels=['SHAP', 'GradCAM'], patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax1.set_title('Deletion AUC\n(Lower is Better)', fontweight='bold')
        ax1.set_ylabel('AUC Score')
        ax1.grid(axis='y', alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    if 'insertion_auc' in results['shap']:
        data = [results['shap']['insertion_auc'], results['gradcam']['insertion_auc']]
        bp = ax2.boxplot(data, labels=['SHAP', 'GradCAM'], patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_title('Insertion AUC\n(Higher is Better)', fontweight='bold')
        ax2.set_ylabel('AUC Score')
        ax2.grid(axis='y', alpha=0.3)

    # 2. Uncertainty Focus Score
    ax3 = fig.add_subplot(gs[0, 2])
    data = [results['shap']['focus_score'], results['gradcam']['focus_score']]
    bp = ax3.boxplot(data, labels=['SHAP', 'GradCAM'], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_title('Uncertainty Focus Score\n(Higher = More Aleatoric)', fontweight='bold')
    ax3.set_ylabel('Focus Score')
    ax3.grid(axis='y', alpha=0.3)

    # 3. Inter-class Similarity
    ax4 = fig.add_subplot(gs[0, 3])
    data = [results['shap']['inter_class_similarity'], results['gradcam']['inter_class_similarity']]
    bp = ax4.boxplot(data, labels=['SHAP', 'GradCAM'], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_title('Inter-class Explanation Similarity', fontweight='bold')
    ax4.set_ylabel('Cosine Similarity')
    ax4.grid(axis='y', alpha=0.3)

    # 4. Spatial Consistency
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.hist(
        results['comparison']['spatial_consistency'],
        bins=20,
        color='#9b59b6',
        alpha=0.7,
        edgecolor='black'
    )
    ax5.set_title('SHAP-GradCAM Spatial Consistency', fontweight='bold')
    ax5.set_xlabel('IoU Score')
    ax5.set_ylabel('Frequency')
    ax5.axvline(
        np.mean(results['comparison']['spatial_consistency']),
        color='red',
        linestyle='--',
        label=f'Mean: {np.mean(results["comparison"]["spatial_consistency"]):.3f}'
    )
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)

    # 5. Uncertainty Type Classification
    ax6 = fig.add_subplot(gs[1, 1])
    shap_types = pd.Series(results['shap']['uncertainty_type']).value_counts()
    gradcam_types = pd.Series(results['gradcam']['uncertainty_type']).value_counts()

    x = np.arange(len(shap_types.index))
    width = 0.35

    ax6.bar(x - width / 2, shap_types.values, width, label='SHAP', color='#3498db', alpha=0.7)
    ax6.bar(x + width / 2, gradcam_types.values, width, label='GradCAM', color='#e74c3c', alpha=0.7)
    ax6.set_xlabel('Uncertainty Type')
    ax6.set_ylabel('Count')
    ax6.set_title('Uncertainty Type Distribution', fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(shap_types.index)
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)

    # 6. Relationship between set size and focus score
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.scatter(
        results['comparison']['set_size'],
        results['shap']['focus_score'],
        alpha=0.5,
        label='SHAP',
        color='#3498db'
    )
    ax7.scatter(
        results['comparison']['set_size'],
        results['gradcam']['focus_score'],
        alpha=0.5,
        label='GradCAM',
        color='#e74c3c'
    )
    ax7.set_xlabel('Prediction Set Size')
    ax7.set_ylabel('Focus Score')
    ax7.set_title('Set Size vs Focus Score', fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 7. Conformal coverage
    ax8 = fig.add_subplot(gs[1, 3])
    true_in_set = results['comparison']['true_in_set']
    coverage = np.mean(true_in_set)
    ax8.bar(
        ['Coverage', 'Miss Rate'],
        [coverage, 1 - coverage],
        color=['#2ecc71', '#e74c3c'],
        alpha=0.7
    )
    ax8.set_ylabel('Proportion')
    ax8.set_title(f'Conformal Coverage\n(Target: 95%)', fontweight='bold')
    ax8.set_ylim([0, 1])
    ax8.grid(axis='y', alpha=0.3)

    # 8. Focus score distribution by set size
    ax9 = fig.add_subplot(gs[2, :2])
    set_sizes = results['comparison']['set_size']
    unique_sizes = sorted(list(set(set_sizes)))

    shap_focus_by_size = [[] for _ in unique_sizes]
    gradcam_focus_by_size = [[] for _ in unique_sizes]

    for i, size in enumerate(set_sizes):
        size_idx = unique_sizes.index(size)
        shap_focus_by_size[size_idx].append(results['shap']['focus_score'][i])
        gradcam_focus_by_size[size_idx].append(results['gradcam']['focus_score'][i])

    positions = np.arange(len(unique_sizes))
    bp1 = ax9.boxplot(
        shap_focus_by_size,
        positions=positions - 0.2,
        widths=0.3,
        patch_artist=True,
        labels=unique_sizes
    )
    bp2 = ax9.boxplot(
        gradcam_focus_by_size,
        positions=positions + 0.2,
        widths=0.3,
        patch_artist=True,
        labels=unique_sizes
    )

    for patch in bp1['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.7)
    for patch in bp2['boxes']:
        patch.set_facecolor('#e74c3c')
        patch.set_alpha(0.7)

    ax9.set_xlabel('Prediction Set Size')
    ax9.set_ylabel('Focus Score')
    ax9.set_title('Focus Score Distribution by Prediction Set Size', fontweight='bold')
    ax9.set_xticks(positions)
    ax9.set_xticklabels(unique_sizes)
    ax9.legend([bp1['boxes'][0], bp2['boxes'][0]], ['SHAP', 'GradCAM'])
    ax9.grid(axis='y', alpha=0.3)

    # 9. Statistical comparison table
    ax10 = fig.add_subplot(gs[2, 2:])
    ax10.axis('tight')
    ax10.axis('off')

    # Create comparison statistics
    metrics = []
    shap_means = []
    gradcam_means = []
    p_values = []

    if 'deletion_auc' in results['shap'] and len(results['shap']['deletion_auc']) > 0:
        metrics.append('Deletion AUC')
        shap_means.append(f"{np.mean(results['shap']['deletion_auc']):.3f}")
        gradcam_means.append(f"{np.mean(results['gradcam']['deletion_auc']):.3f}")
        _, p = stats.ttest_ind(results['shap']['deletion_auc'], results['gradcam']['deletion_auc'])
        p_values.append(f"{p:.4f}")

    if 'insertion_auc' in results['shap'] and len(results['shap']['insertion_auc']) > 0:
        metrics.append('Insertion AUC')
        shap_means.append(f"{np.mean(results['shap']['insertion_auc']):.3f}")
        gradcam_means.append(f"{np.mean(results['gradcam']['insertion_auc']):.3f}")
        _, p = stats.ttest_ind(results['shap']['insertion_auc'], results['gradcam']['insertion_auc'])
        p_values.append(f"{p:.4f}")

    metrics.append('Focus Score')
    shap_means.append(f"{np.mean(results['shap']['focus_score']):.3f}")
    gradcam_means.append(f"{np.mean(results['gradcam']['focus_score']):.3f}")
    _, p = stats.ttest_ind(results['shap']['focus_score'], results['gradcam']['focus_score'])
    p_values.append(f"{p:.4f}")

    metrics.append('Inter-class Sim')
    shap_means.append(f"{np.mean(results['shap']['inter_class_similarity']):.3f}")
    gradcam_means.append(f"{np.mean(results['gradcam']['inter_class_similarity']):.3f}")
    _, p = stats.ttest_ind(
        results['shap']['inter_class_similarity'],
        results['gradcam']['inter_class_similarity']
    )
    p_values.append(f"{p:.4f}")

    metrics.append('Spatial Consistency')
    shap_means.append("-")
    gradcam_means.append("-")
    p_values.append(f"{np.mean(results['comparison']['spatial_consistency']):.3f}")

    table_data = list(zip(metrics, shap_means, gradcam_means, p_values))
    table = ax10.table(
        cellText=table_data,
        colLabels=['Metric', 'SHAP (μ)', 'GradCAM (μ)', 'p-value/IoU'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax10.set_title('Statistical Comparison', fontweight='bold', pad=20)

    plt.suptitle(
        'Comprehensive Evaluation: SHAP vs GradCAM for Conformal Uncertainty',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig


def create_uncertainty_landscape_plot(insights: Dict, save_path: Optional[str] = None):
    """
    Create a comprehensive visualization of the uncertainty landscape.

    Args:
        insights: Dictionary containing uncertainty insights
        save_path: Path to save the visualization

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Pattern distribution pie chart
    ax1 = fig.add_subplot(gs[0, 0])
    patterns = list(insights['uncertainty_patterns'].keys())
    counts = list(insights['uncertainty_patterns'].values())
    colors_chart = plt.cm.Set3(np.linspace(0, 1, len(patterns)))

    ax1.pie(counts, labels=patterns, autopct='%1.1f%%', colors=colors_chart)
    ax1.set_title('Distribution of Uncertainty Patterns', fontweight='bold')

    # 2. Entropy vs Prominence scatter
    ax2 = fig.add_subplot(gs[0, 1])

    entropies = []
    prominences = []
    set_sizes = []

    for size, sigs in insights['signature_by_set_size'].items():
        for sig in sigs:
            entropies.append(sig['shap']['entropy'])
            prominences.append(sig['shap']['prominence'])
            set_sizes.append(size)

    scatter = ax2.scatter(entropies, prominences, c=set_sizes, cmap='viridis', s=50, alpha=0.6)
    ax2.set_xlabel('Entropy (Dispersion)')
    ax2.set_ylabel('Prominence (Peak Strength)')
    ax2.set_title('Uncertainty Signature Space', fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='Set Size')

    # Add quadrant labels
    ax2.text(
        0.05, 0.95, 'Focused\nSpecific',
        transform=ax2.transAxes,
        fontsize=9, ha='left', va='top', style='italic'
    )
    ax2.text(
        0.95, 0.95, 'Sparse\nPeaks',
        transform=ax2.transAxes,
        fontsize=9, ha='right', va='top', style='italic'
    )
    ax2.text(
        0.05, 0.05, 'Uniform\nWeak',
        transform=ax2.transAxes,
        fontsize=9, ha='left', va='bottom', style='italic'
    )
    ax2.text(
        0.95, 0.05, 'Diffuse\nGlobal',
        transform=ax2.transAxes,
        fontsize=9, ha='right', va='bottom', style='italic'
    )

    # 3. Coverage by set size
    ax3 = fig.add_subplot(gs[0, 2])

    coverages_by_size = defaultdict(list)
    for size, sigs in insights['signature_by_set_size'].items():
        for sig in sigs:
            coverages_by_size[size].append(sig['shap']['coverage'])

    sizes = sorted(coverages_by_size.keys())
    coverages = [coverages_by_size[s] for s in sizes]

    bp = ax3.boxplot(coverages, labels=sizes, patch_artist=True)
    for patch, size in zip(bp['boxes'], sizes):
        patch.set_facecolor(plt.cm.viridis(size / max(sizes)))
        patch.set_alpha(0.7)

    ax3.set_xlabel('Prediction Set Size')
    ax3.set_ylabel('Coverage (Fraction of Image)')
    ax3.set_title('Attention Coverage by Set Size', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # 4. Class co-occurrence heatmap
    ax4 = fig.add_subplot(gs[1, :])

    # Convert confusion matrix to array
    unique_classes = list(
        set(list(insights['class_confusion_matrix'].keys()) +
            [c for v in insights['class_confusion_matrix'].values() for c in v.keys()])
    )[:20]

    matrix = np.zeros((len(unique_classes), len(unique_classes)))
    for i, c1 in enumerate(unique_classes):
        for j, c2 in enumerate(unique_classes):
            if c1 in insights['class_confusion_matrix'] and c2 in insights['class_confusion_matrix'][c1]:
                matrix[i, j] = insights['class_confusion_matrix'][c1][c2]

    # Make symmetric
    matrix = matrix + matrix.T
    np.fill_diagonal(matrix, 0)

    im = ax4.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax4.set_xticks(range(len(unique_classes)))
    ax4.set_yticks(range(len(unique_classes)))
    ax4.set_xticklabels([f'C{c}' for c in unique_classes], rotation=45, ha='right')
    ax4.set_yticklabels([f'C{c}' for c in unique_classes])
    ax4.set_title('Class Co-occurrence in Prediction Sets', fontweight='bold')
    ax4.set_xlabel('Class Index')
    ax4.set_ylabel('Class Index')

    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04, label='Co-occurrence Count')

    plt.suptitle('Uncertainty Landscape Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return fig


# ==============================================================================
# 5. ANALYSIS FUNCTIONS
# ==============================================================================

def evaluate_on_conformal_sets(
    evaluator: ConformalExplainabilityEvaluator,
    model: nn.Module,
    test_dataset: Dataset,
    conformal_predictor,
    shap_fn: Callable,
    gradcam_fn: Callable,
    n_samples: int = 100,
    verbose: bool = True
) -> Dict:
    """
    Main evaluation function for conformal prediction sets.

    Args:
        evaluator: ConformalExplainabilityEvaluator instance
        model: Trained model
        test_dataset: Test dataset
        conformal_predictor: Conformal predictor object
        shap_fn: Function to generate SHAP explanations
        gradcam_fn: Function to generate GradCAM explanations
        n_samples: Number of samples to evaluate
        verbose: Whether to print progress

    Returns:
        Dictionary containing evaluation results
    """
    results = {
        'shap': defaultdict(list),
        'gradcam': defaultdict(list),
        'comparison': defaultdict(list)
    }

    # Sample indices for evaluation
    sample_indices = np.random.choice(
        len(test_dataset),
        min(n_samples, len(test_dataset)),
        replace=False
    )

    for idx in tqdm(sample_indices, desc="Evaluating samples"):
        image, true_label = test_dataset[idx][:2]

        # Get conformal prediction set
        with torch.no_grad():
            image_batch = image.unsqueeze(0).to(evaluator.device)
            prediction_set = conformal_predictor.predict(image_batch)
            prediction_set = prediction_set[0].cpu().numpy()

        # Skip if prediction set has only one class
        if len(prediction_set) <= 1:
            continue

        # Generate explanations for all classes in prediction set
        shap_explanations = {}
        gradcam_explanations = {}

        for class_idx in prediction_set:
            shap_explanations[class_idx] = shap_fn(image, class_idx)
            gradcam_explanations[class_idx] = gradcam_fn(image, class_idx)

        # Compute differential heatmaps
        shap_differential = evaluator.compute_differential_heatmap(shap_explanations)
        gradcam_differential = evaluator.compute_differential_heatmap(gradcam_explanations)

        # Uncertainty-specific metrics
        shap_focus = evaluator.uncertainty_focus_score(shap_differential)
        gradcam_focus = evaluator.uncertainty_focus_score(gradcam_differential)

        shap_similarity = evaluator.inter_class_explanation_similarity(shap_explanations)
        gradcam_similarity = evaluator.inter_class_explanation_similarity(gradcam_explanations)

        results['shap']['focus_score'].append(shap_focus)
        results['shap']['inter_class_similarity'].append(shap_similarity)
        results['gradcam']['focus_score'].append(gradcam_focus)
        results['gradcam']['inter_class_similarity'].append(gradcam_similarity)

        # Comparison metrics
        for class_idx in prediction_set:
            consistency = evaluator.spatial_consistency_score(
                shap_explanations[class_idx],
                gradcam_explanations[class_idx]
            )
            results['comparison']['spatial_consistency'].append(consistency)

        # Classify uncertainty type
        shap_type = evaluator.uncertainty_type_classification(shap_explanations, shap_differential)
        gradcam_type = evaluator.uncertainty_type_classification(
            gradcam_explanations,
            gradcam_differential
        )

        results['shap']['uncertainty_type'].append(shap_type)
        results['gradcam']['uncertainty_type'].append(gradcam_type)

        # Store metadata
        results['comparison']['set_size'].append(len(prediction_set))
        results['comparison']['true_in_set'].append(true_label in prediction_set)

    return results


def generate_differential_insights(
    model: nn.Module,
    test_dataset: Dataset,
    conformal_predictor,
    shap_fn: Callable,
    gradcam_fn: Callable,
    class_names: Dict,
    n_samples: int = 50
) -> Dict:
    """
    Generate interesting insights from differential heatmap analysis.

    Args:
        model: Trained model
        test_dataset: Test dataset
        conformal_predictor: Conformal predictor object
        shap_fn: Function to generate SHAP explanations
        gradcam_fn: Function to generate GradCAM explanations
        class_names: Dictionary mapping class indices to names
        n_samples: Number of samples to analyze

    Returns:
        Dictionary containing insights
    """
    analyzer = DifferentialHeatmapAnalyzer(model)

    insights = {
        'uncertainty_patterns': defaultdict(int),
        'class_confusion_matrix': defaultdict(lambda: defaultdict(float)),
        'signature_by_set_size': defaultdict(list),
        'interesting_cases': []
    }

    all_shap_explanations = []
    all_gradcam_explanations = []

    print("\n" + "=" * 80)
    print("DIFFERENTIAL HEATMAP ANALYSIS - GENERATING INSIGHTS")
    print("=" * 80)

    for idx in range(min(n_samples, len(test_dataset))):
        image, true_label = test_dataset[idx][:2]

        # Get prediction set
        with torch.no_grad():
            image_batch = image.unsqueeze(0).to('cuda')
            prediction_set = conformal_predictor.predict(image_batch)
            prediction_set = prediction_set[0].cpu().numpy()

        if len(prediction_set) <= 1:
            continue

        # Generate explanations
        shap_explanations = {}
        gradcam_explanations = {}

        for class_idx in prediction_set[:5]:
            shap_explanations[class_idx] = shap_fn(image, class_idx)
            gradcam_explanations[class_idx] = gradcam_fn(image, class_idx)

        all_shap_explanations.append(shap_explanations)
        all_gradcam_explanations.append(gradcam_explanations)

        # Compute differential maps
        evaluator = ConformalExplainabilityEvaluator(model)

        shap_differential = evaluator.compute_differential_heatmap(shap_explanations)
        gradcam_differential = evaluator.compute_differential_heatmap(gradcam_explanations)

        # Generate uncertainty signature
        shap_signature = analyzer.generate_uncertainty_signature(shap_differential)
        gradcam_signature = analyzer.generate_uncertainty_signature(gradcam_differential)

        # Collect insights
        insights['uncertainty_patterns'][shap_signature['pattern']] += 1
        insights['signature_by_set_size'][len(prediction_set)].append({
            'shap': shap_signature,
            'gradcam': gradcam_signature
        })

        # Track class confusion
        for c1 in prediction_set:
            for c2 in prediction_set:
                if c1 != c2:
                    insights['class_confusion_matrix'][c1][c2] += 1

        # Identify interesting cases
        if shap_signature['pattern'] != gradcam_signature['pattern']:
            insights['interesting_cases'].append({
                'idx': idx,
                'set_size': len(prediction_set),
                'shap_pattern': shap_signature['pattern'],
                'gradcam_pattern': gradcam_signature['pattern'],
                'disagreement': True
            })

    # Perform clustering analysis
    print("\n1. CLUSTERING ANALYSIS OF UNCERTAINTY PATTERNS")
    print("-" * 40)

    shap_clustering = analyzer.compute_explanation_clustering(all_shap_explanations)

    if shap_clustering:
        clusters = fcluster(shap_clustering['linkage'], t=0.5, criterion='distance')
        unique_clusters = np.unique(clusters)

        print(f"Found {len(unique_clusters)} distinct uncertainty patterns")

        for cluster_id in unique_clusters[:3]:
            cluster_indices = np.where(clusters == cluster_id)[0]
            cluster_classes = [shap_clustering['labels'][i] for i in cluster_indices]

            class_counts = Counter(cluster_classes)
            top_classes = class_counts.most_common(3)

            print(f"\nCluster {cluster_id} (size: {len(cluster_indices)}):")
            print("  Most common classes:")
            for cls, count in top_classes:
                print(f"    - {class_names[cls]}: {count} occurrences")

    # Analyze uncertainty patterns
    print("\n2. UNCERTAINTY PATTERN DISTRIBUTION")
    print("-" * 40)
    total = sum(insights['uncertainty_patterns'].values())
    for pattern, count in insights['uncertainty_patterns'].items():
        percentage = 100 * count / total if total > 0 else 0
        print(f"  {pattern}: {count} ({percentage:.1f}%)")

    # Analyze pattern vs set size relationship
    print("\n3. UNCERTAINTY CHARACTERISTICS BY PREDICTION SET SIZE")
    print("-" * 40)
    for set_size in sorted(insights['signature_by_set_size'].keys())[:5]:
        signatures = insights['signature_by_set_size'][set_size]
        if signatures:
            avg_entropy = np.mean([s['shap']['entropy'] for s in signatures])
            avg_prominence = np.mean([s['shap']['prominence'] for s in signatures])
            avg_coverage = np.mean([s['shap']['coverage'] for s in signatures])

            print(f"\n  Set Size {set_size}:")
            print(f"    Avg Entropy:    {avg_entropy:.3f}")
            print(f"    Avg Prominence: {avg_prominence:.3f}")
            print(f"    Avg Coverage:   {avg_coverage:.3f}")

            if set_size <= 3:
                print("    → Small set: Likely similar classes with subtle differences")
            elif set_size <= 5:
                print("    → Medium set: Multiple plausible interpretations")
            else:
                print("    → Large set: High model uncertainty or out-of-distribution")

    # Find most confused class pairs
    print("\n4. MOST FREQUENTLY CO-OCCURRING CLASSES IN PREDICTION SETS")
    print("-" * 40)

    class_pairs = []
    for c1, connections in insights['class_confusion_matrix'].items():
        for c2, count in connections.items():
            if c1 < c2:
                class_pairs.append((c1, c2, count))

    class_pairs = sorted(class_pairs, key=lambda x: x[2], reverse=True)[:5]

    for c1, c2, count in class_pairs:
        print(f"  {class_names[c1]} ↔ {class_names[c2]}: {int(count)} co-occurrences")

    # Interesting cases where methods disagree
    print("\n5. CASES WHERE SHAP AND GRADCAM DISAGREE ON UNCERTAINTY TYPE")
    print("-" * 40)

    disagreements = [c for c in insights['interesting_cases'] if c['disagreement']][:3]
    for case in disagreements:
        print(f"\n  Sample {case['idx']}:")
        print(f"    Set size: {case['set_size']}")
        print(f"    SHAP pattern:    {case['shap_pattern']}")
        print(f"    GradCAM pattern: {case['gradcam_pattern']}")
        print("    → Methods interpret uncertainty differently")

    return insights


def generate_summary_report(
    results: Dict,
    confused_analysis: List,
    save_path: str = 'summary_report.txt'
) -> str:
    """
    Generate a comprehensive text report of the evaluation.

    Args:
        results: Dictionary containing evaluation results
        confused_analysis: List of confused class pair analysis results
        save_path: Path to save the report

    Returns:
        Report text as string
    """
    report = []
    report.append("=" * 80)
    report.append("EXPLAINABILITY EVALUATION REPORT FOR CONFORMAL PREDICTION UNCERTAINTY")
    report.append("=" * 80)
    report.append("")

    # Overall statistics
    report.append("1. OVERALL PERFORMANCE METRICS")
    report.append("-" * 40)

    if 'deletion_auc' in results['shap'] and len(results['shap']['deletion_auc']) > 0:
        report.append(f"   Deletion AUC (↓ better):")
        report.append(
            f"   - SHAP:    {np.mean(results['shap']['deletion_auc']):.3f} "
            f"± {np.std(results['shap']['deletion_auc']):.3f}"
        )
        report.append(
            f"   - GradCAM: {np.mean(results['gradcam']['deletion_auc']):.3f} "
            f"± {np.std(results['gradcam']['deletion_auc']):.3f}"
        )
        report.append("")

    if 'insertion_auc' in results['shap'] and len(results['shap']['insertion_auc']) > 0:
        report.append(f"   Insertion AUC (↑ better):")
        report.append(
            f"   - SHAP:    {np.mean(results['shap']['insertion_auc']):.3f} "
            f"± {np.std(results['shap']['insertion_auc']):.3f}"
        )
        report.append(
            f"   - GradCAM: {np.mean(results['gradcam']['insertion_auc']):.3f} "
            f"± {np.std(results['gradcam']['insertion_auc']):.3f}"
        )
        report.append("")

    # Uncertainty-specific metrics
    report.append("2. UNCERTAINTY EXPLANATION METRICS")
    report.append("-" * 40)
    report.append(f"   Focus Score (↑ = more aleatoric):")
    report.append(
        f"   - SHAP:    {np.mean(results['shap']['focus_score']):.3f} "
        f"± {np.std(results['shap']['focus_score']):.3f}"
    )
    report.append(
        f"   - GradCAM: {np.mean(results['gradcam']['focus_score']):.3f} "
        f"± {np.std(results['gradcam']['focus_score']):.3f}"
    )
    report.append("")
    report.append(f"   Inter-class Similarity:")
    report.append(
        f"   - SHAP:    {np.mean(results['shap']['inter_class_similarity']):.3f} "
        f"± {np.std(results['shap']['inter_class_similarity']):.3f}"
    )
    report.append(
        f"   - GradCAM: {np.mean(results['gradcam']['inter_class_similarity']):.3f} "
        f"± {np.std(results['gradcam']['inter_class_similarity']):.3f}"
    )
    report.append("")

    # Method comparison
    report.append("3. METHOD COMPARISON")
    report.append("-" * 40)
    report.append(f"   Spatial Consistency (SHAP vs GradCAM):")
    report.append(f"   - Mean IoU: {np.mean(results['comparison']['spatial_consistency']):.3f}")
    report.append(f"   - Std IoU:  {np.std(results['comparison']['spatial_consistency']):.3f}")
    report.append("")

    # Uncertainty type distribution
    report.append("4. UNCERTAINTY TYPE CLASSIFICATION")
    report.append("-" * 40)
    shap_types = pd.Series(results['shap']['uncertainty_type']).value_counts()
    gradcam_types = pd.Series(results['gradcam']['uncertainty_type']).value_counts()

    report.append("   SHAP:")
    for utype, count in shap_types.items():
        pct = 100 * count / len(results['shap']['uncertainty_type'])
        report.append(f"   - {utype}: {count} ({pct:.1f}%)")

    report.append("\n   GradCAM:")
    for utype, count in gradcam_types.items():
        pct = 100 * count / len(results['gradcam']['uncertainty_type'])
        report.append(f"   - {utype}: {count} ({pct:.1f}%)")
    report.append("")

    # Conformal prediction statistics
    report.append("5. CONFORMAL PREDICTION STATISTICS")
    report.append("-" * 40)
    report.append(f"   Coverage Rate: {np.mean(results['comparison']['true_in_set']):.3f}")
    report.append(f"   Average Set Size: {np.mean(results['comparison']['set_size']):.2f}")
    report.append("")

    # Confused class analysis
    if confused_analysis:
        report.append("6. CONFUSED CLASS PAIR ANALYSIS")
        report.append("-" * 40)
        for pair_result in confused_analysis:
            if pair_result['shap_similarities']:
                report.append(f"   {pair_result['true_class']} → {pair_result['predicted_class']}:")
                report.append(f"   - SHAP Similarity:    {np.mean(pair_result['shap_similarities']):.3f}")
                report.append(f"   - GradCAM Similarity: {np.mean(pair_result['gradcam_similarities']):.3f}")
                report.append(f"   - Differential Focus: {np.mean(pair_result['differential_focus']):.3f}")
                report.append("")

    # Key insights
    report.append("7. KEY INSIGHTS")
    report.append("-" * 40)

    insights = []

    if 'deletion_auc' in results['shap'] and len(results['shap']['deletion_auc']) > 0:
        shap_del = np.mean(results['shap']['deletion_auc'])
        grad_del = np.mean(results['gradcam']['deletion_auc'])
        if shap_del < grad_del:
            insights.append("• SHAP provides better deletion scores (more faithful)")
        else:
            insights.append("• GradCAM provides better deletion scores (more faithful)")

    shap_focus = np.mean(results['shap']['focus_score'])
    grad_focus = np.mean(results['gradcam']['focus_score'])
    if shap_focus > grad_focus:
        insights.append("• SHAP explanations are more focused (better for aleatoric uncertainty)")
    else:
        insights.append("• GradCAM explanations are more focused (better for aleatoric uncertainty)")

    consistency = np.mean(results['comparison']['spatial_consistency'])
    if consistency > 0.5:
        insights.append(f"• High spatial agreement between methods (IoU: {consistency:.3f})")
    else:
        insights.append(f"• Low spatial agreement between methods (IoU: {consistency:.3f})")

    for insight in insights:
        report.append(insight)

    report.append("")
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    # Save report
    report_text = '\n'.join(report)
    with open(save_path, 'w') as f:
        f.write(report_text)

    print(report_text)

    return report_text


# ==============================================================================
# 6. MODEL CREATION AND TRAINING UTILITIES
# ==============================================================================

def create_resnet50_model(num_classes: int = 200, pretrained: bool = True) -> nn.Module:
    """
    Create ResNet-50 model with custom classifier head.

    Replaces the final FC layer (2048→1000) with a new 2048→num_classes layer
    for CUB-200-2011 classification.

    Args:
        num_classes: Number of output classes (default: 200 for CUB)
        pretrained: Whether to use ImageNet pretrained weights

    Returns:
        model: ResNet-50 with modified classifier head
    """
    from torchvision import models

    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50(weights=None)

    # Replace final layer: 2048 → num_classes
    num_ftrs = model.fc.in_features  # Should be 2048 for ResNet-50
    model.fc = nn.Linear(num_ftrs, num_classes)

    print(f"Created ResNet-50 model:")
    print(f"  - Feature dimension: {num_ftrs}")
    print(f"  - Classifier head: {num_ftrs}→{num_classes}")
    print(f"  - Pretrained on ImageNet: {pretrained}")

    return model


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion,
    optimizer,
    device: str,
    epoch: int
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} - Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion,
    device: str,
    epoch: int
) -> Tuple[float, float]:
    """
    Evaluate on validation/test set.

    Args:
        model: PyTorch model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to evaluate on
        epoch: Current epoch number

    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Epoch {epoch} - Evaluating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def plot_training_curves(history_df: pd.DataFrame, save_dir: str):
    """
    Plot training curves (loss and accuracy) from training history.

    Args:
        history_df: DataFrame containing training history
        save_dir: Directory to save plots
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Loss curves
    axes[0].plot(history_df['train_loss'], label='Train Loss', linewidth=2, marker='o', markersize=4)
    axes[0].plot(history_df['test_loss'], label='Test Loss', linewidth=2, marker='s', markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training and Test Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Accuracy curves
    axes[1].plot(history_df['train_acc'] * 100, label='Train Acc', linewidth=2, marker='o', markersize=4)
    axes[1].plot(history_df['test_acc'] * 100, label='Test Acc', linewidth=2, marker='s', markersize=4)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Training and Test Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Learning rate
    axes[2].plot(history_df['lr'], linewidth=2, marker='o', markersize=4, color='green')
    axes[2].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Training curves saved to {save_path}")


if __name__ == "__main__":
    print("Helper Functions Module Loaded")
    print("This module contains all helper functions and classes for explainability testing")
