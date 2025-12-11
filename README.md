# Conformal Prediction with Explainability on CUB-200-2011

Fine-tuning ResNet-50 on bird species classification with conformal prediction sets and explainability methods (SHAP, GradCAM, differential explanations).

## Quick Start

### 1. Setup

Run the setup script to install dependencies and download the dataset:

```bash
bash setup.sh
```

This will:
- Create a Python virtual environment
- Install all required packages from `requirements.txt`
- Download and extract the CUB-200-2011 dataset
- Create necessary directories (`checkpoints/`, `results/`)

### 2. Fine-tune Model

Train ResNet-50 on CUB-200-2011 (paper specifications):

**Memory-efficient (recommended for limited GPU/RAM):**
```bash
python fine-tune.py \
    --data-dir ./CUB_200_2011 \
    --num-epochs 50 \
    --batch-size 32 \
    --accumulation-steps 2 \
    --lr 5e-4 \
    --lr-min 5e-5 \
    --use-bbox \
    --mixed-precision
```

**High-memory systems (if you have ≥16GB GPU RAM):**
```bash
python fine-tune.py \
    --data-dir ./CUB_200_2011 \
    --num-epochs 50 \
    --batch-size 64 \
    --accumulation-steps 1 \
    --lr 5e-4 \
    --lr-min 5e-5 \
    --use-bbox
```

**Key parameters (paper specifications):**
- `--batch-size`: Batch size per GPU (default: 32)
- `--accumulation-steps`: Gradient accumulation (default: 2, effective batch=64)
- `--mixed-precision`: Use fp16 to save memory (recommended)
- `--lr`: Initial learning rate (default: 5e-4)
- `--lr-min`: Minimum LR for cosine annealing (default: 5e-5)
- `--weight-decay`: AdamW weight decay (default: 1e-4)
- `--use-bbox`: Crop images to bounding boxes (recommended)

**Data augmentation:** Uses `TrivialAugmentWide()` as specified in paper

**Note:** Effective batch size = `batch-size × accumulation-steps` (paper uses 64)

**Outputs:**
- `checkpoints/best_model.pth`: Best model checkpoint
- `checkpoints/final_model.pth`: Final model after all epochs
- `checkpoints/training_history.csv`: Training metrics per epoch
- `results/training_curves.png`: Loss, accuracy, and LR curves

### 3. Test Model

Evaluate model with metrics, conformal prediction, and explainability:

```bash
python test.py \
  --checkpoint ./checkpoints/best_model.pth \
  --data-dir ./CUB_200_2011 \
  --n-samples 10 \
  --alpha 0.05 \
  --results-dir ./results
```

**Key parameters:**
- `--checkpoint`: Path to model checkpoint
- `--alpha`: Conformal prediction coverage (0.05 = 95% coverage)
- `--n-samples`: Number of samples to visualize (default: 10)
- `--batch-size`: Batch size for evaluation (default: 32)

**Outputs:**
- `results/metrics.csv`: AUROC, AUPRC, and conformal metrics
- `results/conformal_results.csv`: Detailed conformal prediction results
- `results/conformal_analysis.png`: Comparison charts (4 panels)
- `results/sample_XXX.png`: Individual sample visualizations with:
  - Original image
  - Model predictions
  - SHAP explanations
  - GradCAM explanations
  - Differential explanations (unique features per class)

**Conformal Metrics Computed:**
- Coverage and average set size for all combinations of:
  - **Predictors:** Split, Class-Conditional, RC3P
  - **Score Functions:** APS, LAC, Entmax

### 4. Generate Paper Tables and Figures

Generate all analysis from the paper (calibration curves, tables, etc.):

```bash
python analysis.py \
  --checkpoint ./checkpoints/best_model.pth \
  --data-dir ./CUB_200_2011 \
  --batch-size 64 \
  --alpha 0.05 \
  --results-dir ./results
```

**Outputs:**
- `results/lowest_auprc_species.csv` + `.tex`: Species with lowest AUPRC (Table)
- `results/top_confused_pairs.csv` + `.tex`: Most confused class pairs (Table)
- `results/plots/calibration_curve_*.png`: Calibration curves for all predictors
- `results/plots/method_comparison_alpha_0.05.png`: Bar chart comparing methods

### 5. Deep Explainability Analysis

Run comprehensive explainability evaluation:

```bash
python explain.py \
  --checkpoint ./checkpoints/best_model.pth \
  --cub-root ./CUB_200_2011 \
  --n-samples 50 \
  --alpha 0.05 \
  --output-dir ./results
```

**Key parameters:**
- `--n-samples`: Number of samples to analyze (default: 50)
- `--alpha`: Conformal prediction alpha (default: 0.05)
- `--batch-size`: Batch size (default: 64)
- `--image-size`: Input image size (default: 448)

**Outputs:**
- `results/uncertainty_vs_overlap.png`: Uncertainty analysis
- `results/explainability_metrics.csv`: Detailed metrics
- `results/focus_score_comparison.png`: SHAP vs GradCAM focus
- `results/spatial_agreement_distribution.png`: Spatial agreement histogram
- `results/metrics_summary.png`: Summary bar chart
- `results/results_table.tex`: LaTeX table of metrics
- Console output with detailed insights and statistics

## Project Structure

```
.
├── setup.sh                    # Setup script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── fine-tune.py                # Training script (paper specifications)
├── test.py                     # Testing with metrics and explainability
├── analysis.py                 # Generate paper tables and figures (NEW!)
├── explain.py                  # Deep explainability analysis
├── helpers.py                  # Shared utility functions
│
├── CUB_200_2011/              # Dataset (downloaded by setup.sh)
│   ├── images/
│   ├── images.txt
│   ├── train_test_split.txt
│   └── ...
│
├── checkpoints/                # Model checkpoints
│   ├── best_model.pth
│   ├── final_model.pth
│   └── training_history.csv
│
└── results/                    # All outputs
    ├── training_curves.png
    ├── metrics.csv
    ├── conformal_results.csv
    ├── conformal_analysis.png
    ├── lowest_auprc_species.csv + .tex
    ├── top_confused_pairs.csv + .tex
    ├── explainability_metrics.csv
    ├── focus_score_comparison.png
    ├── spatial_agreement_distribution.png
    ├── metrics_summary.png
    ├── uncertainty_vs_overlap.png
    ├── results_table.tex
    ├── sample_*.png
    └── plots/
        ├── calibration_curve_SplitPredictor.png
        ├── calibration_curve_ClassConditionalPredictor.png
        ├── calibration_curve_RC3PPredictor.png
        └── method_comparison_alpha_0.05.png
```

## Key Features

### Fine-Tuning (`fine-tune.py`) - Paper Specifications
- ResNet-50 with 2048→200 classifier head
- AdamW optimizer (lr: 5e-4 → 5e-5, weight decay: 1e-4)
- Cosine annealing learning rate scheduler
- TrivialAugmentWide() augmentation (as in paper)
- Batch size: 64
- Image size: 448×448
- Bounding box cropping (recommended)
- ImageNet normalization
- Automatic training curve visualization

### Testing (`test.py`)
- **Classification Metrics:** AUROC, AUPRC
- **Conformal Prediction:** Tests all predictor/score combinations
- **Explainability Methods:**
  - SHAP (gradient-based)
  - GradCAM (activation-based)
  - Differential explanations (unique features)
- Comprehensive visualizations and numerical results

### Paper Analysis (`analysis.py`) - NEW!
- Per-class AUROC and AUPRC computation
- Species with lowest AUPRC table (LaTeX + CSV)
- Most confused class pairs table (LaTeX + CSV)
- Calibration curves for all conformal predictors
- Method comparison bar charts
- Publication-ready figures

### Explainability Analysis (`explain.py`)
- Spatial agreement between SHAP and GradCAM
- Focus scores (Gini coefficient)
- Inter-class feature overlap analysis
- Confusion analysis for similar species
- Uncertainty landscape visualization
- LaTeX table generation for publications
- Additional visualization plots (distributions, comparisons)

## Dependencies

Main packages (see `requirements.txt` for full list):
- PyTorch
- torchvision
- torchcp (conformal prediction)
- captum (explainability)
- scikit-learn
- pandas
- matplotlib
- seaborn
- tqdm

## Citation

If you use this code, please cite:

```
CUB-200-2011 Dataset:
@techreport{WelinderEtal2010,
  Author = {P. Welinder and S. Branson and T. Mita and C. Wah and F. Schroff and S. Belongie and P. Perona},
  Institution = {California Institute of Technology},
  Number = {CNS-TR-2010-001},
  Title = {{Caltech-UCSD Birds 200}},
  Year = {2010}
}
```

## Troubleshooting

### Out of Memory (OOM) Errors / "Killed" During Training

If training gets killed or you see OOM errors:

**1. Use memory-efficient settings (recommended):**
```bash
python fine-tune.py \
    --batch-size 16 \
    --accumulation-steps 4 \
    --mixed-precision \
    --num-workers 2 \
    --use-bbox
```

**2. Further reduce batch size if needed:**
- Try `--batch-size 8 --accumulation-steps 8`
- Or even `--batch-size 4 --accumulation-steps 16`

**3. Monitor GPU/RAM usage:**
```bash
# For GPU
nvidia-smi -l 1

# For RAM
htop
```

**4. Reduce image size (last resort):**
```bash
python fine-tune.py --image-size 224 --batch-size 32
```

### Gradient Accumulation Explanation

Gradient accumulation lets you use smaller batch sizes while maintaining the same effective batch size:
- `batch-size=32, accumulation-steps=2` → effective batch = 64
- `batch-size=16, accumulation-steps=4` → effective batch = 64
- `batch-size=8, accumulation-steps=8` → effective batch = 64

All achieve the paper's batch size of 64 but use less memory!

## License

MIT License - See LICENSE file for details
