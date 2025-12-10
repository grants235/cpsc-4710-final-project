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

Train ResNet-50 on CUB-200-2011:

```bash
python fine-tune.py \
  --data-dir ./CUB_200_2011 \
  --num-epochs 50 \
  --batch-size 32 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --save-dir ./checkpoints
```

**Key parameters:**
- `--num-epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size for training (default: 32)
- `--lr`: Initial learning rate (default: 1e-4)
- `--weight-decay`: AdamW weight decay (default: 1e-4)
- `--image-size`: Input image size (default: 448)
- `--use-bbox`: Crop images to bounding boxes
- `--freeze-backbone`: Only train classifier head

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

### 4. Deep Explainability Analysis

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
- `results/results_table.tex`: LaTeX table of metrics
- Console output with detailed insights and statistics

## Project Structure

```
.
├── setup.sh                    # Setup script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── fine-tune.py                # Training script
├── test.py                     # Testing with metrics and explainability
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
    ├── uncertainty_vs_overlap.png
    ├── results_table.tex
    └── sample_*.png
```

## Key Features

### Fine-Tuning (`fine-tune.py`)
- ResNet-50 with 2048→200 classifier head
- AdamW optimizer with weight decay
- Cosine annealing learning rate scheduler
- Modern data augmentation (RandomHorizontalFlip, ColorJitter)
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

### Explainability Analysis (`explain.py`)
- Spatial agreement between SHAP and GradCAM
- Focus scores (Gini coefficient)
- Inter-class feature overlap analysis
- Confusion analysis for similar species
- Uncertainty landscape visualization
- LaTeX table generation for publications

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

## License

MIT License - See LICENSE file for details
