#!/bin/bash
# Setup script for Conformal Prediction with Explainability on CUB-200-2011

set -e  # Exit on error

echo "=========================================="
echo "Setting up environment..."
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Error: requirements.txt not found!"
    echo "Please ensure requirements.txt exists."
    exit 1
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p checkpoints
mkdir -p results

# Download CUB-200-2011 dataset if not present
if [ ! -d "CUB_200_2011" ]; then
    echo ""
    echo "=========================================="
    echo "Downloading CUB-200-2011 dataset..."
    echo "=========================================="

    if [ ! -f "CUB_200_2011.tgz" ]; then
        wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
    fi

    echo "Extracting dataset..."
    tar -xzf CUB_200_2011.tgz
    echo "Dataset extracted to ./CUB_200_2011"
else
    echo "CUB-200-2011 dataset already exists"
fi

# Assemble checkpoint if parts exist (for git LFS workaround)
if [ -f "best-epoch=17.ckpt.part.aa" ]; then
    echo "Assembling checkpoint from parts..."
    cat best-epoch=17.ckpt.part.* > best-epoch=17.ckpt
    echo "Checkpoint assembled"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Environment activated. You can now run:"
echo ""
echo "1. Fine-tune ResNet-50 on CUB-200-2011:"
echo "   python fine-tune.py --data-dir ./CUB_200_2011 --num-epochs 50"
echo ""
echo "2. Test model with metrics and explainability:"
echo "   python test.py --checkpoint ./checkpoints/best_model.pth"
echo ""
echo "3. Run deep explainability analysis:"
echo "   python explain.py --checkpoint ./checkpoints/best_model.pth"
echo ""
echo "Note: To activate the environment later, run:"
echo "   source venv/bin/activate"
echo ""
