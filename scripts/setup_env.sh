#!/bin/bash
# GARSEF Environment Setup Script
# Run this to install all required dependencies

echo "=== GARSEF Environment Setup ==="

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Conda found. Creating environment..."

    # Create conda environment (if not exists)
    conda create -n garsef python=3.10 -y 2>/dev/null || true

    echo ""
    echo "To activate the environment, run:"
    echo "  conda activate garsef"
    echo ""
    echo "Then install dependencies with:"
    echo "  pip install -r requirements.txt"
    echo ""
else
    echo "Conda not found. Using pip directly..."
fi

# Install core dependencies via pip
echo "Installing core dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || \
    pip install torch torchvision torchaudio

pip install torch-geometric

pip install transformers>=4.30.0
pip install fair-esm>=2.0.0

pip install biopython>=1.81
pip install networkx>=3.0

pip install numpy pandas scikit-learn tqdm pyyaml wandb

echo ""
echo "=== Core dependencies installed ==="
echo ""
echo "Optional dependencies:"
echo "  - For GPU support: pip install torch --index-url https://download.pytorch.org/whl/cu118"
echo "  - For SaProt: Download from HuggingFace: westlake-repl/SaProt_650M_AF2"
echo "  - For foldseek: https://github.com/steineggerlab/foldseek"
echo ""
echo "To verify installation, run:"
echo "  python -c \"import torch; import esm; print('OK')\""
