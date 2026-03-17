#!/bin/bash
# ==============================================================================
# HPC Leipzig Cluster Setup — Genesis Drone Landing Project
#
# Creates a conda environment with all dependencies for RL training
# on the HPC cluster (A30/V100 GPUs with CUDA 12.6).
#
# Usage:
#   source setup_env.sh          # first time: creates env + installs everything
#   source setup_env.sh --load   # subsequent sessions: load modules + activate
# ==============================================================================

set -e

ENV_NAME="ba"
ENV_DIR="$HOME/.conda/envs/$ENV_NAME"

# ==============================================================================
# 1. MODULE LOADS
# ==============================================================================

module purge
module load Anaconda3
module load CUDA/12.6.0

eval "$(conda shell.bash hook)"

echo "=== Loaded modules ==="
module list

# ==============================================================================
# 2. QUICK RELOAD MODE
# ==============================================================================

if [ "$1" = "--load" ]; then
    if [ ! -d "$ENV_DIR" ]; then
        echo "ERROR: conda env '$ENV_NAME' not found. Run without --load first."
        return 1 2>/dev/null || exit 1
    fi
    conda activate "$ENV_NAME"
    echo "=== Activated: $ENV_NAME ==="
    echo "Python:         $(python --version)"
    echo "Torch:          $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
    echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'unknown')"
    return 0 2>/dev/null || exit 0
fi

# ==============================================================================
# 3. CREATE CONDA ENVIRONMENT
# ==============================================================================

if conda env list | grep -q "^$ENV_NAME "; then
    echo "=== Conda env '$ENV_NAME' already exists, activating ==="
else
    echo "=== Creating conda env '$ENV_NAME' with Python 3.13 ==="
    conda create -y --name "$ENV_NAME" python=3.13
fi

conda activate "$ENV_NAME"
pip install --upgrade pip

# ==============================================================================
# 4. PYTORCH (cu126 wheels)
# ==============================================================================

echo "=== Installing PyTorch ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# ==============================================================================
# 5. CORE DEPENDENCIES
# ==============================================================================

echo "=== Installing core dependencies ==="
pip install \
    numpy \
    scipy \
    pyyaml \
    matplotlib \
    pillow \
    tqdm \
    tensorboard \
    wandb \
    requests

# ==============================================================================
# 6. RL LIBRARIES
# ==============================================================================

echo "=== Installing RL libraries ==="
pip install \
    rsl-rl-lib==5.0.1 \
    stable-baselines3 \
    tensordict

# ==============================================================================
# 7. GENESIS + SIMULATION DEPENDENCIES
# ==============================================================================

echo "=== Installing Genesis (no-deps to avoid HPC module conflicts) ==="
pip install --no-deps genesis-world==0.3.13

# Genesis runtime dependencies not covered by HPC modules
pip install \
    trimesh \
    pydantic \
    rich \
    colorama \
    pyglet \
    glfw \
    PyOpenGL \
    coacd \
    libigl \
    tetgen \
    pycollada \
    pygltflib \
    fast_simplification \
    pysplashsurf \
    gstaichi \
    gs-madrona \
    etils \
    mujoco \
    numba \
    psutil \
    imageio \
    freetype-py \
    pyvista \
    rtree \
    z3-solver

# ==============================================================================
# 8. VERIFICATION
# ==============================================================================

echo ""
echo "=========================================="
echo "  Setup complete"
echo "=========================================="
echo "Python:          $(python --version)"
echo "Torch:           $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available:  $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count:       $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "Genesis:         $(python -c 'import genesis; print(genesis.__version__)' 2>/dev/null || echo 'import failed')"
echo ""
echo "For future sessions, run:  source ~/genesis/hpc/setup_env.sh --load"
