#!/bin/bash
# Complete Tuolumne/ROCm environment setup for Byte Latent Transformer (BLT).
#
# Run this script from the BLT repository root. Before starting it, load the
# user's configured base Conda installation:
#
#   conda_base
#   cd /p/vast1/kirchenb/hlm-root/blt
#   bash install_tuolumne_291_643_blt.sh
#
# The PyTorch and xFormers wheels are large, and some dependencies may build
# with Ninja. Allocate a compute node with ample CPU cores first, for example:
#
# flux alloc -q pdebug --bank=effml --job-name=build -t59 -N1 -n1 -g1 -c96 -ofastload -o mpibind=off --exclusive --unbuffered --label-io

set -euo pipefail

REPO=$(pwd)

if [[ ! -f "${REPO}/pyproject.toml" || ! -f "${REPO}/requirements.txt" || ! -d "${REPO}/bytelatent" ]]; then
    echo "ERROR: run this script from the BLT repository root." >&2
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: Conda is unavailable. Run 'conda_base' before this script." >&2
    exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "base" ]]; then
    echo "ERROR: the base Conda environment must be active." >&2
    echo "Run 'conda_base' before this script." >&2
    exit 1
fi

BASE_PREFIX=$(conda info --base)
if [[ "${CONDA_PREFIX:-}" != "${BASE_PREFIX}" ]]; then
    echo "ERROR: expected active Conda base at ${BASE_PREFIX}, got ${CONDA_PREFIX:-<unset>}." >&2
    echo "Run 'conda_base' before this script." >&2
    exit 1
fi

# Enable `conda activate` in this non-interactive script.
# shellcheck disable=SC1090
source "${BASE_PREFIX}/etc/profile.d/conda.sh"

# Match the installation layout and naming convention used by the known-good
# Tuolumne Singleshot environment.
: "${WRKSPC:?ERROR: WRKSPC is not set}"
INSTALLDIR=${WRKSPC}
ENV_NAME="tuolumne_conda_291_643_blt"
ENV_PREFIX="${INSTALLDIR}/${ENV_NAME}"

if [[ -e "${ENV_PREFIX}" ]]; then
    echo "ERROR: environment path already exists: ${ENV_PREFIX}" >&2
    echo "Move or remove it explicitly before retrying; this script will not overwrite it." >&2
    exit 1
fi

cd "${INSTALLDIR}"

echo "Active base Conda environment:"
conda env list | grep -F '*'

# Create the Python environment and verify that subsequent pip commands use it.
conda create --prefix "${ENV_PREFIX}" python=3.12 --yes -c defaults
conda activate "${ENV_PREFIX}"
echo "Pip executable: $(command -v pip)"
echo "Python executable: $(command -v python)"

# Conda packages required for compatible C++ runtime libraries.
conda install -c conda-forge libstdcxx-ng --yes

# Match the Tuolumne software stack used by the working Singleshot installer.
rocm_version=6.4.3
module load "rocm/${rocm_version}"

######### INSTALL PIP PACKAGES ##############################################

# PyTorch and core requirements. Keep all build settings inline so they reach
# any PEP 517/Ninja subprocesses launched by pip.
MAX_JOBS=48 PYTORCH_ROCM_ARCH='gfx942' GPU_ARCHS='gfx942' \
    python -m pip install \
        torch==2.9.1+rocm6.4 \
        torchvision \
        torchaudio \
        torchmetrics \
        --index-url https://download.pytorch.org/whl/rocm6.4

MAX_JOBS=48 PYTORCH_ROCM_ARCH='gfx942' GPU_ARCHS='gfx942' \
    python -m pip install ninja packaging numpy

# Upstream BLT pins a 2024 CUDA-oriented xFormers source revision. For this
# ROCm stack, use PyTorch's prebuilt ROCm wheel matched to torch 2.9.1.
MAX_JOBS=48 PYTORCH_ROCM_ARCH='gfx942' GPU_ARCHS='gfx942' \
    python -m pip install \
        xformers==0.0.33.post2 \
        --index-url https://download.pytorch.org/whl/rocm6.4

# Install BLT's checked-in dependency set. BLT's upstream Conda/pip workflow
# runs from the repository root rather than installing BLT editable.
cd "${REPO}"
MAX_JOBS=48 PYTORCH_ROCM_ARCH='gfx942' GPU_ARCHS='gfx942' \
    python -m pip install -r requirements.txt

# Add generic support for notebooks.
MAX_JOBS=48 PYTORCH_ROCM_ARCH='gfx942' GPU_ARCHS='gfx942' \
    python -m pip install \
        jupyterlab

# Imported by BLT's entropy preprocessing and Hugging Face model-loading paths
# but omitted from the upstream requirements files. hf_xet enables the transfer
# backend used by the official Meta checkpoint repositories.
MAX_JOBS=48 PYTORCH_ROCM_ARCH='gfx942' GPU_ARCHS='gfx942' \
    python -m pip install \
        jsonlines==4.0.0 \
        safetensors==0.8.0 \
        hf_xet==1.5.2

# The converted BLT checkpoint uses Hugging Face Transformers. Its BLT support
# requires a newer Hub client than upstream native BLT's requirements pin; the
# native ModelHubMixin path is compatible with this version as well.
MAX_JOBS=48 PYTORCH_ROCM_ARCH='gfx942' GPU_ARCHS='gfx942' \
    python -m pip install \
        huggingface-hub==0.36.2 \
        transformers==4.57.6 \
        accelerate==1.14.0

# Lightweight environment checks only; no model download, GPU run, or BLT
# end-to-end smoke test is performed by this installer.
MAX_JOBS=48 PYTORCH_ROCM_ARCH='gfx942' GPU_ARCHS='gfx942' \
    python -m pip check

MAX_JOBS=48 PYTORCH_ROCM_ARCH='gfx942' GPU_ARCHS='gfx942' \
    python -c "import torch, xformers; print(f'torch={torch.__version__}'); print(f'xformers={xformers.__version__}'); print(f'HIP runtime={torch.version.hip}')"

echo
echo "Environment created successfully: ${ENV_PREFIX}"
echo "Activate it with:"
echo "  conda_base"
echo "  conda_activate ${ENV_PREFIX}"
echo "Then run BLT commands from:"
echo "  ${REPO}"
