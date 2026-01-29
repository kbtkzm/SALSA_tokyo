#!/usr/bin/env bash

set -e -o pipefail

CONDA_SH="/work/04/gr44/r44000/miniconda3/etc/profile.d/conda.sh"
ENV_NAME="salsa_env"

export CONDA_PKGS_DIRS="/work/04/gr44/r44000/conda_pkgs"
export CONDA_ENVS_PATH="/work/04/gr44/r44000/conda_envs"
export XDG_CACHE_HOME="/work/04/gr44/r44000/conda_cache"

mkdir -p "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH" "$XDG_CACHE_HOME"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Conda profile not found: $CONDA_SH" >&2
  exit 1
fi

source "$CONDA_SH"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" -y --override-channels -c conda-forge python=3.9 pip
fi

conda activate "$ENV_NAME"

conda install -y --override-channels --strict-channel-priority \
  -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=12.1

python - << 'EOF'
import sys
import torch

print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

if torch.version.cuda is None:
    sys.exit("ERROR: torch.version.cuda is None")
if not torch.cuda.is_available():
    sys.exit("ERROR: CUDA is not available")
gpu_name = torch.cuda.get_device_name(0)
if "A100" not in gpu_name:
    sys.exit(f"ERROR: GPU is not A100 (found: {gpu_name})")
EOF

pip install numpy "scipy==1.7.1" tqdm pyyaml tensorboard
