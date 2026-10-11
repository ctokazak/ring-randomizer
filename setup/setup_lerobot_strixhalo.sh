#!/usr/bin/env bash
# LeRobot v0.4.1 + PyTorch 2.7.1 (ROCm 6.3 wheels) for the hackathon PC:
# AMD Ryzen AI MAX+ 395 (gfx1151) on Ubuntu 26.04, kernel 7.0.
#
# No amdgpu-install needed: the in-kernel amdgpu driver provides /dev/kfd, and the
# PyTorch ROCm wheels bundle their own ROCm runtime. No sudo needed here; GPU access
# needs the user in the 'render' group (sudo usermod -aG render,video,dialout $USER).
set -euo pipefail

ENV_NAME=lerobot
LEROBOT_DIR="$HOME/lerobot"
LEROBOT_TAG=v0.4.1
CONDA_DIR="$HOME/miniforge3"

say() { printf '\n==> %s\n' "$*"; }

# Run gfx1151 on the gfx1100 kernels shipped in the ROCm 6.3 wheels
grep -q 'HSA_OVERRIDE_GFX_VERSION' ~/.bashrc || echo 'export HSA_OVERRIDE_GFX_VERSION=11.0.0' >> ~/.bashrc
export HSA_OVERRIDE_GFX_VERSION=11.0.0

if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  say "Installing Miniforge to $CONDA_DIR"
  tmp=$(mktemp -d)
  wget -q -O "$tmp/miniforge.sh" "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash "$tmp/miniforge.sh" -b -p "$CONDA_DIR"
  "$CONDA_DIR/bin/conda" init bash
fi
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "^$ENV_NAME\s"; then
  say "Creating conda env '$ENV_NAME' (python 3.10)"
  conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"

say "Installing ffmpeg 7.1.1"
conda install -y -c conda-forge ffmpeg=7.1.1

# Install ROCm torch BEFORE lerobot so pip doesn't pull the CUDA build
say "Installing PyTorch 2.7.1 for ROCm 6.3"
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/rocm6.3

if [ ! -d "$LEROBOT_DIR/.git" ]; then
  say "Cloning LeRobot"
  git clone https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
fi
git -C "$LEROBOT_DIR" fetch --tags
git -C "$LEROBOT_DIR" checkout -B "$LEROBOT_TAG" "$LEROBOT_TAG"

say "Installing LeRobot $LEROBOT_TAG (editable) + feetech"
pip install -e "$LEROBOT_DIR[feetech]"

say "Installed packages"
pip list 2>/dev/null | grep -Ei '^(torch|torchvision|torchaudio|pytorch-triton-rocm|lerobot|feetech)'
python -c "import lerobot, scservo_sdk; print('lerobot + feetech SDK import ok')"
say "SETUP DONE"
