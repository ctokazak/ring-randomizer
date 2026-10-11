#!/usr/bin/env bash
# LeRobot v0.4.1 + ROCm 6.3 + PyTorch 2.7.1 setup for AMD Ryzen AI (HX 370) on Ubuntu 24.04.
#
# Run it twice:
#   1st run: installs ROCm, then asks you to reboot.
#   2nd run (after reboot): installs conda env, PyTorch-ROCm, LeRobot, and verifies the GPU.
#
#   bash setup_lerobot_ryzen.sh
set -euo pipefail

ENV_NAME=lerobot
LEROBOT_DIR="$HOME/lerobot"
LEROBOT_TAG=v0.4.1
CONDA_DIR="$HOME/miniforge3"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- stage 1: ROCm
if ! command -v rocminfo >/dev/null 2>&1; then
  say "Stage 1: installing ROCm 6.3.4 (kernel driver from Ubuntu, --no-dkms)"
  sudo apt update
  sudo apt install -y "linux-headers-$(uname -r)" python3-setuptools python3-wheel wget git
  # render/video = GPU access, dialout = USB serial access to the SO-101 motor bus
  sudo usermod -a -G render,video,dialout "$USER"

  tmp=$(mktemp -d)
  wget -P "$tmp" https://repo.radeon.com/amdgpu-install/6.3.4/ubuntu/noble/amdgpu-install_6.3.60304-1_all.deb
  sudo apt install -y "$tmp/amdgpu-install_6.3.60304-1_all.deb"
  amdgpu-install -y --usecase=rocm --no-dkms

  # Make the Ryzen AI 300 iGPU (gfx1150) present as gfx1100 to ROCm
  grep -q 'HSA_OVERRIDE_GFX_VERSION' ~/.bashrc || echo 'export HSA_OVERRIDE_GFX_VERSION=11.0.0' >> ~/.bashrc

  say "ROCm installed. Reboot now (sudo reboot), then run this script again."
  exit 0
fi

# ---------------------------------------------------------------- stage 2: Python stack
export HSA_OVERRIDE_GFX_VERSION=11.0.0
say "Stage 2: ROCm found ($(rocminfo | grep -m1 -o 'gfx[0-9a-f]*' || echo '?'))"

if ! id -nG "$USER" | grep -qw render; then
  echo "You are not in the 'render' group yet. Did you reboot after stage 1?" >&2
  exit 1
fi

# Miniforge = Miniconda with conda-forge as default (no Anaconda ToS prompt)
if ! command -v conda >/dev/null 2>&1 && [ ! -x "$CONDA_DIR/bin/conda" ]; then
  say "Installing Miniforge to $CONDA_DIR"
  tmp=$(mktemp -d)
  wget -O "$tmp/miniforge.sh" "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash "$tmp/miniforge.sh" -b -p "$CONDA_DIR"
  "$CONDA_DIR/bin/conda" init bash
fi
CONDA_BASE=$(command -v conda >/dev/null 2>&1 && conda info --base || echo "$CONDA_DIR")
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

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

# Editable install with the Feetech servo SDK in one step.
# (The guide's separate `pip install 'lerobot[feetech]'` can replace the editable checkout with the PyPI build.)
say "Installing LeRobot $LEROBOT_TAG (editable) + feetech"
pip install -e "$LEROBOT_DIR[feetech]"

# ---------------------------------------------------------------- verify
say "Verifying"
pip list 2>/dev/null | grep -Ei '^(torch|torchvision|torchaudio|pytorch-triton-rocm|lerobot|feetech)'
python - <<'PY'
import torch
print("torch:", torch.__version__, "| HIP:", torch.version.hip)
ok = torch.cuda.is_available()
print("GPU available:", ok)
if ok:
    print("device name [0]:", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    (x @ x).sum().item(); print("matmul on GPU ok")
import lerobot, scservo_sdk  # noqa: F401
print("lerobot + feetech SDK import ok")
PY

say "Done. Next: open a new terminal, then:"
cat <<EOF
  conda activate $ENV_NAME
  lerobot-find-port        # plug/unplug each arm's USB to identify /dev/ttyACM*
Then follow so101_example.md for calibration, teleop, recording and inference.
EOF
