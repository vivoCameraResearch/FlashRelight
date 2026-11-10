#!/usr/bin/env bash
# One-shot environment setup for the FlashRelight inference release.
#
#   bash install.sh                 # create conda env "flashrelight" (cu121)
#   CUDA=cu118 bash install.sh      # pick a different CUDA wheel
#   ENV_NAME=myenv bash install.sh  # pick a different env name
#   WITH_FLASH_ATTN=1 bash install.sh   # also try to build FlashAttention-2
set -euo pipefail

ENV_NAME="${ENV_NAME:-flashrelight}"
PY_VER="${PY_VER:-3.10}"
CUDA="${CUDA:-cu121}"
TORCH_VER="${TORCH_VER:-2.1.0}"
TORCHVISION_VER="${TORCHVISION_VER:-0.16.0}"
WITH_FLASH_ATTN="${WITH_FLASH_ATTN:-0}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found; install Miniconda first: https://docs.conda.io/en/latest/miniconda.html" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "==> conda env '${ENV_NAME}' already exists, reusing it"
else
  echo "==> creating conda env '${ENV_NAME}' (python ${PY_VER})"
  conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
fi
conda activate "${ENV_NAME}"

echo "==> installing torch ${TORCH_VER} (${CUDA})"
pip install --upgrade pip
pip install "torch==${TORCH_VER}" "torchvision==${TORCHVISION_VER}" \
    --index-url "https://download.pytorch.org/whl/${CUDA}"

echo "==> installing FlashRelight requirements"
pip install -r "$(dirname "$0")/requirements.txt"

if [ "${WITH_FLASH_ATTN}" = "1" ]; then
  echo "==> installing FlashAttention-2 (optional, needs nvcc + ~20 min)"
  pip install packaging ninja
  pip install flash-attn --no-build-isolation || \
    echo "!! flash-attn build failed; continuing with the PyTorch SDPA fallback"
fi

echo
echo "==> verifying"
python - <<'PYCHECK'
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import torch, cv2, imageio, diffusers, transformers, gradio
print(f"  torch        {torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}")
print(f"  opencv       {cv2.__version__}")
print(f"  imageio      {imageio.__version__}")
print(f"  diffusers    {diffusers.__version__}")
print(f"  transformers {transformers.__version__}")
print(f"  gradio       {gradio.__version__}")
try:
    import timm, kornia
    print(f"  matting      timm {timm.__version__} / kornia {kornia.__version__}")
except ModuleNotFoundError:
    print("  matting      deps missing -> run inference with --matting none")
try:
    import flash_attn
    print(f"  flash_attn   {flash_attn.__version__}")
except ModuleNotFoundError:
    print("  flash_attn   not installed -> PyTorch SDPA fallback")
PYCHECK

echo
echo "Done. Next:"
echo "  conda activate ${ENV_NAME}"
echo "  See the README for weight setup and the full relighting or standalone delighting commands."
