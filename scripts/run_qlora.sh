#!/usr/bin/env bash
# Wrapper to launch QLoRA/bf16-LoRA training on Jetson/Orin with the
# environment that Jetson PyPI wheels require.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/../orzo-venv"

# Jetson wheels from pypi.jetson-ai-lab.io bundle CUDA libs (e.g. cuDSS) under
# nvidia/cu12/lib. They must be on the loader path before torch imports.
NV_LIB="${VENV_DIR}/lib/python3.10/site-packages/nvidia/cu12/lib"
if [ -d "${NV_LIB}" ]; then
    export LD_LIBRARY_PATH="${NV_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Allow the CUDA allocator to grow/expand segments. This avoids NVMap OOM
# failures when loading the base model into the limited unified memory on Orin.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

exec "${VENV_DIR}/bin/python" "${PROJECT_DIR}/train/train_qlora.py" "$@"
