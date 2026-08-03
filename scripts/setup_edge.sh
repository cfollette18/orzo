#!/usr/bin/env bash
# Set up the edge device (edge device) as the orzo training box — no sudo required.
# Run scripts/setup_swap.sh with sudo FIRST (see that script), then:
#     bash scripts/setup_edge.sh
set -euo pipefail

echo "== swap check (want >= 6 GB total: zram + /swapfile)"
TOTAL_SWAP_MB=$(awk '/SwapTotal/ {print int($2/1024)}' /proc/meminfo)
echo "total swap: ${TOTAL_SWAP_MB} MB"
if [ "$TOTAL_SWAP_MB" -lt 6144 ]; then
    echo "WARNING: low swap. Ask the machine owner to run: sudo bash scripts/setup_swap.sh"
fi

echo "== power mode (want MAXN_SUPER)"
nvpmodel -q | tail -3 || true

echo "== uv (aarch64, user-local)"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "== venv"
uv venv ~/orzo-venv
VENV_PY=~/orzo-venv/bin/python

echo "== torch (JetPack 6 / CUDA 12.6 driver — pin the cu126 aarch64 build)"
uv pip install --python "$VENV_PY" "torch==2.13.0+cu126" --index-url https://download.pytorch.org/whl/cu126
"$VENV_PY" -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())" || {
    echo "torch CUDA check failed — see train/requirements-edge.txt for fallbacks"; exit 1; }

echo "== training deps"
uv pip install --python "$VENV_PY" -r train/requirements-edge.txt

echo "== sanity"
"$VENV_PY" - <<'PY'
import torch, transformers, peft, trl
print("transformers", transformers.__version__)
print("peft", peft.__version__, "| trl", trl.__version__)
try:
    import bitsandbytes as bnb
    print("bitsandbytes", bnb.__version__)
except Exception as e:
    print("bitsandbytes FAILED:", e)
    print("-> fall back to: train_qlora.py --no-4bit")
PY

echo "done. activate with: source ~/orzo-venv/bin/activate"
