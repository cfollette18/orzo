#!/usr/bin/env bash
# One-time setup of heater (Jetson Orin Nano) as the orzo training box.
# Needs sudo for the swap file. Usage: bash scripts/setup_jetson.sh
set -euo pipefail

echo "== power mode (want MAXN_SUPER)"
nvpmodel -q | tail -3

echo "== 8 GB swapfile (QLoRA on 8 GB unified memory needs headroom)"
if [ ! -f /swapfile ]; then
    sudo fallocate -l 8G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi
free -h

echo "== python venv"
sudo apt-get update
sudo apt-get install -y python3-venv cmake build-essential git
python3 -m venv --system-site-packages ~/orzo-venv
source ~/orzo-venv/bin/activate

echo "== torch (JetPack 6 / CUDA 12.6 aarch64 wheel)"
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

echo "== training deps"
pip install -r train/requirements-jetson.txt

echo "== sanity"
python - <<'PY'
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
