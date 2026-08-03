#!/usr/bin/env bash
# The ONLY root-requiring step of edge-device setup: an 8 GB swapfile.
# QLoRA on 8 GB unified memory wants headroom beyond the ~4 GB stock zram.
# The owner of the machine runs this themselves:
#     sudo bash scripts/setup_swap.sh
set -euo pipefail

if [ -f /swapfile ]; then
    echo "/swapfile already exists"; swapon --show; exit 0
fi

fallocate -l 8G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab

echo "swap now:"
swapon --show
free -h | head -2
