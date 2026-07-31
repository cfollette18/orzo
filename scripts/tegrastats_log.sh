#!/usr/bin/env bash
# Log tegrastats alongside a training run — power/thermal evidence for the README.
# Usage: bash scripts/tegrastats_log.sh runs/run1.tegrastats.log
OUT=${1:-tegrastats.log}
echo "logging tegrastats -> $OUT (Ctrl-C to stop)"
tegrastats --interval 5000 | tee "$OUT"
