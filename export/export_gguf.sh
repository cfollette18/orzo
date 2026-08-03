#!/usr/bin/env bash
# Export an orzo checkpoint to GGUF and register it with Ollama.
# Runs on the edge device (the edge device). Usage:
# bash export/export_gguf.sh checkpoints/orzo-qwen25-coder-1.5b Qwen/Qwen2.5-Coder-1.5B-Instruct
set -euo pipefail

ADAPTER=${1:?path to trained LoRA checkpoint}
BASE=${2:?base model id, e.g. Qwen/Qwen2.5-Coder-1.5B-Instruct}
MERGED=${ADAPTER}-merged
GGUF_DIR=${ADAPTER}-gguf

echo "== 1/4 merge LoRA into base weights"
python - "$ADAPTER" "$BASE" "$MERGED" <<'PY'
import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter, base_id, out = sys.argv[1:4]
base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16)
merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
merged.save_pretrained(out)
AutoTokenizer.from_pretrained(base_id).save_pretrained(out)
print(f"merged -> {out}")
PY

echo "== 2/4 llama.cpp"
if [ ! -d llama.cpp ]; then
 git clone --depth 1 https://github.com/ggml-org/llama.cpp
fi
pip install -r llama.cpp/requirements.txt

echo "== 3/4 convert + quantize"
mkdir -p "$GGUF_DIR"
python llama.cpp/convert_hf_to_gguf.py "$MERGED" \
 --outfile "$GGUF_DIR/orzo-f16.gguf" --outtype f16
cmake -B llama.cpp/build -S llama.cpp -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j --target llama-quantize
./llama.cpp/build/bin/llama-quantize \
 "$GGUF_DIR/orzo-f16.gguf" "$GGUF_DIR/orzo-Q4_K_M.gguf" Q4_K_M

echo "== 4/4 ollama"
cat > "$GGUF_DIR/Modelfile" <<EOF
FROM ./orzo-Q4_K_M.gguf
PARAMETER temperature 0.2
SYSTEM "You are orzo, a generator of agent harnesses. Given a spec, you produce tool schemas, function-calling traces, or complete harness code. Output exactly what is asked: JSON when JSON is asked, code when code is asked. No prose."
EOF
(cd "$GGUF_DIR" && ollama create orzo -f Modelfile)

echo "done. try: ollama run orzo"
echo "disk check:"; df -h / | tail -1
