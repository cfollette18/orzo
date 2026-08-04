#!/usr/bin/env python3
"""QLoRA SFT for orzo on the edge device (8 GB unified memory).

Defaults are chosen for the edge device: 4-bit NF4 base, LoRA r=16, seq len 2048,
per-device batch 1 with grad accumulation, paged 8-bit Adam, bf16 compute,
gradient checkpointing. Expect ~15 W board power draw; log tegrastats
alongside every run (scripts/tegrastats_log.sh).

If bitsandbytes refuses to cooperate on aarch64, fall back to plain bf16
LoRA: --no-4bit. A 1.5B model in bf16 is ~3 GB of weights, which still fits.

Example:
    python train/train_qlora.py \
        --data data/generated/train.jsonl \
        --output checkpoints/orzo-qwen25-coder-1.5b
"""

import argparse
import glob
import os
import sys

# Jetson PyPI wheels (e.g. torch from pypi.jetson-ai-lab.io) ship with
# CUDA libraries such as cuDSS in nvidia/cu12/lib. Ensure they are on the
# dynamic linker path before importing torch so CUDA kernels load cleanly.
_nvidia_lib = glob.glob(
    os.path.join(sys.prefix, "lib", "python*", "site-packages", "nvidia", "cu12", "lib")
)
if _nvidia_lib and os.path.isdir(_nvidia_lib[0]):
    _ld = os.environ.get("LD_LIBRARY_PATH", "")
    if _nvidia_lib[0] not in _ld:
        os.environ["LD_LIBRARY_PATH"] = _nvidia_lib[0] + (":" + _ld if _ld else "")

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--data", required=True, help="train JSONL with a 'messages' field")
    ap.add_argument("--valid", default=None, help="optional validation JSONL")
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--no-4bit", action="store_true", help="bf16 LoRA fallback (no bitsandbytes)")
    ap.add_argument("--wandb", action="store_true",
                    help="log to Weights & Biases (run `wandb login` first; "
                         "public project = shareable live dashboard)")
    args = ap.parse_args()

    quant = None
    if not args.no_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # flash-attn is not available on edge device
        trust_remote_code=True,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    dataset = load_dataset("json", data_files=args.data, split="train")
    eval_dataset = None
    if args.valid:
        eval_dataset = load_dataset("json", data_files=args.valid, split="train")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    # Some custom architectures (e.g. Kimi K3) do not implement gradient
    # checkpointing. Detect support and fall back gracefully so training can
    # still run on memory-constrained edge devices.
    supports_gc = getattr(model, "supports_gradient_checkpointing", False)
    if not supports_gc:
        print("[orzo] gradient checkpointing not supported by this model; continuing without it")

    sft_config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        max_length=args.seq_len,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=supports_gc,
        bf16=True,
        optim="paged_adamw_8bit" if quant else "adamw_torch_fused",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,  # the NVMe is small; keep one checkpoint
        eval_strategy="epoch" if eval_dataset else "no",
        report_to=["wandb"] if args.wandb else [],
        seed=42,
    )

    # Apply LoRA manually with autocast_adapter_dtype=False so adapter weights
    # stay in the model's native dtype (bfloat16), avoiding PEFT's default
    # float32 cast. This keeps memory low and sidesteps fp32 CUDA kernel issues
    # if a generic wheel without sm_87 support is ever installed.
    model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
