# orzo

**Small pasta, small model.** An agent-harness generator fine-tuned entirely on a $249 NVIDIA Jetson Orin Nano (8 GB).

Give `orzo` a natural-language spec — *"an agent that watches a GitHub repo and opens an issue when CI goes red"* — and it generates a **working agent harness**: tool schemas, the tool-use loop, dispatch, retries, and the system prompt. Not a chatbot. A machine that builds the scaffolding agents run on.

> Status: **early development**. See the [roadmap](#roadmap) — nothing below claims results that don't exist yet.

## Why this exists

Most "I fine-tuned a model" repos are a notebook and a screenshot. This one is built around the uncomfortable constraints instead:

- **Edge training** — every training run happens on a Jetson Orin Nano named `heater`, with 8 GB of *unified* memory. QLoRA, gradient checkpointing, swap discipline, and `tegrastats` power/thermal logs for every run.
- **Functional evals, not vibes** — generated harnesses are compiled, smoke-run against mock tools, and scored on dispatch correctness. Base model vs. fine-tune, side by side.
- **Reproducible everything** — the dataset spec, generation code, training config, and export path are all in this repo. The dataset and final GGUF are published to Hugging Face Hub.

## The pipeline

```
natural-language agent specs
        │
        ▼
┌────────────────┐   ┌─────────────────┐   ┌──────────────────────┐
│ teacher model  │──▶│ orzo dataset    │──▶│ QLoRA SFT            │
│ (API, filtered │   │ (ChatML JSONL,  │   │ (Qwen2.5-Coder-1.5B, │
│  + validated)  │   │  3 task types)  │   │  on the Orin Nano)   │
└────────────────┘   └─────────────────┘   └──────────┬───────────┘
                                                      ▼
┌────────────────┐   ┌─────────────────┐   ┌──────────────────────┐
│ functional     │◀──│ ollama serve    │◀──│ merge LoRA → GGUF    │
│ evals          │   │ (on heater)     │   │ Q4_K_M               │
└────────────────┘   └─────────────────┘   └──────────────────────┘
```

## Repo layout

- `data/` — dataset spec, spec generator, teacher-driven dataset builder
- `train/` — QLoRA training script tuned for the Orin Nano's 8 GB
- `eval/` — functional eval harness (does the generated harness *run*?)
- `export/` — LoRA merge → GGUF → Ollama
- `scripts/` — Jetson setup (power mode, swap, deps)

## The three dataset tasks

| Task | Input | Target output |
|------|-------|---------------|
| `tool_schema` | agent spec | JSON tool schemas (name, params, returns) |
| `react_trace` | spec + user goal | thought/action/observation trace with correct tool calls |
| `harness_scaffold` | agent spec | complete runnable Python harness (loop, dispatch, retries) |

See `data/README.md` for the full spec.

## Roadmap

- [x] Repurpose + wipe the Orin Nano (`heater`) as a dedicated training box
- [x] Repo scaffold
- [ ] Spec generator + frozen held-out test set
- [ ] Teacher-driven dataset generation (~2–5k validated examples)
- [ ] QLoRA training runs on `heater` (loss curves + tegrastats logs)
- [ ] Merge → GGUF → serve via Ollama on the Jetson
- [ ] Functional evals: base vs. fine-tuned comparison
- [ ] Dataset + model published to Hugging Face Hub
- [ ] Demo GIF + write-up

## Hardware

| | |
|---|---|
| Trainer + server | Jetson Orin Nano 8 GB (JetPack 6.x, L4T R36) |
| Power mode | MAXN_SUPER |
| Storage | NVMe |
| Remote access | Tailscale |

## License

MIT — see `LICENSE`.
