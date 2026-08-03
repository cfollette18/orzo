# orzo

An agent-harness generator fine-tuned entirely on an NVIDIA Jetson Orin Nano (8 GB).

[![Built with jig](https://img.shields.io/badge/built%20with-jig-58a6ff)](https://github.com/cfollette18/jig)

The generic training, dataset, eval, and observability tooling that built orzo
lives in its own repository: **[jig](https://github.com/cfollette18/jig)**.
This repo is the model, dataset, and results.

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
- `dashboard/` — live pipeline dashboard (dataset progress, loss curve, tegrastats)
- `scripts/` — Jetson setup (power mode, swap, deps)

## Live observability

The whole pipeline is watchable in real time:

- `dashboard/serve.py` — dependency-free status page covering dataset
  generation (per-task progress vs. targets), training loss curve (from
  `trainer_state.json`), tegrastats power/thermals, and disk:

  ```bash
  python dashboard/serve.py --root . --port 8000
  # laptop: http://localhost:8000 — jetson: http://heater:8000
  ```

- Training can also log to **Weights & Biases** for a full shareable
  dashboard (loss curves, system metrics, public project page):

  ```bash
  wandb login
  python train/train_qlora.py --data ... --output ... --wandb
  ```

- `scripts/tegrastats_log.sh` records power/thermal draw alongside every run.

## The curriculum: anatomy of an agent harness

The dataset teaches each harness component in isolation, then the full
assembly. `data/README.md` defines every
category and its validation rules.

| Task | Share | What it teaches |
|------|-------|-----------------|
| `harness_scaffold` | ~40% | the **full harness**: rules, tools, hooks, guardrails, sqlite-backed memory, loop control |
| `react_trace` | ~20% | tool-calling traces, incl. guardrail denials and retries |
| `tool_schema` | ~15% | tool design, incl. `db_read` / `db_write` |
| `rules` | ~10% | the behavioral contract (role, must/must-not, constraints) |
| `guardrails` | ~5% | allow/deny lists, approval gates, output validation |
| `hooks` | ~5% | lifecycle interception (`pre_tool_call`, `post_tool_call`, `on_error`) |
| `skills` | ~5% | packaged multi-step capabilities (`SKILL` + `run()`) |

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
