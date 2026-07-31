# The orzo dataset

Spec for `orzo-harness-sft`: a synthetic SFT dataset that teaches a small code
model to produce **agent harnesses** — tool schemas, function-calling traces,
and full Python scaffolds.

Everything is ChatML (`{"messages": [...]}`) JSONL. One example per line.

## Task types

### 1. `tool_schema`

- **user**: natural-language agent spec (1–3 sentences)
- **assistant**: a JSON array of tool schemas. Each tool:
  `{"name", "description", "parameters": {<json-schema>}, "returns": <description>}`
- **Validation**: output parses as JSON; every entry has `name`, `description`,
  `parameters`; names are `snake_case`; 2–8 tools.

### 2. `react_trace`

- **user**: agent spec + a concrete user goal + the tool schemas
- **assistant**: a JSON array of steps, each
  `{"thought": str, "action": {"tool": str, "args": {...}}, "observation": <mock result>}`
  ending with `{"thought": str, "action": {"tool": "finish", "args": {"answer": str}}}`
- **Validation**: parses as JSON; every `tool` exists in the provided schemas;
  args match the schema's required params; trace ends with `finish`; ≤ 12 steps.

### 3. `harness_scaffold`

- **user**: agent spec + constraints (retry policy, max steps, approval gates)
- **assistant**: a single complete Python file. Required shape:
  - `TOOLS` dict mapping name → callable
  - `dispatch(name, args)` with try/except + bounded retries
  - a main loop that calls the model, parses tool calls, dispatches, appends
    observations, and terminates on `finish` or `max_steps`
  - stdlib only (plus an OpenAI-compatible client) so generated harnesses run anywhere
- **Validation**: `ast.parse` succeeds; contains `dispatch`, a loop, and `max_steps`;
  no imports outside the allowlist.

## Splits

- `test.jsonl` — **50 specs, frozen before any training.** Never touched by
  training or prompt iteration. Eval-only.
- `valid.jsonl` — ~5% for loss curves.
- `train.jsonl` — the rest.

## System prompt (shared across tasks)

```
You are orzo, a generator of agent harnesses. Given a spec, you produce
tool schemas, function-calling traces, or complete harness code. Output
exactly what is asked: JSON when JSON is asked, code when code is asked.
No prose.
```

## Generation flow

1. `gen_specs.py` — combinatorial spec sampler (domain × tools × constraints ×
   persona). Deterministic with a seed; deduped.
2. `gen_dataset.py` — sends specs to a teacher model (any OpenAI-compatible
   API), validates the output against the rules above, retries once, appends
   to JSONL. Resumable: existing IDs are skipped.
3. Manual spot-check of a sample (yes, actually reading them).
4. Upload to HF Hub with a dataset card.
