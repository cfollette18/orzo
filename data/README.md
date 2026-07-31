# The orzo dataset

Spec for `orzo-harness-sft`: a synthetic SFT dataset that teaches a small code
model to produce **complete agent harnesses** — every component, and the full
assembly. The curriculum mirrors the anatomy of a real harness, so the model
learns the parts *and* how they fit together.

Everything is ChatML (`{"messages": [...]}`) JSONL. One example per line.

## Anatomy of an agent harness (what each category does)

| Component | What it is | Why it exists |
|-----------|-----------|---------------|
| **Rules** | The behavioral contract in the system prompt: role, must/must-not rules, output format, when to stop | Without explicit rules a model improvises; rules make behavior reproducible |
| **Tools** | Typed functions the agent may call — schemas + implementations. Includes `db_read` / `db_write`, file I/O, shell, HTTP | The *only* way the agent touches the world; schemas keep calls valid |
| **Tool calls & dispatch** | The protocol: model emits a JSON call → harness validates args → executes → appends the observation | Separates deciding from doing; every action is parseable, checkable, logged |
| **Hooks** | Lifecycle interception points: `pre_tool_call`, `post_tool_call`, `on_error`, `on_finish` | Logging, metrics, blocking dangerous calls, retries — without touching the loop |
| **Skills** | Packaged multi-step capabilities (bigger than a single tool): a `SKILL` descriptor + `run()` the agent can invoke | Reusable procedures the model doesn't have to re-derive every run |
| **Guardrails** | Input/output validation, tool allow/deny lists, human-approval gates, step & budget caps | Keeps the agent inside its mandate; failures become denials, not damage |
| **Memory & persistence** | Run state, audit log, checkpoints — backed by a real database (`sqlite3`), read *and* written | Agents that can't record what they did can't resume, be audited, or learn |
| **Loop control** | `max_steps`, bounded retries with backoff, explicit termination on `finish` | The difference between an agent and an infinite bill |

## Dataset tasks

The flagship task is the full harness. Component tasks exist so the model
learns each part in isolation — like scales before the concerto.

| Task | Share | Input | Target output |
|------|-------|-------|---------------|
| `harness_scaffold` | ~40% | spec + constraints | **complete runnable Python harness** wiring rules, tools, hooks, guardrails, sqlite memory, and loop control together |
| `react_trace` | ~20% | spec + goal + tools | JSON thought/action/observation trace, incl. a guardrail denial or a retry when the scenario calls for it |
| `tool_schema` | ~15% | spec | JSON array of tool schemas incl. `db_read`/`db_write` where the spec involves persistence |
| `rules` | ~10% | spec + persona | markdown rules doc with `## Role`, `## Rules`, `## Constraints` |
| `guardrails` | ~5% | spec + tools | Python module: `validate_tool_call`, allow/deny lists, approval gate |
| `hooks` | ~5% | spec + tools | Python module: `pre_tool_call`, `post_tool_call`, `on_error` |
| `skills` | ~5% | spec | Python module: `SKILL` descriptor (name, description, when_to_use) + `run(ctx, **kwargs)` |

## Validation (what the generator enforces)

- `tool_schema` — parses as JSON; 3–8 tools; each has `name` (snake_case),
  `description`, `parameters`
- `react_trace` — parses as JSON; ≤ 12 steps; every `tool` exists in the given
  schemas; ends with a `finish` action
- `rules` — contains `## Role`, `## Rules`, `## Constraints` headers
- `hooks` — `ast.parse` passes; defines `pre_tool_call`, `post_tool_call`,
  `on_error`
- `skills` — `ast.parse` passes; defines `SKILL` and `run`
- `guardrails` — `ast.parse` passes; defines `validate_tool_call`, has a deny
  list, has an approval gate
- `harness_scaffold` — `ast.parse` passes; contains `TOOLS`, `dispatch`,
  `pre_tool_call`, `validate_tool_call`, `sqlite3`, a loop, and `max_steps`;
  imports restricted to an allowlist (stdlib + `openai`)

## Splits

- `test.jsonl` — **50 specs, frozen before any training.** Never touched by
  training or prompt iteration. Eval-only.
- `valid.jsonl` — ~5% for loss curves.
- `train.jsonl` — the rest.

## System prompt (shared across tasks)

```
You are orzo, a generator of agent harnesses. Given a spec, you produce the
parts of an agent harness — rules, tool schemas, hooks, guardrails, skills,
function-calling traces — or a complete harness that assembles them. Output
exactly what is asked: JSON when JSON is asked, code when code is asked,
markdown when markdown is asked. No prose.
```

## Generation flow

1. `gen_specs.py` — combinatorial spec sampler (domain × tools × constraints ×
   persona). Deterministic with a seed; deduped. Every spec's tool pool
   includes `db_read` / `db_write` so persistence shows up everywhere.
2. `gen_dataset.py` — sends specs to a teacher model (any OpenAI-compatible
   API), validates output against the rules above, retries once, appends to
   JSONL. Resumable: existing IDs are skipped.
3. Manual spot-check of a sample (yes, actually reading them).
4. Upload to HF Hub with a dataset card.
