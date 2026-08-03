#!/usr/bin/env python3
"""Build the orzo SFT dataset with a teacher model.

Reads specs from gen_specs.py, asks the teacher for one of the component or
assembly tasks (see data/README.md), validates the result, and appends ChatML
examples to JSONL. Resumable: IDs already in the output file are skipped.

Teacher is any OpenAI-compatible API:
    export ORZO_TEACHER_BASE_URL=https://api.openai.com/v1
    export ORZO_TEACHER_API_KEY=...
    export ORZO_TEACHER_MODEL=gpt-4o-mini
"""

import argparse
import ast
import json
import os
import re
import sys

SYSTEM_PROMPT = (
    "You are orzo, a generator of agent harnesses. Given a spec, you produce "
    "the parts of an agent harness — rules, tool schemas, hooks, guardrails, "
    "skills, function-calling traces — or a complete harness that assembles "
    "them. Output exactly what is asked: JSON when JSON is asked, code when "
    "code is asked, markdown when markdown is asked. No prose."
)

ALLOWED_IMPORTS = {
    "json", "os", "re", "sys", "time", "math", "random", "pathlib",
    "dataclasses", "typing", "collections", "datetime", "urllib",
    "openai", "logging", "argparse", "subprocess", "hashlib", "sqlite3",
    "io", "csv", "functools", "itertools", "enum", "uuid", "string",
    "textwrap", "traceback", "contextlib", "copy", "shutil", "tempfile",
}

TASK_PROMPTS = {
    "tool_schema": (
        "Agent spec: {spec}\nPersona: {persona}\n\n"
        "Design the tool set for this agent using these capabilities: {tools}.\n"
        "Output a JSON array of tool schemas. Each tool needs: name (snake_case), "
        "description, parameters (JSON Schema object), returns (short description). "
        "If the agent persists anything, include db_read and db_write tools. "
        "If memory_store and memory_search are among the capabilities, design them "
        "as embedding-backed vector memory: memory_store(text, metadata) and "
        "memory_search(query, k) returning the k most similar past items. "
        "Output JSON only."
    ),
    "react_trace": (
        "Agent spec: {spec}\nPersona: {persona}\nTools available: {tools}\n"
        "Constraint: {constraints}\n\n"
        "Write a realistic example run as a JSON array of steps. Each step: "
        '{{"thought": str, "action": {{"tool": str, "args": {{...}}}}, '
        '"observation": <plausible mock result>}}. Include at least one step '
        "where a tool call fails and the agent diagnoses the error and fixes "
        "the call (wrong args, bad path, transient error), and one step where "
        "the agent records a reusable learning (db_write to a learnings table, "
        "or memory_store if available). If a step would violate the "
        "constraint, show the guardrail denial as the observation and the agent "
        "adjusting. End with a finish action carrying the final answer. "
        "At most 12 steps. Output JSON only."
    ),
    "rules": (
        "Agent spec: {spec}\nPersona: {persona}\nConstraint: {constraints}\n\n"
        "Write the rules document for this agent — the behavioral contract that "
        "goes in its system prompt. Markdown with exactly these sections: "
        "## Role (what the agent is), ## Rules (numbered must/must-not rules, "
        "including one rule about recording learnings and one about diagnosing "
        "and fixing failed actions before retrying), "
        "## Constraints (limits, termination conditions, when to ask a human). "
        "Markdown only."
    ),
    "hooks": (
        "Agent spec: {spec}\nTools available: {tools}\n\n"
        "Write a Python module of lifecycle hooks for this agent's harness: "
        "pre_tool_call(name, args) (may raise to block a call), "
        "post_tool_call(name, args, result) (logging/auditing), and "
        "on_error(name, args, error) (decide retry vs. abort). Stdlib only. "
        "Output code only, no markdown fences."
    ),
    "skills": (
        "Agent spec: {spec}\nTools available: {tools}\n\n"
        "Write a Python module defining one reusable skill for this agent: a "
        "SKILL dict with name, description, and when_to_use, plus a "
        "run(ctx, **kwargs) function that performs the multi-step procedure "
        "using the tools on ctx. Stdlib only. Output code only, no markdown fences."
    ),
    "guardrails": (
        "Agent spec: {spec}\nTools available: {tools}\nConstraint: {constraints}\n\n"
        "Write a Python module of guardrails for this agent's harness: "
        "validate_tool_call(name, args) that enforces allow/deny lists and "
        "raises on violations, an APPROVAL_REQUIRED set of tools that need a "
        "human yes before running, and validate_output(text) for the final "
        "answer. Stdlib only. Output code only, no markdown fences."
    ),
    "harness_scaffold": (
        "Agent spec: {spec}\nPersona: {persona}\nTools available: {tools}\n"
        "Constraints: {constraints}\n\n"
        "Write one complete, runnable Python file implementing this agent's "
        "full harness: RULES (the system-prompt contract), a TOOLS dict of "
        "stub implementations, dispatch(name, args), lifecycle hooks "
        "(pre_tool_call, post_tool_call, on_error), guardrails "
        "(validate_tool_call with allow/deny and approval gates), run state "
        "and an audit log persisted with sqlite3 (db_read/db_write helpers), "
        "and a main loop that queries an OpenAI-compatible model, parses tool "
        "calls as JSON, validates and dispatches them through the hooks and "
        "guardrails, appends observations, and stops on finish or max_steps. "
        "on_error must diagnose the failure and return a fixed call when one "
        "is plausible (corrected args, backoff, alternative tool) rather than "
        "blindly retrying. After each run and each failure, append what "
        "worked or failed to a learnings table, and read past learnings when "
        "planning. If memory_store/memory_search are among the tools, "
        "implement them as vector memory: embed text via the "
        "OpenAI-compatible embeddings endpoint, store embeddings in sqlite, "
        "and answer memory_search(query, k) with cosine-similarity top-k; "
        "recall relevant memories at the start of each run. "
        "Stdlib plus the openai package only. Output code only, no markdown fences."
    ),
}


def strip_fences(text: str) -> str:
    m = re.search(r"```(?:json|python|markdown)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _parse_python(text: str) -> ast.Module | None:
    try:
        return ast.parse(strip_fences(text))
    except SyntaxError:
        return None


def _check_imports(tree: ast.Module) -> bool:
    imports = {
        node.names[0].name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    return imports <= ALLOWED_IMPORTS


def valid_tool_schema(text: str) -> bool:
    try:
        tools = json.loads(strip_fences(text))
    except json.JSONDecodeError:
        return False
    if not isinstance(tools, list) or not 3 <= len(tools) <= 8:
        return False
    for t in tools:
        if not all(k in t for k in ("name", "description", "parameters")):
            return False
        if not re.fullmatch(r"[a-z][a-z0-9_]*", str(t["name"])):
            return False
    return True


def valid_react_trace(text: str, tool_names: list[str]) -> bool:
    try:
        steps = json.loads(strip_fences(text))
    except json.JSONDecodeError:
        return False
    if not isinstance(steps, list) or not 1 <= len(steps) <= 12:
        return False
    for s in steps:
        if not isinstance(s, dict) or "thought" not in s or "action" not in s:
            return False
        if s["action"].get("tool") not in set(tool_names) | {"finish"}:
            return False
    return steps[-1]["action"].get("tool") == "finish"


def valid_rules(text: str) -> bool:
    doc = strip_fences(text)
    return all(h in doc for h in ("## Role", "## Rules", "## Constraints"))


def valid_hooks(text: str) -> bool:
    tree = _parse_python(text)
    if tree is None or not _check_imports(tree):
        return False
    code = strip_fences(text)
    return all(n in code for n in ("pre_tool_call", "post_tool_call", "on_error"))


def valid_skills(text: str) -> bool:
    tree = _parse_python(text)
    if tree is None or not _check_imports(tree):
        return False
    code = strip_fences(text)
    return "SKILL" in code and "def run(" in code


def valid_guardrails(text: str) -> bool:
    tree = _parse_python(text)
    if tree is None or not _check_imports(tree):
        return False
    code = strip_fences(text)
    return (
        "validate_tool_call" in code
        and "APPROVAL_REQUIRED" in code
        and ("DENY" in code or "deny" in code)
    )


def valid_harness(text: str, spec: dict) -> bool:
    tree = _parse_python(text)
    if tree is None or not _check_imports(tree):
        return False
    code = strip_fences(text)
    has_loop = any(isinstance(n, (ast.While, ast.For)) for n in ast.walk(tree))
    required = (
        "RULES", "TOOLS", "dispatch", "pre_tool_call", "post_tool_call",
        "validate_tool_call", "sqlite3", "max_steps", "learnings",
    )
    if not (has_loop and all(r in code for r in required)):
        return False
    # vector memory is mandatory when the spec carries memory tools
    if "memory_search" in spec["tools"]:
        return all(
            r in code
            for r in ("memory_store", "memory_search", "cosine", "embedding")
        )
    return True


VALIDATORS = {
    "tool_schema": lambda text, spec: valid_tool_schema(text),
    "react_trace": lambda text, spec: valid_react_trace(text, spec["tools"]),
    "rules": lambda text, spec: valid_rules(text),
    "hooks": lambda text, spec: valid_hooks(text),
    "skills": lambda text, spec: valid_skills(text),
    "guardrails": lambda text, spec: valid_guardrails(text),
    "harness_scaffold": lambda text, spec: valid_harness(text, spec),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default="data/generated/specs.jsonl")
    ap.add_argument("--task", choices=TASK_PROMPTS, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    from openai import OpenAI  # lazy: validators/prompts must import without the client

    client = OpenAI(
        base_url=os.environ["ORZO_TEACHER_BASE_URL"],
        api_key=os.environ["ORZO_TEACHER_API_KEY"],
    )
    model = os.environ["ORZO_TEACHER_MODEL"]

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(line)["spec_id"] for line in f}

    specs = [json.loads(line) for line in open(args.specs)]
    if args.limit:
        specs = specs[: args.limit]

    validate = VALIDATORS[args.task]

    def generate(spec):
        prompt = TASK_PROMPTS[args.task].format(
            spec=spec["spec"], persona=spec["persona"],
            tools=", ".join(spec["tools"]), constraints=spec["constraints"],
        )
        for _ in range(2):  # one retry on validation failure
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=8192,
                # deepseek-v4 is a reasoning model; keep the whole budget
                # for content or reasoning can starve it to empty output
                extra_body={"thinking": {"type": "disabled"}},
            )
            text = resp.choices[0].message.content
            if validate(text, spec):
                return spec, prompt, strip_fences(text)
        return spec, prompt, None

    todo = [s for s in specs if s["id"] not in done]
    skipped = len(specs) - len(todo)
    written = failed = 0

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with open(args.out, "a") as out, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(generate, s) for s in todo]
        for fut in as_completed(futures):
            spec, prompt, target = fut.result()
            if target is None:
                failed += 1
                continue
            out.write(json.dumps({
                "spec_id": spec["id"],
                "task": args.task,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target},
                ],
            }) + "\n")
            out.flush()
            written += 1
            if written % 25 == 0:
                print(f"{written} written, {failed} failed", flush=True)

    print(f"done: {written} written, {skipped} skipped, {failed} failed",
          file=sys.stderr)


if __name__ == "__main__":
    main()
