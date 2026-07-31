#!/usr/bin/env python3
"""Build the orzo SFT dataset with a teacher model.

Reads specs from gen_specs.py, asks the teacher for one of the three task
outputs (tool_schema / react_trace / harness_scaffold), validates the result,
and appends ChatML examples to JSONL. Resumable: IDs already in the output
file are skipped.

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

from openai import OpenAI

SYSTEM_PROMPT = (
    "You are orzo, a generator of agent harnesses. Given a spec, you produce "
    "tool schemas, function-calling traces, or complete harness code. Output "
    "exactly what is asked: JSON when JSON is asked, code when code is asked. "
    "No prose."
)

ALLOWED_IMPORTS = {
    "json", "os", "re", "sys", "time", "math", "random", "pathlib",
    "dataclasses", "typing", "collections", "datetime", "urllib",
    "openai", "logging", "argparse", "subprocess", "hashlib",
}

TASK_PROMPTS = {
    "tool_schema": (
        "Agent spec: {spec}\nPersona: {persona}\n\n"
        "Design the tool set for this agent using these capabilities: {tools}.\n"
        "Output a JSON array of tool schemas. Each tool needs: name (snake_case), "
        "description, parameters (JSON Schema object), returns (short description). "
        "Output JSON only."
    ),
    "react_trace": (
        "Agent spec: {spec}\nPersona: {persona}\nTools available: {tools}\n"
        "Constraint: {constraints}\n\n"
        "Write a realistic example run as a JSON array of steps. Each step: "
        '{{"thought": str, "action": {{"tool": str, "args": {{...}}}}, '
        '"observation": <plausible mock result>}}. End with a finish action '
        'carrying the final answer. At most 12 steps. Output JSON only.'
    ),
    "harness_scaffold": (
        "Agent spec: {spec}\nPersona: {persona}\nTools available: {tools}\n"
        "Constraints: {constraints}\n\n"
        "Write one complete, runnable Python file implementing this agent's "
        "harness. Requirements: a TOOLS dict of stub implementations, a "
        "dispatch(name, args) with bounded retries, a main loop that queries an "
        "OpenAI-compatible model, parses tool calls as JSON, dispatches them, "
        "appends observations, and stops on finish or max_steps. Stdlib plus "
        "the openai package only. Output code only, no markdown fences."
    ),
}


def strip_fences(text: str) -> str:
    m = re.search(r"```(?:json|python)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def valid_tool_schema(text: str) -> bool:
    try:
        tools = json.loads(strip_fences(text))
    except json.JSONDecodeError:
        return False
    if not isinstance(tools, list) or not 2 <= len(tools) <= 8:
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


def valid_harness(text: str) -> bool:
    code = strip_fences(text)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    imports = {
        node.names[0].name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    if not imports <= ALLOWED_IMPORTS:
        return False
    has_loop = any(isinstance(n, (ast.While, ast.For)) for n in ast.walk(tree))
    return has_loop and "dispatch" in code and "max_steps" in code


VALIDATORS = {
    "tool_schema": lambda text, spec: valid_tool_schema(text),
    "react_trace": lambda text, spec: valid_react_trace(text, spec["tools"]),
    "harness_scaffold": lambda text, spec: valid_harness(text),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default="data/generated/specs.jsonl")
    ap.add_argument("--task", choices=TASK_PROMPTS, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

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
    written = skipped = failed = 0
    with open(args.out, "a") as out:
        for spec in specs:
            if spec["id"] in done:
                skipped += 1
                continue
            prompt = TASK_PROMPTS[args.task].format(
                spec=spec["spec"], persona=spec["persona"],
                tools=", ".join(spec["tools"]), constraints=spec["constraints"],
            )
            target = None
            for _ in range(2):  # one retry on validation failure
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
                text = resp.choices[0].message.content
                if validate(text, spec):
                    target = strip_fences(text)
                    break
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
