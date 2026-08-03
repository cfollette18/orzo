#!/usr/bin/env python3
"""Functional evals for orzo — does the generated harness actually work?

Prompts a served model (Ollama's OpenAI-compatible endpoint on the edge device) with
the frozen test specs and scores the outputs. The base model and the
fine-tune get the exact same specs so the comparison is fair.

Metrics per task:
  tool_schema      — JSON parses, schema fields present, valid names
  react_trace      — JSON parses, tools exist, args satisfy required params,
                     ends with finish
  harness_scaffold — ast.parse succeeds, required structure, safe imports,
                     and a subprocess smoke-run of `python harness.py --help`
                     (5 s timeout, no network)

Usage:
    python eval/run_eval.py --endpoint http://the edge device:11434/v1 \
        --model orzo --specs data/generated/test.jsonl --task harness_scaffold
"""

import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile

from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from gen_dataset import (  # noqa: E402
    SYSTEM_PROMPT,
    TASK_PROMPTS,
    VALIDATORS,
    strip_fences,
    valid_harness,
)


def smoke_run(code: str, timeout: int = 5) -> bool:
    """The harness must at least start and parse its args without crashing."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "harness.py")
        with open(path, "w") as f:
            f.write(code)
        env = {**os.environ, "OPENAI_API_KEY": "eval-dummy", "OPENAI_BASE_URL": "http://127.0.0.1:1"}
        try:
            proc = subprocess.run(
                [sys.executable, path, "--help"],
                capture_output=True, timeout=timeout, env=env,
            )
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://the edge device:11434/v1")
    ap.add_argument("--model", required=True, help="served model name, e.g. orzo or a HF id")
    ap.add_argument("--specs", required=True)
    ap.add_argument("--task", choices=TASK_PROMPTS, required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--smoke", action="store_true", help="also smoke-run harnesses")
    ap.add_argument("--out", default=None, help="write per-spec results JSONL here")
    args = ap.parse_args()

    client = OpenAI(base_url=args.endpoint, api_key="ollama")
    specs = [json.loads(line) for line in open(args.specs)][: args.limit]

    results = []
    passed = 0
    for spec in specs:
        prompt = TASK_PROMPTS[args.task].format(
            spec=spec["spec"], persona=spec["persona"],
            tools=", ".join(spec["tools"]), constraints=spec["constraints"],
        )
        resp = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        text = resp.choices[0].message.content
        ok = VALIDATORS[args.task](text, spec)
        smoke = None
        if ok and args.task == "harness_scaffold" and args.smoke:
            smoke = smoke_run(strip_fences(text))
            ok = ok and smoke
        passed += ok
        results.append({"spec_id": spec["id"], "valid": ok, "smoke_run": smoke})
        print(f"{'PASS' if ok else 'FAIL'} {spec['id']}", flush=True)

    score = passed / max(len(results), 1)
    print(f"\n{args.model} on {args.task}: {passed}/{len(results)} = {score:.0%}")
    if args.out:
        with open(args.out, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
