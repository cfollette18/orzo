#!/usr/bin/env python3
"""Create train/valid/test splits from generated task JSONL files.

Reads all task JSONL files in data/generated (except specs.jsonl and
*_specs.jsonl), shuffles, and writes train.jsonl/valid.jsonl/test.jsonl.
"""

import argparse
import json
import os
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="project root")
    ap.add_argument("--valid-pct", type=float, default=0.05)
    ap.add_argument("--test-pct", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    gen_dir = Path(args.root) / "data" / "generated"
    examples = []
    for path in gen_dir.glob("*.jsonl"):
        if path.name in ("specs.jsonl", "train_specs.jsonl", "test.jsonl", "valid.jsonl", "train.jsonl"):
            continue
        with open(path) as f:
            for line in f:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    n = len(examples)
    nt = int(n * args.test_pct)
    nv = int(n * args.valid_pct)
    test = examples[:nt]
    valid = examples[nt:nt + nv]
    train = examples[nt + nv:]

    for name, split in [("test", test), ("valid", valid), ("train", train)]:
        out = gen_dir / f"{name}.jsonl"
        with open(out, "w") as f:
            for ex in split:
                f.write(json.dumps(ex) + "\n")

    print(f"train={len(train)} valid={len(valid)} test={len(test)}")


if __name__ == "__main__":
    main()
