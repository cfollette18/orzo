#!/usr/bin/env python3
"""Combinatorial sampler of agent specs for the orzo dataset.

Deterministic with --seed. Emits JSONL: {"id", "domain", "spec", "tools",
"constraints", "persona"} per line. These specs are the inputs the teacher
model turns into training examples (see gen_dataset.py).
"""

import argparse
import hashlib
import itertools
import json
import random

DOMAINS = {
    "devops": "an agent that {goal} for a software team's infrastructure",
    "research": "an agent that {goal} for an academic researcher",
    "data": "an agent that {goal} in a data engineering pipeline",
    "home": "an agent that {goal} for a smart home",
    "support": "an agent that {goal} for a customer support team",
    "personal": "an agent that {goal} for a busy individual",
    "security": "an agent that {goal} for a security operations team",
    "finance": "an agent that {goal} for a small finance team",
}

GOALS = {
    "devops": [
        "watches a GitHub repo and opens an issue when CI goes red",
        "triages alerts and restarts flaky services after confirmation",
        "summarizes nightly deploy logs and flags regressions",
        "keeps dependency versions up to date with small PRs",
    ],
    "research": [
        "finds new papers on a topic and writes a weekly digest",
        "extracts claims and citations from PDFs into a notes database",
        "cross-checks references in a manuscript",
        "monitors conference deadlines and drafts submission checklists",
    ],
    "data": [
        "validates incoming CSV drops and quarantines bad rows",
        "monitors pipeline runs and backfills failed partitions",
        "profiles tables and suggests schema migrations",
        "deduplicates records using fuzzy matching with approval gates",
    ],
    "home": [
        "optimizes thermostat schedules around weather forecasts",
        "inventories the pantry from receipts and suggests shopping lists",
        "monitors energy usage and identifies vampire loads",
        "coordinates robot vacuum runs around the family calendar",
    ],
    "support": [
        "drafts first replies to tickets and escalates angry customers",
        "groups duplicate bug reports and links them to known issues",
        "keeps the help-center FAQ in sync with resolved tickets",
        "surfaces churn-risk accounts from support sentiment",
    ],
    "personal": [
        "plans meals for the week and orders groceries",
        "tracks subscriptions and flags price increases",
        "summarizes the day's email and drafts replies",
        "organizes photo backups and frees up phone storage",
    ],
    "security": [
        "reviews auth logs and locks accounts showing brute-force patterns",
        "scans public buckets for accidental exposure",
        "correlates IDS alerts into incident timelines",
        "verifies TLS cert expiry across all owned domains",
    ],
    "finance": [
        "reconciles bank statements against the ledger",
        "flags anomalous expenses for human review",
        "prepares monthly cash-flow summaries",
        "chases overdue invoices with polite follow-up emails",
    ],
}

TOOL_POOLS = {
    "devops": ["http_request", "shell", "git", "read_file", "write_file", "send_slack", "db_read", "db_write"],
    "research": ["web_search", "read_pdf", "read_file", "write_file", "db_read", "db_write", "send_email"],
    "data": ["db_read", "db_write", "read_file", "write_file", "shell", "http_request", "send_email"],
    "home": ["http_request", "db_read", "db_write", "calendar", "send_notification", "smart_home_api"],
    "support": ["ticket_api", "db_read", "db_write", "send_email", "crm_lookup", "send_slack", "http_request"],
    "personal": ["send_email", "calendar", "http_request", "db_read", "db_write", "search"],
    "security": ["shell", "db_read", "db_write", "http_request", "sql_query", "send_slack", "write_file"],
    "finance": ["db_read", "db_write", "sql_query", "read_file", "send_email", "http_request", "write_file"],
}

CONSTRAINTS = [
    "max 10 steps per run",
    "max 25 steps per run",
    "retry failed tool calls up to 3 times with backoff",
    "ask for human approval before any write operation",
    "ask for human approval before sending any external message",
    "never retry a 4xx response",
    "log every tool call to a local file",
]

PERSONAS = [
    "cautious and literal",
    "fast and pragmatic",
    "verbose and explanatory",
    "terse and quiet unless something is wrong",
]


def spec_id(domain: str, goal: str, tools: tuple, constraint: str, persona: str) -> str:
    raw = "|".join([domain, goal, ",".join(tools), constraint, persona])
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/generated/specs.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-domain", type=int, default=0,
                    help="cap specs per domain (0 = no cap)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen, specs = set(), []

    for domain, template in DOMAINS.items():
        combos = []
        for goal in GOALS[domain]:
            pool = TOOL_POOLS[domain]
            rest = tuple(t for t in sorted(pool) if t not in ("db_read", "db_write"))
            for k in (3, 4, 5):  # k = total tools; db_read/db_write always included
                for extra in itertools.combinations(rest, k - 2):
                    tools = ("db_read", "db_write") + extra
                    for constraint in CONSTRAINTS:
                        for persona in PERSONAS:
                            combos.append((goal, tools, constraint, persona))
        rng.shuffle(combos)
        if args.per_domain:
            combos = combos[: args.per_domain]
        for goal, tools, constraint, persona in combos:
            sid = spec_id(domain, goal, tools, constraint, persona)
            if sid in seen:
                continue
            seen.add(sid)
            tool_list = list(tools)
            # deterministic: ~60% of specs also get embedding-backed memory
            if int(sid, 16) % 5 < 3:
                tool_list += ["memory_store", "memory_search"]
            specs.append({
                "id": sid,
                "domain": domain,
                "spec": template.format(goal=goal),
                "tools": tool_list,
                "constraints": constraint,
                "persona": persona,
            })

    rng.shuffle(specs)
    with open(args.out, "w") as f:
        for s in specs:
            f.write(json.dumps(s) + "\n")
    print(f"wrote {len(specs)} specs -> {args.out}")


if __name__ == "__main__":
    main()
