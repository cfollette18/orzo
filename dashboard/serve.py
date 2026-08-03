#!/usr/bin/env python3
"""Live pipeline dashboard for orzo — stdlib only, no dependencies.

Serves a self-refreshing page covering the full pipeline:
  dataset generation progress (per task, vs. targets)
  training metrics (loss curve, from the newest trainer_state.json)
  tegrastats (latest power/thermal reading, when logging)
  disk usage

Usage:
    python dashboard/serve.py --root . --port 8000
    # local:  http://localhost:8000
    # jetson: http://heater:8000  (over LAN or Tailscale)
"""

import argparse
import glob
import html
import json
import os
import subprocess
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TASK_TARGETS = {
    "harness_scaffold": 1200, "react_trace": 600, "tool_schema": 450,
    "rules": 300, "guardrails": 150, "hooks": 150, "skills": 150,
}


def dataset_progress(root: str) -> list[tuple[str, int, int]]:
    gen = os.path.join(root, "data", "generated")
    rows = []
    for task, target in TASK_TARGETS.items():
        path = os.path.join(gen, f"{task}.jsonl")
        n = 0
        if os.path.exists(path):
            with open(path, "rb") as f:
                n = sum(1 for _ in f)
        rows.append((task, n, target))
    return rows


def training_metrics(root: str) -> dict | None:
    states = glob.glob(
        os.path.join(root, "**", "trainer_state.json"), recursive=True)
    if not states:
        return None
    newest = max(states, key=os.path.getmtime)
    state = json.load(open(newest))
    losses = [(h.get("step", 0), h["loss"])
              for h in state.get("log_history", []) if "loss" in h]
    return {
        "run": os.path.relpath(os.path.dirname(newest), root),
        "losses": losses,
        "step": state.get("global_step"),
        "max_steps": state.get("max_steps"),
        "epoch": state.get("epoch"),
        "mtime": datetime.fromtimestamp(os.path.getmtime(newest)),
    }


def svg_curve(points: list[tuple[int, float]], w: int = 640, h: int = 160) -> str:
    if len(points) < 2:
        return "<p><em>no training data yet</em></p>"
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs) or 1
    y0, y1 = min(ys), max(ys)
    span = (y1 - y0) or 1e-9
    pts = " ".join(
        f"{(x - x0) / ((x1 - x0) or 1) * (w - 20) + 10:.1f},"
        f"{h - 10 - (y - y0) / span * (h - 20):.1f}"
        for x, y in points
    )
    return (
        f'<svg width="{w}" height="{h}" style="background:#111;border:1px solid #333">'
        f'<polyline points="{pts}" fill="none" stroke="#4fc3f7" stroke-width="2"/>'
        f'<text x="10" y="16" fill="#888" font-size="11">loss {ys[-1]:.4f} '
        f'(min {y0:.4f})</text></svg>'
    )


def tegrastats_tail(root: str) -> str:
    logs = glob.glob(os.path.join(root, "**", "*.tegrastats.log"), recursive=True)
    if not logs:
        return "not logging"
    newest = max(logs, key=os.path.getmtime)
    try:
        lines = open(newest).read().strip().splitlines()
        return lines[-1] if lines else "empty log"
    except OSError:
        return "unreadable"


def disk_usage() -> str:
    try:
        return subprocess.run(
            ["df", "-h", "/"], capture_output=True, text=True
        ).stdout.splitlines()[-1]
    except Exception:
        return "n/a"


def bar(n: int, target: int, width: int = 30) -> str:
    filled = int(width * min(n / target, 1.0))
    return "█" * filled + "░" * (width - filled)


def render(root: str) -> str:
    rows = []
    total_n = total_t = 0
    for task, n, target in dataset_progress(root):
        pct = n / target
        color = "#66bb6a" if pct >= 1 else "#4fc3f7" if pct > 0 else "#555"
        rows.append(
            f"<tr><td>{task}</td><td style='font-family:monospace;color:{color}'>"
            f"{bar(n, target)}</td><td>{n} / {target}</td><td>{pct:.0%}</td></tr>")
        total_n += n
        total_t += target

    tm = training_metrics(root)
    if tm:
        train_html = (
            f"<p>run <code>{html.escape(str(tm['run']))}</code> — step "
            f"{tm['step']}/{tm['max_steps']} (epoch {tm['epoch']:.2f}), "
            f"updated {tm['mtime']:%H:%M:%S}</p>" + svg_curve(tm["losses"]))
    else:
        train_html = "<p><em>no training run detected yet</em></p>"

    return f"""<!doctype html>
<html><head><meta http-equiv="refresh" content="10">
<title>orzo pipeline</title>
<style>
body {{ background:#1a1a1a; color:#ddd; font-family:sans-serif; margin:2em; }}
td {{ padding:2px 12px 2px 0; }} code {{ color:#4fc3f7; }}
h2 {{ color:#aaa; border-bottom:1px solid #333; padding-bottom:4px; }}
</style></head><body>
<h1>orzo — pipeline status</h1>
<p>{datetime.now():%Y-%m-%d %H:%M:%S} (auto-refresh 10s) — root <code>{html.escape(root)}</code></p>
<h2>dataset generation ({total_n} / {total_t})</h2>
<table>{''.join(rows)}</table>
<h2>training</h2>
{train_html}
<h2>tegrastats (latest)</h2>
<pre>{html.escape(tegrastats_tail(root))}</pre>
<h2>disk</h2>
<pre>{html.escape(disk_usage())}</pre>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    root = "."

    def do_GET(self):
        body = render(self.root).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    Handler.root = os.path.abspath(args.root)
    print(f"orzo dashboard: http://localhost:{args.port} (root={Handler.root})")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
