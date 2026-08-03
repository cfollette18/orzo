#!/usr/bin/env python3
"""Live pipeline dashboard for orzo — enterprise-style UI, stdlib only.

Tabs:
  Pipeline  — step-by-step guide with live stage status
  Dataset   — per-task generation progress
  Examples  — browse actual dataset points
  Training  — loss curve and run metrics
  System    — tegrastats, disk, hardware

The server is dependency-free Python and runs identically on the laptop
and on the edge device over LAN or Tailscale.

Usage:
    python dashboard/serve.py --root . --port 8000
    # local:       http://localhost:8000
    # edge device: http://edge-device:8000
"""

import argparse
import glob
import json
import os
import subprocess
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

TASK_TARGETS = {
    "harness_scaffold": 1200, "react_trace": 600, "tool_schema": 450,
    "rules": 300, "guardrails": 150, "hooks": 150, "skills": 150,
}

PIPELINE = [
    {
        "id": "specs",
        "title": "Generate dataset",
        "desc": "Produce synthetic agent-harness examples from specs, validate every output, and freeze a held-out test set.",
        "cmds": [
            "python data/gen_specs.py --out data/generated/specs.jsonl",
            "python data/gen_dataset.py --task harness_scaffold --out data/generated/harness_scaffold.jsonl --limit 1200 --workers 8",
            "# ... repeat for react_trace, tool_schema, rules, hooks, skills, guardrails",
        ],
    },
    {
        "id": "train",
        "title": "Train custom model on edge device",
        "desc": "QLoRA fine-tune Qwen2.5-Coder-1.5B on the 8 GB edge device with tegrastats logging.",
        "cmds": [
            "bash scripts/setup_edge.sh",
            "bash scripts/tegrastats_log.sh runs/orzo.tegrastats.log &",
            "python train/train_qlora.py --data data/train.jsonl --valid data/valid.jsonl --output checkpoints/orzo-qwen25-coder-1.5b --wandb",
        ],
    },
    {
        "id": "export",
        "title": "Export to GGUF + Ollama",
        "desc": "Merge LoRA adapters, convert to Q4_K_M GGUF, and register the model with Ollama on the edge device.",
        "cmds": [
            "bash export/export_gguf.sh checkpoints/orzo-qwen25-coder-1.5b Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "ollama run orzo",
        ],
    },
    {
        "id": "eval",
        "title": "Run functional evals",
        "desc": "Score base vs fine-tuned on the frozen test set: does each output compile, run, and dispatch tools correctly?",
        "cmds": [
            "python eval/run_eval.py --model orzo --specs data/examples/test_specs.jsonl --task harness_scaffold --smoke --out eval/orzo.jsonl",
            "python eval/run_eval.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --specs data/examples/test_specs.jsonl --task harness_scaffold --smoke --out eval/base.jsonl",
        ],
    },
    {
        "id": "publish",
        "title": "Publish artifacts",
        "desc": "Push the repo, upload the dataset and GGUF to Hugging Face Hub, and add the results table + demo to the README.",
        "cmds": ["git push", "huggingface-cli upload ...", "ollama push orzo"],
    },
]


def dataset_progress(root):
    gen = os.path.join(root, "data", "generated")
    tasks = []
    for task, target in TASK_TARGETS.items():
        path = os.path.join(gen, f"{task}.jsonl")
        n = 0
        if os.path.exists(path):
            with open(path, "rb") as f:
                n = sum(1 for _ in f)
        tasks.append({"task": task, "n": n, "target": target})
    return tasks


def training_metrics(root):
    states = glob.glob(os.path.join(root, "**", "trainer_state.json"),
                       recursive=True)
    if not states:
        return None
    newest = max(states, key=os.path.getmtime)
    state = json.load(open(newest))
    losses = [{"step": h.get("step", 0), "loss": h["loss"]}
              for h in state.get("log_history", []) if "loss" in h]
    return {
        "run": os.path.relpath(os.path.dirname(newest), root),
        "losses": losses, "step": state.get("global_step"),
        "max_steps": state.get("max_steps"), "epoch": state.get("epoch"),
        "mtime": os.path.getmtime(newest),
    }


def pipeline_state(root):
    tasks = dataset_progress(root)
    total_n = sum(t["n"] for t in tasks)
    total_t = sum(t["target"] for t in tasks)
    tm = training_metrics(root)
    now = time.time()

    def exists(rel):
        return os.path.exists(os.path.join(root, rel))

    specs_done = exists("data/examples/test_specs.jsonl") and total_n >= 1
    train_done = exists("checkpoints/orzo-qwen25-coder-1.5b/adapter_model.safetensors") or \
                 (tm and tm.get("step") and tm.get("max_steps") and tm["step"] >= tm["max_steps"])
    train_active = tm and (now - tm["mtime"] < 120) and not train_done
    export_done = exists("checkpoints/orzo-qwen25-coder-1.5b-gguf/orzo-Q4_K_M.gguf")
    eval_done = exists("eval/orzo.jsonl") and exists("eval/base.jsonl")

    return {
        "specs": {"status": "done" if specs_done else "running" if total_n > 0 else "pending",
                  "progress": total_n / max(total_t, 1),
                  "detail": f"{total_n} / {total_t} examples"},
        "train": {"status": "done" if train_done else "running" if train_active else "pending",
                  "progress": (tm["step"] / tm["max_steps"]) if tm and tm.get("max_steps") else 0,
                  "detail": f"step {tm.get('step') or 0}/{tm.get('max_steps') or '?'}" if tm else "not started"},
        "export": {"status": "done" if export_done else "pending", "detail": "Q4_K_M.gguf present" if export_done else "waiting on training"},
        "eval": {"status": "done" if eval_done else "pending", "detail": "results present" if eval_done else "waiting on export"},
        "publish": {"status": "done" if False else "pending", "detail": "manual step"},
    }


def read_example(root, task, index):
    path = os.path.join(root, "data", "generated", f"{task}.jsonl")
    if task not in TASK_TARGETS or not os.path.exists(path):
        return None
    with open(path) as f:
        lines = f.readlines()
    if not lines:
        return None
    index = index % len(lines)
    ex = json.loads(lines[index])
    msgs = ex["messages"]
    assistant = msgs[2]["content"]
    kind = "code"
    try:
        assistant = json.dumps(json.loads(assistant), indent=2)
        kind = "json"
    except (json.JSONDecodeError, TypeError):
        if assistant.lstrip().startswith("##"):
            kind = "markdown"
    return {
        "task": task, "index": index, "total": len(lines),
        "spec_id": ex.get("spec_id"),
        "raw": json.dumps(ex, indent=2),
        "user": msgs[1]["content"], "assistant": assistant, "kind": kind,
    }


def tail_examples(root, n=5):
    gen = os.path.join(root, "data", "generated")
    candidates = []
    for task in TASK_TARGETS:
        path = os.path.join(gen, f"{task}.jsonl")
        if os.path.exists(path):
            with open(path) as f:
                lines = f.readlines()
            if len(lines) < TASK_TARGETS[task]:
                candidates.append((os.path.getmtime(path), task, lines))
    if not candidates:
        # if all done, just tail the most recently modified file
        for task in TASK_TARGETS:
            path = os.path.join(gen, f"{task}.jsonl")
            if os.path.exists(path):
                with open(path) as f:
                    lines = f.readlines()
                candidates.append((os.path.getmtime(path), task, lines))
    if not candidates:
        return []
    candidates.sort(reverse=True)
    _, task, lines = candidates[0]
    return [{"task": task, "spec_id": json.loads(l).get("spec_id"), "raw": l.rstrip()}
            for l in lines[-n:]]


def tegrastats_tail(root):
    logs = glob.glob(os.path.join(root, "**", "*.tegrastats.log"),
                     recursive=True)
    if not logs:
        return "not logging"
    try:
        lines = open(max(logs, key=os.path.getmtime)).read().strip().splitlines()
        return lines[-1] if lines else "empty log"
    except OSError:
        return "unreadable"


def disk_usage():
    try:
        out = subprocess.run(["df", "-h", "/"], capture_output=True,
                             text=True).stdout.splitlines()[-1].split()
        return {"filesystem": out[0], "size": out[1], "used": out[2],
                "avail": out[3], "pct": out[4]}
    except Exception:
        return {}


def status(root):
    tasks = dataset_progress(root)
    return {
        "now": datetime.now().isoformat(timespec="seconds"),
        "dataset": {"tasks": tasks,
                    "total_n": sum(t["n"] for t in tasks),
                    "total_t": sum(t["target"] for t in tasks)},
        "training": training_metrics(root),
        "pipeline": pipeline_state(root),
        "tegrastats": tegrastats_tail(root),
        "disk": disk_usage(),
    }


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>orzo · pipeline</title>
<style>
:root { --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3;
        --dim:#8b949e; --accent:#58a6ff; --green:#3fb950; --amber:#d29922; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text);
       font-family:-apple-system,'Segoe UI',Roboto,sans-serif; display:flex; }
nav { width:200px; min-height:100vh; background:var(--panel);
      border-right:1px solid var(--border); padding:18px 0; flex-shrink:0; }
nav h1 { font-size:19px; padding:0 18px 14px; margin:0;
         border-bottom:1px solid var(--border); }
nav h1 span { color:var(--accent); }
nav a { display:block; padding:10px 18px; color:var(--dim); cursor:pointer;
        text-decoration:none; font-size:14px; }
nav a.active { color:var(--text); background:#1f6feb22;
               border-left:3px solid var(--accent); }
main { flex:1; padding:22px 28px; max-width:1150px; }
.tab { display:none; } .tab.active { display:block; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
         gap:14px; margin-bottom:18px; }
.card { background:var(--panel); border:1px solid var(--border);
        border-radius:8px; padding:14px 16px; }
.card .kpi { font-size:26px; font-weight:600; font-family:monospace; }
.card .lbl { color:var(--dim); font-size:12px; text-transform:uppercase;
             letter-spacing:.5px; margin-bottom:6px; }
h2 { font-size:15px; color:var(--dim); text-transform:uppercase;
     letter-spacing:.6px; margin:22px 0 10px; }
table { width:100%; border-collapse:collapse; font-size:13px; }
td, th { padding:7px 10px; border-bottom:1px solid var(--border);
         text-align:left; }
th { color:var(--dim); font-weight:500; font-size:12px; }
.barbg { background:#21262d; border-radius:4px; height:8px; width:100%; }
.bar { background:var(--accent); height:8px; border-radius:4px; }
.bar.done { background:var(--green); }
pre { background:#0a0d12; border:1px solid var(--border); border-radius:8px;
      padding:14px; overflow:auto; font-size:12.5px; line-height:1.5;
      white-space:pre-wrap; word-break:break-word; max-height:520px; }
button, select { background:#21262d; color:var(--text);
      border:1px solid var(--border); border-radius:6px;
      padding:6px 14px; cursor:pointer; font-size:13px; }
button:hover { border-color:var(--accent); }
.muted { color:var(--dim); } .mono { font-family:monospace; }
.ok { color:var(--green); } .warn { color:var(--amber); }
canvas { background:#0a0d12; border:1px solid var(--border);
         border-radius:8px; width:100%; }
.pill { display:inline-block; padding:2px 10px; border-radius:20px;
        font-size:11px; background:#1f6feb33; color:var(--accent); }

/* pipeline stepper */
.step { display:flex; gap:14px; margin-bottom:18px; }
.step .dot { width:34px; height:34px; border-radius:50%; flex-shrink:0;
             display:flex; align-items:center; justify-content:center;
             font-size:15px; background:#21262d; border:2px solid var(--border); }
.step.done .dot { background:var(--green); border-color:var(--green); color:#000; }
.step.running .dot { background:var(--accent); border-color:var(--accent); color:#000;
                     animation:pulse 1.4s infinite; }
.step .body { flex:1; background:var(--panel); border:1px solid var(--border);
              border-radius:8px; padding:14px 16px; }
.step.running .body { border-color:var(--accent); }
.step .title { font-weight:600; margin-bottom:4px; }
.step .desc { color:var(--dim); font-size:13px; margin-bottom:8px; }
.step .meta { font-size:12px; color:var(--dim); margin-bottom:8px; }
.step .barbg { margin:8px 0; }
.step .cmds { background:#0a0d12; border:1px solid var(--border); border-radius:6px;
              padding:10px 14px; font-size:12px; margin-top:8px; }
.step .cmds summary { cursor:pointer; color:var(--dim); }
.step .cmds pre { margin:8px 0 0; padding:10px; max-height:160px; }
@keyframes pulse { 0%,100%{box-shadow:0 0 0 0 #58a6ff55} 50%{box-shadow:0 0 0 6px #58a6ff00} }
</style></head><body>
<nav><h1>orzo<span>·</span>pipeline</h1>
<a data-tab="pipeline" class="active">Pipeline</a>
<a data-tab="dataset">Dataset</a>
<a data-tab="examples">Examples</a>
<a data-tab="training">Training</a>
<a data-tab="system">System</a></nav>
<main>
<div id="pipeline" class="tab active">
  <h2>End-to-end procedure</h2>
  <div id="steps"></div>
</div>
<div id="dataset" class="tab">
  <div class="cards" id="dsKpis"></div>
  <div class="card"><table id="dsTasks"></table></div>
</div>
<div id="examples" class="tab">
  <div class="cards" id="feedKpis"></div>
  <h2>Live feed (newest examples from the active task)</h2>
  <pre id="liveFeed"><em>waiting for generation...</em></pre>
  <h2>Browse examples</h2>
  <p>
    <select id="exTask"></select>
    <button onclick="exNav(-1)">◀ prev</button>
    <button onclick="exNav(1)">next ▶</button>
    <button onclick="exNav(0)">random</button>
    <span class="muted" id="exMeta"></span>
  </p>
  <h3>Raw dataset record</h3><pre id="exRaw"></pre>
  <h3>User (prompt)</h3><pre id="exUser"></pre>
  <h3>Assistant (target)</h3><pre id="exAssistant"></pre>
</div>
<div id="training" class="tab">
  <div class="cards" id="trKpis"></div>
  <h2>Loss curve</h2><canvas id="lossChart" height="220"></canvas>
</div>
<div id="system" class="tab">
  <div class="cards" id="sysKpis"></div>
  <h2>tegrastats (latest)</h2><pre id="tegra"></pre>
</div>
</main>
<script>
let exIdx = 0, lastSample = null, ratePerMin = 0;
const $ = id => document.getElementById(id);
const steps = ${steps_json};

document.querySelectorAll('nav a').forEach(a => a.onclick = () => {
  document.querySelectorAll('nav a').forEach(x => x.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
  a.classList.add('active'); $(a.dataset.tab).classList.add('active');
});

function taskRows(tasks) {
  return '<tr><th>task</th><th style="width:42%">progress</th><th>count</th><th>%</th></tr>' +
    tasks.map(t => { const p = Math.min(t.n / t.target, 1);
      return `<tr><td>${t.task}</td>
        <td><div class="barbg"><div class="bar ${p>=1?'done':''}" style="width:${p*100}%"></div></div></td>
        <td class="mono">${t.n} / ${t.target}</td><td class="mono">${(p*100).toFixed(0)}%</td></tr>`;
    }).join('');
}

function kpi(label, value, cls='') {
  return `<div class="card"><div class="lbl">${label}</div><div class="kpi ${cls}">${value}</div></div>`;
}

function lineChart(cv, pts) {
  const ctx = cv.getContext('2d'), W = cv.width = cv.clientWidth * 2, H = cv.height = 440;
  ctx.clearRect(0, 0, W, H);
  if (!pts || pts.length < 2) { ctx.fillStyle = '#8b949e'; ctx.font = '28px sans-serif'; ctx.fillText('no training data yet', 30, 60); return; }
  const xs = pts.map(p => p.step), ys = pts.map(p => p.loss);
  const x0 = Math.min(...xs), x1 = Math.max(...xs) || 1;
  const y0 = Math.min(...ys), y1 = Math.max(...ys), span = (y1 - y0) || 1e-9;
  const X = s => 40 + (s - x0) / ((x1 - x0) || 1) * (W - 70);
  const Y = l => H - 40 - (l - y0) / span * (H - 80);
  ctx.strokeStyle = '#30363d'; ctx.beginPath();
  for (let i = 0; i <= 4; i++) { const y = 40 + i * (H - 80) / 4; ctx.moveTo(40, y); ctx.lineTo(W - 30, y); }
  ctx.stroke();
  ctx.strokeStyle = '#58a6ff'; ctx.lineWidth = 3; ctx.beginPath();
  pts.forEach((p, i) => i ? ctx.lineTo(X(p.step), Y(p.loss)) : ctx.moveTo(X(p.step), Y(p.loss)));
  ctx.stroke();
  ctx.fillStyle = '#8b949e'; ctx.font = '22px monospace';
  ctx.fillText(`loss ${ys[ys.length-1].toFixed(4)}  (min ${y0.toFixed(4)})`, 44, 30);
}

function renderPipeline(ps) {
  const icons = {pending:'○', running:'●', done:'✓'};
  $('steps').innerHTML = steps.map(s => {
    const st = ps[s.id] || {status:'pending', detail:'', progress:0};
    const cls = st.status;
    const bar = `<div class="barbg"><div class="bar ${st.status==='done'?'done':''}" style="width:${(st.progress||0)*100}%"></div></div>`;
    const cmds = `<details class="cmds"><summary>How to run this step</summary><pre>${s.cmds.join('\n')}</pre></details>`;
    return `<div class="step ${cls}">
      <div class="dot">${icons[st.status]}</div>
      <div class="body">
        <div class="title">${s.title}</div>
        <div class="desc">${s.desc}</div>
        <div class="meta">status: <b class="${st.status==='done'?'ok':st.status==='running'?'warn':''}">${st.status}</b> — ${st.detail}</div>
        ${st.progress > 0 || st.status==='running' ? bar : ''}
        ${cmds}
      </div>
    </div>`;
  }).join('');
}

async function poll() {
  const s = await (await fetch('/api/status')).json();
  const d = s.dataset;
  if (lastSample) ratePerMin = (d.total_n - lastSample.n) / ((Date.now() - lastSample.t) / 60000) || ratePerMin;
  lastSample = { n: d.total_n, t: Date.now() };
  const pct = (d.total_n / d.total_t * 100).toFixed(1);
  const eta = ratePerMin > 0 ? Math.round((d.total_t - d.total_n) / ratePerMin) : null;

  renderPipeline(s.pipeline);

  $('dsKpis').innerHTML =
    kpi('Examples generated', d.total_n.toLocaleString()) +
    kpi('Complete', pct + '%') +
    kpi('Rate', ratePerMin ? ratePerMin.toFixed(1) + '/min' : '—') +
    kpi('ETA', eta ? `~${eta} min` : '—');
  $('dsTasks').innerHTML = taskRows(d.tasks);

  const t = s.training;
  $('trKpis').innerHTML = t ?
    kpi('Run', `<span style="font-size:15px">${t.run}</span>`) +
    kpi('Step', `${t.step} / ${t.max_steps}`) +
    kpi('Epoch', t.epoch?.toFixed(2) ?? '—') +
    kpi('Last loss', t.losses.length ? t.losses[t.losses.length-1].loss.toFixed(4) : '—', 'ok')
    : kpi('Training', 'not started', 'warn');
  lineChart($('lossChart'), t ? t.losses : null);

  $('sysKpis').innerHTML =
    kpi('Disk free', s.disk.avail || '—') +
    kpi('Used', s.disk.pct || '—') +
    kpi('Updated', s.now.split('T')[1]);
  $('tegra').textContent = s.tegrastats;
}

async function loadExample() {
  const task = $('exTask').value;
  const e = await (await fetch(`/api/example?task=${task}&i=${exIdx}`)).json();
  exIdx = e.index;
  $('exMeta').innerHTML = ` <span class="pill">${e.spec_id}</span> ${e.index+1} / ${e.total}`;
  $('exRaw').textContent = e.raw;
  $('exUser').textContent = e.user;
  $('exAssistant').textContent = e.assistant;
}

async function updateFeed() {
  const rows = await (await fetch('/api/tail?n=8')).json();
  $('feedKpis').innerHTML = rows.length ?
    `<div class="card"><div class="lbl">Active task</div><div class="kpi">${rows[0].task}</div></div>` +
    `<div class="card"><div class="lbl">Latest spec id</div><div class="kpi" style="font-size:14px">${rows[rows.length-1].spec_id}</div></div>` : '';
  $('liveFeed').textContent = rows.map(r => `[${r.task}] ${r.spec_id}: ${r.raw}`).join('\n');
}

function exNav(d) {
  if (d === 0) exIdx = Math.floor(Math.random() * 1e6);
  else exIdx = Math.max(0, exIdx + d);
  loadExample();
}

window.onload = () => {
  $('exTask').innerHTML = Object.keys(${task_targets}).map(t => `<option>${t}</option>`).join('');
  $('exTask').onchange = () => { exIdx = 0; loadExample(); };
  poll(); loadExample(); updateFeed();
  setInterval(poll, 5000);
  setInterval(updateFeed, 5000);
};
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    root = "."

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            body = PAGE.replace("${task_targets}", json.dumps(TASK_TARGETS)) \
                       .replace("${steps_json}", json.dumps(PIPELINE)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/api/status":
            self._json(status(self.root))
        elif url.path == "/api/example":
            q = parse_qs(url.query)
            task = q.get("task", ["harness_scaffold"])[0]
            index = int(q.get("i", ["0"])[0])
            ex = read_example(self.root, task, index)
            self._json(ex or {"error": "no examples yet"},
                       200 if ex else 404)
        elif url.path == "/api/tail":
            q = parse_qs(url.query)
            n = int(q.get("n", ["5"])[0])
            self._json(tail_examples(self.root, n))
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, *args):
        pass


def main():
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
