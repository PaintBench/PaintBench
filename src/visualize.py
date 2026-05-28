"""PaintBench — interactive web visualizer.

Usage:
    python src/visualize.py [--port 8765]
    python src/visualize.py --benchmark benchmarks/PaintBench [--port 8765]
    python src/visualize.py --benchmarks benchmarks \\
                            --model-outputs model_outputs \\
                            --eval-outputs eval_outputs [--port 8765]

Open http://localhost:8765 in a browser.

Three view modes
----------------
Generate   : Generate problems on the fly for any task/seed/params.
             Seed grid or param sweep.
Benchmark  : Browse pre-generated benchmark problems from disk.
             Requires --benchmark DIR pointing to one of the generated
             benchmark dirs (PaintBench / TinyGrafixBench).
Eval       : Compare model outputs against ground truth (CIE76 diff).
             Requires --benchmarks + --model-outputs. --eval-outputs is
             optional and supplies pre-computed stats from src/eval.py
             (problem_stats.jsonl) plus the cached ΔE diff PNGs that
             `make eval` writes by default. Anything missing is
             computed on-demand and cached to disk for next time.

Eval-mode disk layout (post model_outputs/ ⟷ eval_outputs/ split):
    --benchmarks    benchmarks/<bench>/<task>/<NNN>_input.png   (3-digit)
                                             <NNN>_answer.png
                                             <NNN>.json
    --model-outputs model_outputs/<model>/<bench>/<task>/<NNNN>_output.png  (4-digit)
    --eval-outputs  eval_outputs/problem_stats.jsonl                (cached stats)
                    eval_outputs/<model>/<bench>/<task>/<NNNN>_diff_cie76_<t>.png  (4 thresholds)
                                                       <NNNN>_normalized_output.png  (only if differs)
                    (per-problem PNGs populated by `make eval` — pass
                     --no-save-images / use `make eval-quick` to skip)
"""
from __future__ import annotations
import base64
import io
import json
import os
import sys
import importlib
import http.server
import socketserver
import threading
import urllib.parse
from collections import OrderedDict
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import matplotlib
matplotlib.use("Agg")

import random as _random
from generate_benchmark import TASKS, _color_split, _striped_bg
from core.background import BackgroundSpec
from core.colors import STANDARD_PALETTE, NONSTANDARD_PALETTE

# Eval helpers (imported lazily to avoid hard dependency when not used)
def _get_eval():
    """Return the eval module (imports numpy/PIL on first call)."""
    import eval as _eval
    return _eval

_TASK_MODULES: dict[str, str] = {name: mod for mod, name in TASKS}

# tinygrafixbench tasks — exposed via "tgf:<graph>.<task>" keys.
_TGF_GRAPHS = ["bar_chart", "scatter_plot", "line_chart", "heatmap", "network"]


def _tgf_task_keys() -> list[tuple[str, str]]:
    """Return (key, label) pairs for every tinygrafixbench graph/task."""
    pairs: list[tuple[str, str]] = []
    for g in _TGF_GRAPHS:
        mod = importlib.import_module(f"tinygrafixbench.{g}")
        for t in mod.TASKS:
            pairs.append((f"tgf:{g}.{t}", f"[TGB] {g.replace('_', ' ').title()} · {t.replace('_', ' ').title()}"))
    return pairs


def _fig_to_b64(fig) -> str:
    """PNG-encode a matplotlib Figure and close it."""
    import matplotlib.pyplot as _plt
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG")
    _plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

_DEFAULT_W = 1024
_DEFAULT_H = 1024
_BENCHMARK_DIR:    str | None = None   # set by --benchmark CLI flag
_MODEL_OUTPUTS_DIR: str | None = None  # set by --model-outputs CLI flag
_BENCHMARKS_ROOT:   str | None = None  # set by --benchmarks CLI flag
_RESULTS_DIR:       str | None = None  # set by --eval-outputs CLI flag (pre-computed eval output)

# ΔE thresholds the Eval card renders (must match `_DE_THRESHOLDS` in the JS).
# Restricting live-compute to these four cuts /eval_problem latency by
# ~190 ms per request vs iterating range(11).
_LIVE_DIFF_THRESHOLDS: tuple[int, ...] = (0, 2, 5, 10)


def _load_module(task_name: str):
    return importlib.import_module(_TASK_MODULES[task_name])


def _img_to_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ---------------------------------------------------------------------------
# HTML — uses __TASK_OPTIONS__ and __TASK_PARAMS__ as simple string tokens
#        so that Python .format() is never called (avoids conflict with JS
#        brace syntax, including spread operators like {...obj}).
# ---------------------------------------------------------------------------

_SIZE_OPTIONS = "".join(
    f'<option value="{s}"{" selected" if s == 1024 else ""}>{s}</option>'
    for s in range(256, 1025, 64)
)

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PaintBench Visualizer</title>
<style>
*{box-sizing:border-box}
body{font-family:system-ui,sans-serif;background:#0e0e1a;color:#eee;margin:0;padding:0}
header{background:#12192d;padding:12px 22px;border-bottom:2px solid #1e3a5f;
       display:flex;align-items:center;gap:14px}
header h1{margin:0;font-size:1.3rem;color:#e94560}
header span{color:#666;font-size:.82rem}
.mode-tabs{display:flex;gap:0;padding:0 22px;background:#12192d;
           border-bottom:2px solid #1e3a5f}
.mode-tab{background:none;border:none;border-bottom:3px solid transparent;
          margin-bottom:-2px;padding:9px 20px;color:#666;cursor:pointer;
          font-size:.82rem;font-weight:600;letter-spacing:.04em;transition:color .15s}
.mode-tab.active{color:#e94560;border-bottom-color:#e94560}
.mode-tab:hover:not(.active){color:#aaa}
.mode-tab:disabled{color:#333;cursor:default}
.controls{padding:12px 22px;background:#12192d;display:flex;flex-wrap:wrap;
          gap:10px;align-items:flex-end;border-bottom:1px solid #1e3a5f}
.ctrl-group{display:flex;flex-direction:column;gap:3px}
label{font-size:.68rem;color:#888;text-transform:uppercase;letter-spacing:.05em}
select,input[type=number]{background:#08101f;color:#eee;border:1px solid #1e3a5f;
  border-radius:4px;padding:4px 8px;font-size:.85rem}
.btn{background:#e94560;color:#fff;border:none;border-radius:4px;padding:6px 16px;
     cursor:pointer;font-size:.85rem;font-weight:600;align-self:flex-end}
.btn:hover{background:#c73652}.btn:disabled{background:#444;cursor:default}
.divider{width:1px;background:#1e3a5f;align-self:stretch;margin:0 4px}
.loader{display:none;align-items:center;gap:10px;min-width:240px;max-width:420px;
        align-self:flex-end;padding-bottom:5px}
.loader-label{font-size:.72rem;color:#e94560;font-weight:600;white-space:nowrap;
              letter-spacing:.04em}
.progress-bar{flex:1;height:6px;background:#08101f;border:1px solid #1e3a5f;
              border-radius:3px;overflow:hidden}
.progress-fill{height:100%;background:#e94560;width:0%;transition:width .15s ease}
.progress-text{font-size:.7rem;color:#888;white-space:nowrap;font-variant-numeric:tabular-nums}
.sweep-row{padding:7px 22px;background:#11172b;border-bottom:1px solid #1e3a5f;
           display:flex;align-items:center;gap:10px}
.sweep-row label{font-size:.68rem;color:#888;text-transform:uppercase;
                 letter-spacing:.05em;white-space:nowrap}
.sweep-hint{font-size:.72rem;color:#444}
#resultArea{padding:14px 22px}
.grid-wrap{display:grid;gap:12px}
.card{background:#12192d;border:1px solid #1e3a5f;border-radius:8px;overflow:hidden}
.card-head{padding:6px 10px;border-bottom:1px solid #1e3a5f;
           display:flex;align-items:center;flex-wrap:wrap;gap:5px}
.seed-badge{background:#08101f;border:1px solid #1e3a5f;border-radius:4px;
            padding:2px 6px;font-size:.68rem;color:#666;font-family:monospace}
.pal-badge{background:#0a1a0a;border:1px solid #1e4a1e;border-radius:4px;
           padding:2px 6px;font-size:.68rem;color:#6a6;font-family:monospace}
.chip{background:#08101f;border-radius:20px;padding:2px 8px;font-size:.7rem;
      border:1px solid #1e3a5f;display:flex;gap:3px}
.chip .k{color:#555}.chip .v{color:#b0d0f0;font-weight:600}
.chip.swept{border-color:#e94560}.chip.swept .v{color:#e94560}
.instr{padding:6px 10px;font-size:.78rem;color:#ccc;border-bottom:1px solid #1e3a5f;
       line-height:1.45}
.imgs{display:flex;gap:5px;padding:7px}
.iw{flex:1;display:flex;flex-direction:column;align-items:center;gap:2px}
.iw span{font-size:.62rem;color:#444;text-transform:uppercase;letter-spacing:.04em}
.iw img{width:100%;border-radius:3px}
.diff-placeholder{width:100%;aspect-ratio:1/1;background:#08101f;border-radius:3px;
                  display:flex;align-items:center;justify-content:center;
                  color:#333;font-size:.65rem}
.err{background:#2a0010;border:1px solid #e94560;border-radius:8px;
     padding:10px;font-family:monospace;font-size:.72rem;white-space:pre-wrap;color:#f88}
.sweep-hdr{background:#0f3460;border-radius:5px;padding:6px 10px;text-align:center;
           font-size:.72rem;color:#e94560;font-weight:700;text-transform:uppercase;
           letter-spacing:.06em}
.bench-msg{color:#666;font-size:.82rem;align-self:center;padding:4px 0}
.bench-toggle-wrap{display:flex;align-items:center;gap:5px;font-size:.78rem;color:#888;
  cursor:pointer;align-self:flex-end;padding-bottom:6px;user-select:none}
.bench-toggle-wrap input{accent-color:#e94560;width:14px;height:14px;cursor:pointer}
.model-outputs{padding:5px 7px;border-top:1px solid #1e3a5f}
.model-outputs-row{display:grid;grid-template-columns:repeat(auto-fill,minmax(100px,1fr));gap:6px}
.model-out-wrap{display:flex;flex-direction:column;align-items:center;gap:3px}
.model-out-wrap span{font-size:.58rem;color:#555;word-break:break-all;text-align:center;line-height:1.3}
.model-out-wrap img{width:100%;aspect-ratio:1/1;object-fit:contain;border-radius:3px;background:#080f1e}
.eval-nav{display:flex;align-items:center;gap:8px;padding:6px 22px;background:#12192d;
          border-bottom:1px solid #1e3a5f;flex-wrap:wrap}
.eval-nav span{font-size:.78rem;color:#888}
.eval-prob-grid{display:grid;gap:14px;padding:14px 22px}
.eval-prob-card{background:#12192d;border:1px solid #1e3a5f;border-radius:8px;overflow:hidden}
.eval-prob-head{padding:6px 10px;border-bottom:1px solid #1e3a5f;font-size:.78rem;color:#aaa;
                display:flex;gap:10px;align-items:center}
.eval-img-row{display:flex;gap:5px;padding:7px;overflow-x:auto}
.eval-iw{flex:0 0 auto;display:flex;flex-direction:column;align-items:center;gap:2px;min-width:0}
.eval-iw span{font-size:.58rem;color:#444;text-transform:uppercase;letter-spacing:.04em;white-space:nowrap}
.eval-iw img{width:140px;height:140px;object-fit:contain;border-radius:3px;background:#080f1e}
.eval-iw.lg img{width:200px;height:200px}
.eval-stats-wrap{padding:6px 10px;overflow-x:auto}
.eval-stats-tbl{border-collapse:collapse;font-size:.7rem;width:100%}
.eval-stats-tbl th{text-align:left;color:#666;padding:2px 8px;border-bottom:1px solid #1e3a5f;
                   font-weight:400;white-space:nowrap}
.eval-stats-tbl td{padding:2px 8px;color:#ccc;white-space:nowrap}
.eval-stats-tbl tr:hover td{background:#0a1529}
.eval-agg{padding:14px 22px}
.eval-agg h3{color:#e94560;font-size:.85rem;margin:10px 0 5px}
.eval-miss{color:#555;font-size:.8rem;padding:14px 22px}
.eval-img-section{padding:2px 7px 0;font-size:.6rem;color:#555;text-transform:uppercase;letter-spacing:.05em}
</style>
</head>
<body>
<header>
  <h1>PaintBench</h1>
  <span>deterministic evaluation of precise visual editing</span>
</header>

<div class="mode-tabs">
  <button class="mode-tab active" id="tabGen"   onclick="setMode('gen')">Generate</button>
  <button class="mode-tab"        id="tabBench" onclick="setMode('bench')">Benchmark</button>
  <button class="mode-tab"        id="tabEval"  onclick="setMode('eval')">Eval</button>
</div>

<!-- ── Generate panel ── -->
<div id="genPanel" class="controls">
  <div class="ctrl-group">
    <label>Task</label>
    <select id="taskSel" onchange="onTaskChange()">__TASK_OPTIONS__</select>
  </div>
  <div class="ctrl-group">
    <label>Start seed</label>
    <input type="number" id="seedIn" value="42" min="0" max="999999" style="width:80px">
  </div>
  <div class="ctrl-group">
    <label>Count</label>
    <input type="number" id="cntIn" value="10" min="1" max="100" style="width:55px">
  </div>
  <div class="ctrl-group">
    <label>Width</label>
    <select id="widthSel">__SIZE_OPTIONS__</select>
  </div>
  <div class="ctrl-group">
    <label>Height</label>
    <select id="heightSel">__SIZE_OPTIONS__</select>
  </div>
  <div class="divider"></div>
  <div class="ctrl-group">
    <label>Palette</label>
    <select id="palSel">
      <option value="random">(random)</option>
      <option value="standard" selected>standard</option>
      <option value="nonstandard">nonstandard</option>
    </select>
  </div>
  <div class="ctrl-group">
    <label>Background</label>
    <select id="bgSel">
      <option value="random">(random)</option>
      <option value="solid" selected>solid</option>
      <option value="striped">striped</option>
    </select>
  </div>
  <div class="divider"></div>
  <div id="paramCtrl" style="display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end"></div>
  <div class="divider"></div>
  <button class="btn" id="genBtn" onclick="doGenerate()">Generate &#9654;</button>
  <div class="loader" id="genLoader">
    <span class="loader-label">Generating</span>
    <div class="progress-bar"><div class="progress-fill"></div></div>
    <span class="progress-text"></span>
  </div>
</div>

<!-- ── Benchmark panel ── -->
<div id="benchPanel" class="controls" style="display:none">
  <div class="ctrl-group" id="benchBenchGroup" style="display:none">
    <label>Benchmark</label>
    <select id="benchBenchSel" onchange="onBenchBenchChange()"></select>
  </div>
  <div class="ctrl-group">
    <label>Task</label>
    <select id="benchTaskSel">
      <option value="">(all tasks)</option>
    </select>
  </div>
  <label class="bench-toggle-wrap" title="Show each model's output under every problem">
    <input type="checkbox" id="benchShowModels">
    <span>Model outputs</span>
  </label>
  <button class="btn" id="benchBtn" onclick="doBenchmark()">Load &#9654;</button>
  <div class="loader" id="benchLoader">
    <span class="loader-label">Loading</span>
    <div class="progress-bar"><div class="progress-fill"></div></div>
    <span class="progress-text"></span>
  </div>
  <span class="bench-msg" id="benchMsg"></span>
</div>

<!-- ── Eval panel ── -->
<div id="evalPanel" class="controls" style="display:none">
  <div class="ctrl-group">
    <label>Model</label>
    <select id="eModelSel" onchange="onEvalModelChange()"></select>
  </div>
  <div class="ctrl-group">
    <label>Benchmark</label>
    <select id="eBenchSel" onchange="onEvalBenchChange()"></select>
  </div>
  <div class="ctrl-group">
    <label>Task</label>
    <select id="eTaskSel" onchange="onEvalTaskChange()">
      <option value="">(all tasks)</option>
    </select>
  </div>
  <div class="ctrl-group">
    <label>Problem</label>
    <select id="eProbSel">
      <option value="">(all)</option>
    </select>
  </div>
  <button class="btn" id="eLoadBtn" onclick="doEvalLoad()">Load &#9654;</button>
  <button class="btn" id="eStatsBtn" onclick="doEvalAgg()" style="background:#1e3a5f">Agg Stats &#9654;</button>
  <div class="loader" id="eLoader">
    <span class="loader-label">Loading</span>
    <div class="progress-bar"><div class="progress-fill"></div></div>
    <span class="progress-text"></span>
  </div>
</div>

<!-- ── Sweep row (generate mode only) ── -->
<div id="sweepRow" class="sweep-row">
  <label>Sweep:</label>
  <select id="sweepSel" onchange="onSweepChange()">
    <option value="">(none &mdash; seed grid)</option>
  </select>
  <span class="sweep-hint" id="sweepHint"></span>
</div>

<div id="resultArea"></div>

<script>
const TP = __TASK_PARAMS__;
const TGB_LABELS = __TGB_TASK_LABELS__;

// ── Mode switching ──────────────────────────────────────────────────────────

let _benchIndex = null;  // cached /bench_index response
let _activeBench = null; // currently selected benchmark name

async function initBenchmark() {
  if (_benchIndex !== null) return;

  // Fetch available benchmarks
  let benchList = { available: false, benchmarks: [] };
  try {
    const r = await fetch('/bench_list');
    benchList = await r.json();
  } catch(e) {}

  if (benchList.available && benchList.benchmarks.length > 1) {
    // Show benchmark selector
    const group = document.getElementById('benchBenchGroup');
    group.style.display = '';
    const sel = document.getElementById('benchBenchSel');
    sel.innerHTML = '';
    for (const b of benchList.benchmarks) {
      const o = document.createElement('option');
      o.value = b; o.textContent = b;
      sel.appendChild(o);
    }
  }

  // Load first (or only) benchmark
  const first = benchList.benchmarks.length > 0 ? benchList.benchmarks[0] : null;
  await _loadBenchIndex(first);
}

async function onBenchBenchChange() {
  const bench = document.getElementById('benchBenchSel').value;
  _benchIndex = null;
  await _loadBenchIndex(bench);
}

async function _loadBenchIndex(bench) {
  _activeBench = bench;
  const url = bench ? '/bench_index?benchmark=' + encodeURIComponent(bench) : '/bench_index';
  try {
    const r = await fetch(url);
    _benchIndex = await r.json();
  } catch(e) {
    _benchIndex = { available: false, problems: [] };
  }

  const tab = document.getElementById('tabBench');
  if (!_benchIndex.available) {
    tab.disabled = true;
    tab.title = 'No benchmark directory — restart with --benchmark DIR or --benchmarks DIR';
    return;
  }

  // Repopulate task dropdown from problems.jsonl (task = folder name).
  const tasks = [...new Set(
    _benchIndex.problems.map(p => p.task)
  )].sort();
  const sel = document.getElementById('benchTaskSel');
  sel.innerHTML = '<option value="">(all tasks)</option>';
  for (const t of tasks) {
    const o = document.createElement('option');
    o.value = t;
    o.textContent = TGB_LABELS[t] ?? t.replace(/_/g,' ').replace(/\b\w/g, c => c.toUpperCase());
    sel.appendChild(o);
  }
}

function setMode(mode) {
  document.getElementById('tabGen').classList.toggle('active',   mode === 'gen');
  document.getElementById('tabBench').classList.toggle('active', mode === 'bench');
  document.getElementById('tabEval').classList.toggle('active',  mode === 'eval');
  document.getElementById('genPanel').style.display   = mode === 'gen'   ? '' : 'none';
  document.getElementById('benchPanel').style.display = mode === 'bench' ? '' : 'none';
  document.getElementById('evalPanel').style.display  = mode === 'eval'  ? '' : 'none';
  document.getElementById('sweepRow').style.display   = mode === 'gen'   ? '' : 'none';
  document.getElementById('resultArea').innerHTML = '';
  if (mode === 'bench') initBenchmark();
  if (mode === 'eval')  initEval();
}

// ── Task / param controls (generate mode) ──────────────────────────────────

function onTaskChange() {
  const task = document.getElementById('taskSel').value;
  const params = TP[task] || {};
  const pc = document.getElementById('paramCtrl');
  pc.innerHTML = '';
  for (const [k, vs] of Object.entries(params)) {
    const g = document.createElement('div');
    g.className = 'ctrl-group'; g.id = 'pg_' + k;
    const lbl = document.createElement('label');
    lbl.textContent = k; g.appendChild(lbl);
    if (k === 'n_min' || k === 'n_max') {
      const inp = document.createElement('input');
      inp.type = 'number'; inp.id = 'p_' + k;
      inp.value = '3'; inp.min = '1'; inp.max = '999';
      inp.style = 'width:60px';
      g.appendChild(inp);
    } else {
      const sel = document.createElement('select');
      sel.id = 'p_' + k;
      const o0 = document.createElement('option');
      o0.value = ''; o0.textContent = '(random)'; sel.appendChild(o0);
      for (const v of vs) {
        const o = document.createElement('option');
        o.value = v; o.textContent = v; sel.appendChild(o);
      }
      g.appendChild(sel);
    }
    pc.appendChild(g);
  }
  const sw = document.getElementById('sweepSel');
  sw.innerHTML = '<option value="">(none &mdash; seed grid)</option>';
  for (const k of Object.keys(params)) {
    const o = document.createElement('option');
    o.value = k; o.textContent = k; sw.appendChild(o);
  }
  sw.value = ('mode' in params) ? 'mode' : '';
  onSweepChange();
}

function onSweepChange() {
  const task = document.getElementById('taskSel').value;
  const params = TP[task] || {};
  const sw = document.getElementById('sweepSel').value;
  for (const k of Object.keys(params)) {
    const g = document.getElementById('pg_' + k);
    if (g) g.style.display = (sw && k === sw) ? 'none' : '';
  }
  const vals = sw ? (params[sw] || []) : [];
  document.getElementById('sweepHint').textContent = sw ? vals.length + ' values' : '';
}

// ── Generate ────────────────────────────────────────────────────────────────

async function doGenerate() {
  const task   = document.getElementById('taskSel').value;
  const seed0  = parseInt(document.getElementById('seedIn').value) || 0;
  const count  = Math.max(1, Math.min(100, parseInt(document.getElementById('cntIn').value) || 1));
  const W      = document.getElementById('widthSel').value;
  const H      = document.getElementById('heightSel').value;
  const pal    = document.getElementById('palSel').value;
  const bg     = document.getElementById('bgSel').value;
  const sw     = document.getElementById('sweepSel').value;
  const params = TP[task] || {};

  const pinned = {};
  for (const k of Object.keys(params)) {
    if (k === sw) continue;
    const el = document.getElementById('p_' + k);
    if (!el) continue;
    if (k === 'n_min' || k === 'n_max') {
      pinned[k] = el.value || '3';
    } else if (el.value) {
      pinned[k] = el.value;
    }
  }

  let jobs = [];
  if (!sw) {
    for (let i = 0; i < count; i++)
      jobs.push({ seed: seed0 + i, params: Object.assign({}, pinned), col: -1 });
  } else {
    const svs = params[sw] || [];
    for (let c = 0; c < svs.length; c++) {
      for (let i = 0; i < count; i++) {
        const p = Object.assign({}, pinned);
        p[sw] = svs[c];
        jobs.push({ seed: seed0 + i, params: p, col: c, sv: svs[c] });
      }
    }
  }

  document.getElementById('genBtn').disabled = true;
  loaderStart('genLoader', 'Generating', jobs.length);
  document.getElementById('resultArea').innerHTML = '';

  try {
    const results = await fetchBatched(jobs, j =>
      fetch('/gen?' + new URLSearchParams({
        task, seed: j.seed, params: JSON.stringify(j.params), W, H, palette: pal, background: bg
      }))
        .then(r => r.json())
        .then(d => Object.assign({}, d, j))
        .catch(e => Object.assign({ error: String(e) }, j))
    , 6, (d) => loaderUpdate('genLoader', d));
    loaderStop('genLoader');
    document.getElementById('genBtn').disabled = false;
    renderGrid(results, sw, params[sw] || [], count);
  } catch (e) {
    loaderStop('genLoader');
    document.getElementById('genBtn').disabled = false;
    document.getElementById('resultArea').innerHTML =
      '<div class="err">' + e + '</div>';
  }
}

// ── Benchmark ───────────────────────────────────────────────────────────────

async function doBenchmark() {
  if (!_benchIndex || !_benchIndex.available) {
    document.getElementById('benchMsg').textContent = 'No benchmark loaded.';
    return;
  }

  const filterTask = document.getElementById('benchTaskSel').value;
  let problems = _benchIndex.problems;
  if (filterTask) problems = problems.filter(p => p.task === filterTask);

  if (!problems.length) {
    document.getElementById('benchMsg').textContent = 'No problems match filters.';
    return;
  }

  // Resolve models list before loading problems so cards render complete.
  const showModels = document.getElementById('benchShowModels').checked;
  let benchModels = [];
  if (showModels) {
    if (_evalIndex && _evalIndex.models) {
      benchModels = _evalIndex.models;
    } else {
      try {
        const r = await fetch('/eval_index');
        const idx = await r.json();
        if (_evalIndex === null) _evalIndex = idx;
        benchModels = idx.models || [];
      } catch(e) {}
    }
  }

  document.getElementById('benchBtn').disabled = true;
  loaderStart('benchLoader', 'Loading', problems.length);
  document.getElementById('benchMsg').textContent = '';
  document.getElementById('resultArea').innerHTML = '';

  try {
    const mkParams = p => {
      const q = { task: p.task, problem_id: p.problem_id };
      if (_activeBench) q.benchmark = _activeBench;
      return q;
    };
    const results = await fetchBatched(problems, p =>
      fetch('/bench_get?' + new URLSearchParams(mkParams(p)))
        .then(r => r.json())
        .then(d => Object.assign({}, d, { bench_meta: p }))
        .catch(e => ({ error: String(e), bench_meta: p }))
    , 8, (d) => loaderUpdate('benchLoader', d));
    loaderStop('benchLoader');
    document.getElementById('benchBtn').disabled = false;
    document.getElementById('benchMsg').textContent =
      results.length + ' problem' + (results.length !== 1 ? 's' : '');
    renderBenchGrid(results, benchModels);
  } catch (e) {
    loaderStop('benchLoader');
    document.getElementById('benchBtn').disabled = false;
    document.getElementById('resultArea').innerHTML =
      '<div class="err">' + e + '</div>';
  }
}

function renderBenchGrid(results, models = []) {
  const area = document.getElementById('resultArea');
  area.innerHTML = '';
  const n = results.length;
  const cols = n <= 1 ? 1 : n <= 2 ? 2 : n <= 4 ? 2 : 3;
  const wrap = document.createElement('div');
  wrap.className = 'grid-wrap';
  wrap.style.gridTemplateColumns = 'repeat(' + cols + ',1fr)';
  for (const r of results) wrap.appendChild(makeBenchCard(r, models));
  area.appendChild(wrap);
}

function makeBenchCard(d, models = []) {
  if (!d || d.error) {
    const e = document.createElement('div');
    e.className = 'err';
    e.textContent = (d && d.error) || 'Unknown error';
    return e;
  }
  const m = d.bench_meta || {};
  const augmented = Object.assign({}, d, {
    seed: m.seed,
    params: Object.assign({ palette: m.palette, problem_id: m.problem_id }, d.params || {}),
  });
  const card = makeCard(augmented, null);

  if (models.length) {
    const bench = _activeBench || '';
    const task  = m.task || '';
    const pid   = m.problem_id ?? 0;
    const section = document.createElement('div');
    section.className = 'model-outputs';
    const row = document.createElement('div');
    row.className = 'model-outputs-row';
    for (const model of models) {
      const url = '/eval_image?' + new URLSearchParams({model, benchmark: bench, task, idx: pid, kind: 'output'});
      const mw = document.createElement('div');
      mw.className = 'model-out-wrap';
      const img = document.createElement('img');
      img.src = url; img.loading = 'lazy';
      img.onerror = () => { img.style.opacity = '0.15'; };
      const lbl = document.createElement('span');
      lbl.textContent = model;
      mw.appendChild(img); mw.appendChild(lbl);
      row.appendChild(mw);
    }
    section.appendChild(row);
    card.appendChild(section);
  }
  return card;
}

// ── Grid rendering ──────────────────────────────────────────────────────────

function renderGrid(results, sw, svs, count) {
  const area = document.getElementById('resultArea');
  area.innerHTML = '';
  if (!sw) {
    const n = results.length;
    const cols = n <= 1 ? 1 : n <= 2 ? 2 : n <= 4 ? 2 : 3;
    const wrap = document.createElement('div');
    wrap.className = 'grid-wrap';
    wrap.style.gridTemplateColumns = 'repeat(' + cols + ',1fr)';
    for (const r of results) wrap.appendChild(makeCard(r, null));
    area.appendChild(wrap);
  } else {
    const nc = svs.length;
    const wrap = document.createElement('div');
    wrap.className = 'grid-wrap';
    wrap.style.gridTemplateColumns = 'repeat(' + nc + ',1fr)';
    for (const v of svs) {
      const h = document.createElement('div');
      h.className = 'sweep-hdr';
      h.textContent = sw + ': ' + v;
      wrap.appendChild(h);
    }
    for (let i = 0; i < count; i++)
      for (let c = 0; c < nc; c++)
        wrap.appendChild(makeCard(results[c * count + i], sw));
    area.appendChild(wrap);
  }
}

// ── Card ────────────────────────────────────────────────────────────────────

function makeCard(d, sweepParam) {
  if (!d || d.error) {
    const e = document.createElement('div');
    e.className = 'err';
    e.textContent = (d && d.error) || 'Unknown error';
    return e;
  }
  const card = document.createElement('div');
  card.className = 'card';

  // Header
  const hd = document.createElement('div');
  hd.className = 'card-head';
  if (d.seed !== undefined) {
    const sb = document.createElement('span');
    sb.className = 'seed-badge';
    sb.textContent = 'seed ' + d.seed;
    hd.appendChild(sb);
  }
  for (const [k, v] of Object.entries(d.params || {})) {
    const ch = document.createElement('span');
    ch.className = 'chip' + (k === sweepParam ? ' swept' : '');
    ch.innerHTML = '<span class="k">' + k + '</span><span class="v">' + v + '</span>';
    hd.appendChild(ch);
  }
  card.appendChild(hd);

  // Instruction
  const ins = document.createElement('div');
  ins.className = 'instr';
  ins.textContent = d.instruction;
  card.appendChild(ins);

  // Images + diff
  const imgs = document.createElement('div');
  imgs.className = 'imgs';

  // Input
  const inWrap = document.createElement('div');
  inWrap.className = 'iw';
  const inImg = document.createElement('img');
  inImg.src = 'data:image/png;base64,' + d.input_b64;
  inImg.loading = 'lazy';
  const inLbl = document.createElement('span'); inLbl.textContent = 'Input';
  inWrap.appendChild(inImg); inWrap.appendChild(inLbl);

  // Answer
  const ansWrap = document.createElement('div');
  ansWrap.className = 'iw';
  const ansImg = document.createElement('img');
  ansImg.src = 'data:image/png;base64,' + d.answer_b64;
  ansImg.loading = 'lazy';
  const ansLbl = document.createElement('span'); ansLbl.textContent = 'Answer';
  ansWrap.appendChild(ansImg); ansWrap.appendChild(ansLbl);

  // Diff (computed asynchronously)
  const diffWrap = document.createElement('div');
  diffWrap.className = 'iw';
  const diffPH = document.createElement('div');
  diffPH.className = 'diff-placeholder';
  diffPH.textContent = 'computing diff\u2026';
  const diffLbl = document.createElement('span'); diffLbl.textContent = 'Diff';
  diffWrap.appendChild(diffPH); diffWrap.appendChild(diffLbl);

  imgs.appendChild(inWrap);
  imgs.appendChild(ansWrap);
  imgs.appendChild(diffWrap);
  card.appendChild(imgs);

  // Kick off diff computation
  computeDiff(d.input_b64, d.answer_b64).then(url => {
    const di = document.createElement('img');
    di.src = url; di.style.width = '100%'; di.style.borderRadius = '3px';
    diffWrap.replaceChild(di, diffPH);
  }).catch(() => { diffPH.textContent = 'diff error'; });

  return card;
}

// ── Pixel diff ──────────────────────────────────────────────────────────────
// Changed pixels  → bright green (#14dc3c)
// Unchanged pixels → very dim version of the input image (25% brightness)

function _loadImgData(b64) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement('canvas');
      c.width = img.width; c.height = img.height;
      c.getContext('2d').drawImage(img, 0, 0);
      resolve(c.getContext('2d').getImageData(0, 0, c.width, c.height));
    };
    img.onerror = reject;
    img.src = 'data:image/png;base64,' + b64;
  });
}

async function computeDiff(inputB64, answerB64) {
  const [d1, d2] = await Promise.all([
    _loadImgData(inputB64),
    _loadImgData(answerB64),
  ]);
  const W = d1.width, H = d1.height;
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const ctx = c.getContext('2d');
  const out = ctx.createImageData(W, H);
  const od = out.data, a = d1.data, b = d2.data;
  for (let i = 0; i < a.length; i += 4) {
    const diff = Math.abs(a[i]-b[i]) + Math.abs(a[i+1]-b[i+1]) + Math.abs(a[i+2]-b[i+2]);
    if (diff > 10) {
      // changed → green
      od[i] = 20; od[i+1] = 220; od[i+2] = 60; od[i+3] = 255;
    } else {
      // unchanged → dim input
      od[i] = a[i]>>2; od[i+1] = a[i+1]>>2; od[i+2] = a[i+2]>>2; od[i+3] = 255;
    }
  }
  ctx.putImageData(out, 0, 0);
  return c.toDataURL('image/png');
}

// ── Batched fetch ────────────────────────────────────────────────────────────

async function fetchBatched(items, fetchFn, batchSize, onProgress, onItem) {
  // onProgress(done, total) fires once per item completion (not per batch),
  // so a progress bar wired to it advances smoothly even within a batch of
  // concurrent requests.
  // onItem(result, globalIdx) (optional) fires per item with the original
  // index in `items`, so callers can stream results into the UI as they
  // arrive instead of waiting for the whole batch.
  const results = new Array(items.length);
  let done = 0;
  const total = items.length;
  for (let i = 0; i < total; i += batchSize) {
    const batch = items.slice(i, i + batchSize);
    await Promise.all(batch.map(async (item, j) => {
      const globalIdx = i + j;
      const r = await fetchFn(item);
      results[globalIdx] = r;
      done++;
      if (onProgress) onProgress(done, total);
      if (onItem)     onItem(r, globalIdx);
    }));
  }
  return results;
}

// ── Loader / progress bar ────────────────────────────────────────────────────
// Three modes share the same progress UI: a label + thin bar + "N / M (P%) ·
// ETA Xs" text. ETA is computed from (elapsed / done) × remaining and only
// shown once at least one request has completed (so the first paint isn't
// "ETA Infinity"). For single-shot operations (Agg Stats), passing total ≤ 1
// shows just the label with an ellipsis — no bar fill / ETA noise.

const _loaderState = {};

function loaderStart(id, label, total) {
  _loaderState[id] = { start: performance.now(), total: total, label: label };
  document.getElementById(id).style.display = 'flex';
  loaderUpdate(id, 0);
}

function loaderUpdate(id, done) {
  const el = document.getElementById(id);
  const s = _loaderState[id];
  if (!el || !s) return;
  const lbl  = el.querySelector('.loader-label');
  const fill = el.querySelector('.progress-fill');
  const txt  = el.querySelector('.progress-text');
  if (s.total <= 1) {
    lbl.textContent = s.label + '\u2026';
    fill.style.width = (done >= s.total ? 100 : 0) + '%';
    txt.textContent = '';
    return;
  }
  const pct = (done / s.total) * 100;
  const remaining = s.total - done;
  let etaText = '';
  if (done > 0 && remaining > 0) {
    const eta = (performance.now() - s.start) / done * remaining;
    etaText = ' \u00b7 ETA ' + _fmtMs(eta);
  }
  lbl.textContent = s.label;
  fill.style.width = pct + '%';
  txt.textContent = `${done} / ${s.total} (${pct.toFixed(0)}%)` + etaText;
}

function loaderStop(id) {
  document.getElementById(id).style.display = 'none';
  delete _loaderState[id];
}

function _fmtMs(ms) {
  if (ms < 1000)  return Math.round(ms) + 'ms';
  if (ms < 60000) return (ms / 1000).toFixed(1) + 's';
  const m = Math.floor(ms / 60000), s = Math.round((ms % 60000) / 1000);
  return m + 'm' + String(s).padStart(2, '0') + 's';
}

// ── Eval ─────────────────────────────────────────────────────────────────────

let _evalIndex = null;

async function initEval() {
  if (_evalIndex !== null) return;
  const r = await fetch('/eval_index');
  _evalIndex = await r.json();
  if (!_evalIndex.available) {
    const tab = document.getElementById('tabEval');
    tab.disabled = true;
    tab.title = 'No --model-outputs / --benchmarks dirs set';
    return;
  }
  const mSel = document.getElementById('eModelSel');
  mSel.innerHTML = '';
  for (const m of _evalIndex.models) {
    const o = document.createElement('option'); o.value = m; o.textContent = m; mSel.appendChild(o);
  }
  onEvalModelChange();
}

function onEvalModelChange() {
  if (!_evalIndex) return;
  const model = document.getElementById('eModelSel').value;
  const bSel  = document.getElementById('eBenchSel');
  bSel.innerHTML = '';
  const benches = (_evalIndex.tree[model] || {});
  for (const b of Object.keys(benches).sort()) {
    const o = document.createElement('option'); o.value = b; o.textContent = b; bSel.appendChild(o);
  }
  onEvalBenchChange();
}

function onEvalBenchChange() {
  if (!_evalIndex) return;
  const model  = document.getElementById('eModelSel').value;
  const bench  = document.getElementById('eBenchSel').value;
  const tSel   = document.getElementById('eTaskSel');
  tSel.innerHTML = '<option value="">(all tasks)</option>';
  const tasks = ((_evalIndex.tree[model] || {})[bench] || {});
  for (const t of Object.keys(tasks).sort()) {
    const o = document.createElement('option');
    o.value = t;
    o.textContent = TGB_LABELS[t] ?? t.replace(/_/g,' ').replace(/\b\w/g, c => c.toUpperCase());
    tSel.appendChild(o);
  }
  onEvalTaskChange();
}

function onEvalTaskChange() {
  if (!_evalIndex) return;
  const model = document.getElementById('eModelSel').value;
  const bench = document.getElementById('eBenchSel').value;
  const task  = document.getElementById('eTaskSel').value;
  const pSel  = document.getElementById('eProbSel');
  pSel.innerHTML = '<option value="">(all)</option>';
  const tree = ((_evalIndex.tree[model] || {})[bench] || {});

  // Each numeric option means "idx N from each task in the current scope":
  //   single-task mode   → 1 card
  //   all-tasks mode     → N cards (one per task that has this idx)
  // So the dropdown stays compact (~30 options instead of 1000+ in
  // all-tasks mode) AND every option does something useful — picking
  // "0000" in all-tasks mode shows the first example from every task,
  // perfect for scanning model behaviour across the benchmark without
  // paging through near-duplicates.
  const taskKeys = task ? [task] : Object.keys(tree);
  const idxs = [...new Set(taskKeys.flatMap(t => tree[t] || []))]
                 .sort((a, b) => a - b);
  for (const idx of idxs) {
    const o = document.createElement('option');
    o.value = String(idx);
    o.textContent = String(idx).padStart(4, '0');
    pSel.appendChild(o);
  }
}

async function doEvalLoad() {
  if (!_evalIndex) return;
  const model  = document.getElementById('eModelSel').value;
  const bench  = document.getElementById('eBenchSel').value;
  const task   = document.getElementById('eTaskSel').value;
  const probVal = document.getElementById('eProbSel').value;

  // Build list of (task, idx) to fetch.
  // probVal === ''     → "(all)": every (task, idx) pair in scope
  // probVal === 'N'    → idx N from each task in scope (1 card if a
  //                      single task is selected, N cards in all-tasks
  //                      mode where N tasks have that idx)
  const tree  = ((_evalIndex.tree[model] || {})[bench] || {});
  const tasks = task ? [task] : Object.keys(tree).sort();

  let jobs = [];
  if (probVal === '') {
    for (const t of tasks)
      for (const idx of (tree[t] || []))
        jobs.push({task: t, idx});
  } else {
    const idx = parseInt(probVal, 10);
    for (const t of tasks)
      if ((tree[t] || []).includes(idx))
        jobs.push({task: t, idx});
  }

  loaderStart('eLoader', 'Loading', jobs.length);
  document.getElementById('eLoadBtn').disabled = true;
  // Pre-create the grid with skeleton cards (one per job). As each
  // /eval_problem response comes in we replace the matching skeleton with
  // the real card, so the user sees results streaming in instead of
  // waiting for the whole batch.
  const wrap = renderEvalGridSkeleton(jobs);

  try {
    await fetchBatched(jobs, j =>
      fetch('/eval_problem?' + new URLSearchParams({model, benchmark: bench, task: j.task, idx: j.idx}))
        .then(r => r.json())
        .then(d => Object.assign({}, d, {task: j.task, idx: j.idx}))
        .catch(e => ({error: String(e), task: j.task, idx: j.idx}))
    , 4,
      (d) => loaderUpdate('eLoader', d),
      (result, idx) => {
        const card = makeEvalCard(result);
        const ph   = wrap.children[idx];
        if (ph) wrap.replaceChild(card, ph); else wrap.appendChild(card);
      });
    loaderStop('eLoader');
    document.getElementById('eLoadBtn').disabled = false;
  } catch(e) {
    loaderStop('eLoader');
    document.getElementById('eLoadBtn').disabled = false;
    document.getElementById('resultArea').innerHTML = '<div class="err">' + e + '</div>';
  }
}

async function doEvalAgg() {
  if (!_evalIndex) return;
  const model = document.getElementById('eModelSel').value;
  const bench = document.getElementById('eBenchSel').value;
  const task  = document.getElementById('eTaskSel').value;
  loaderStart('eLoader', 'Aggregating', 1);
  const r = await fetch('/eval_stats?' + new URLSearchParams({model, benchmark: bench, task}));
  const stats = await r.json();
  loaderStop('eLoader');
  document.getElementById('resultArea').innerHTML = '';
  document.getElementById('resultArea').appendChild(renderAggStats(stats, task || bench));
}

function renderEvalGridSkeleton(jobs) {
  // Pre-render one placeholder card per job so the user sees the layout
  // (and total count) immediately, then we swap each placeholder for the
  // real card as its /eval_problem response arrives. Returns the wrap
  // element so callers can index into wrap.children[i].
  const area = document.getElementById('resultArea');
  area.innerHTML = '';
  const wrap = document.createElement('div');
  wrap.className = 'eval-prob-grid';
  wrap.style.gridTemplateColumns = '1fr';
  for (const j of jobs) {
    const ph = document.createElement('div');
    ph.className = 'eval-prob-card eval-prob-skeleton';
    ph.innerHTML = `<div class="eval-prob-head">`
      + `<span style="color:#444">${j.task} / ${String(j.idx).padStart(4,'0')}</span>`
      + ` &nbsp;<span style="color:#333">loading\u2026</span></div>`
      + `<div style="height:160px;background:#0a1322;margin:7px;border-radius:3px"></div>`;
    wrap.appendChild(ph);
  }
  area.appendChild(wrap);
  return wrap;
}

const _DE_THRESHOLDS = [0, 2, 5, 10];

function makeEvalCard(d) {
  if (d.error) {
    const e = document.createElement('div'); e.className = 'err'; e.textContent = d.error; return e;
  }
  const card = document.createElement('div'); card.className = 'eval-prob-card';
  const s = d.stats || {};
  const pct = v => v !== undefined ? (v*100).toFixed(1)+'%' : '—';

  // Header
  const hd = document.createElement('div'); hd.className = 'eval-prob-head';
  hd.innerHTML = `<strong>${d.task} / ${String(d.idx).padStart(4,'0')}</strong>`
    + ` &nbsp;|&nbsp; has_output: ${s.has_output ?? '—'}`
    + ` &nbsp;|&nbsp; same_dim: ${s.same_dimensions !== undefined ? s.same_dimensions : '—'}`
    + ` &nbsp;|&nbsp; changed: ${s.n_changed ?? '—'} px`
    + ` &nbsp;|&nbsp; IoU@ΔE≤2: ${pct(s.iou_2)}`;
  card.appendChild(hd);

  // Helper: append a labelled scrollable image row.
  // `items` is [label, url|null] pairs; url=null renders an "N/A" placeholder.
  // Images use loading="lazy" so the browser only fetches them when scrolled
  // into view — the dominant perf win for the 1000+ card grids.
  function mkRow(label, items) {
    if (label) {
      const sec = document.createElement('div'); sec.className = 'eval-img-section';
      sec.textContent = label; card.appendChild(sec);
    }
    const row = document.createElement('div'); row.className = 'eval-img-row';
    for (const [lbl, url] of items) {
      const iw = document.createElement('div'); iw.className = 'eval-iw';
      if (url) {
        const img = document.createElement('img');
        // `loading` MUST be set before `src` — otherwise the browser may
        // start the fetch on the src= assignment, ignoring the lazy hint.
        img.loading = 'lazy';
        // Reserve layout space so the browser knows what's off-viewport
        // without having to load the image first. CSS still controls
        // final display size.
        img.width = 140; img.height = 140;
        img.src = url;
        iw.appendChild(img);
      } else {
        const ph = document.createElement('div');
        ph.style = 'width:140px;height:140px;background:#08101f;border-radius:3px;display:flex;align-items:center;justify-content:center;color:#333;font-size:.6rem';
        ph.textContent = 'N/A'; iw.appendChild(ph);
      }
      const sl = document.createElement('span'); sl.textContent = lbl; iw.appendChild(sl);
      row.appendChild(iw);
    }
    card.appendChild(row);
  }

  // Row 1: source images
  mkRow('Images', [
    ['Input',      d.input_url],
    ['Answer',     d.answer_url],
    ['Output',     d.output_url],
    ['Normalized', d.normalized_url],
  ]);

  // Row 2: ΔE diff maps; backend computes (0, 2, 5, 10).
  const diffUrls = d.diff_urls || {};
  mkRow('ΔE maps — strict (agg, r=0)',
    _DE_THRESHOLDS.map(t => [`ΔE≤${t}`, diffUrls[String(t)] || null])
  );

  // (Lenient agg r=1.5 + edge maps were never wired to the backend; the
  // old code only conditionally rendered them when those keys appeared.
  // Removed here — when/if they come back, add corresponding URL keys.)

  // Stats table
  if (d.stats) {
    const sw = document.createElement('div'); sw.className = 'eval-stats-wrap';
    const tbl = document.createElement('table'); tbl.className = 'eval-stats-tbl';
    const regions = ['all', 'changed', 'unchanged'];
    tbl.innerHTML = `<tr><th>Metric</th>${regions.map(r=>`<th>${r}</th>`).join('')}<th>IoU</th></tr>`;
    for (const t of _DE_THRESHOLDS) {
      const cells = regions.map(r => {
        const v = s[`prop_le_${t}_${r}`];
        return `<td>${v !== undefined ? (v*100).toFixed(1)+'%' : '—'}</td>`;
      }).join('');
      const iou = s[`iou_${t}`];
      tbl.innerHTML += `<tr><td>ΔE≤${t}</td>${cells}`
        + `<td>${iou !== undefined ? (iou*100).toFixed(1)+'%' : '—'}</td></tr>`;
    }
    sw.appendChild(tbl); card.appendChild(sw);
  }
  return card;
}

function renderAggStats(stats, title) {
  const wrap = document.createElement('div'); wrap.className = 'eval-agg';
  if (stats.error) {
    wrap.innerHTML = '<div class="err">' + stats.error + '</div>'; return wrap;
  }

  wrap.innerHTML = `<h3>${title}</h3>`;
  const pct = v => v !== undefined ? (v*100).toFixed(1)+'%' : '—';
  const num = (v, d=2) => v !== undefined ? (isFinite(v) ? (+v).toFixed(d) : '∞') : '—';

  // ── Coverage
  const h3c = document.createElement('h3'); h3c.textContent = 'Coverage'; wrap.appendChild(h3c);
  const covTbl = document.createElement('table'); covTbl.className = 'eval-stats-tbl';
  covTbl.style.maxWidth = '400px';
  covTbl.innerHTML = `<tr><th>Metric</th><th>Value</th></tr>
    <tr><td>With output</td><td>${pct(stats.prop_with_output)}</td></tr>
    <tr><td>Same dimensions</td><td>${pct(stats.prop_same_dimensions)}</td></tr>`;
  wrap.appendChild(covTbl);

  // ── IoU by ΔE threshold
  // ── CIE76 mean summary (avg over thresholds 0–10)
  const h3s = document.createElement('h3'); h3s.textContent = 'CIE76 mean (avg 0–10)'; wrap.appendChild(h3s);
  const sTbl = document.createElement('table'); sTbl.className = 'eval-stats-tbl';
  sTbl.style.maxWidth = '500px';
  sTbl.innerHTML = `<tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Mean IoU</td><td>${pct(stats.cie76_mean_iou)}</td></tr>
    <tr><td>Edit accuracy</td><td>${pct(stats.cie76_mean_edit_accuracy)}</td></tr>
    <tr><td>Preservation accuracy</td><td>${pct(stats.cie76_mean_preservation_accuracy)}</td></tr>`;
  wrap.appendChild(sTbl);

  // ── IoU by ΔE threshold
  const h3m = document.createElement('h3'); h3m.textContent = 'IoU by ΔE threshold'; wrap.appendChild(h3m);
  const mTbl = document.createElement('table'); mTbl.className = 'eval-stats-tbl';
  mTbl.style.maxWidth = '600px';
  mTbl.innerHTML = `<tr><th>Threshold</th><th>Mean IoU</th><th>IoU≥99%</th><th>IoU≥95%</th></tr>`;
  for (const t of _DE_THRESHOLDS) {
    mTbl.innerHTML += `<tr><td>ΔE≤${t}</td>`
      + `<td>${pct(stats[`mean_agg_r0_iou_${t}`])}</td>`
      + `<td>${pct(stats[`prop_agg_r0_iou_ge_99_${t}`])}</td>`
      + `<td>${pct(stats[`prop_agg_r0_iou_ge_95_${t}`])}</td></tr>`;
  }
  wrap.appendChild(mTbl);

  // ── Per-task breakdown
  if (stats.tasks) {
    const h3 = document.createElement('h3'); h3.textContent = 'Per-task summary'; wrap.appendChild(h3);
    const tbl2 = document.createElement('table'); tbl2.className = 'eval-stats-tbl';
    tbl2.style.maxWidth = '800px';
    tbl2.innerHTML = '<tr><th>Task</th><th>N</th><th>With output</th><th>Same dim</th>'
      + '<th>Edit accuracy</th><th>Pres. accuracy</th><th>IoU@ΔE≤2</th></tr>';
    for (const [task, ts] of Object.entries(stats.tasks).sort()) {
      tbl2.innerHTML += `<tr>
        <td>${task}</td>
        <td>${ts.n_problems ?? '—'}</td>
        <td>${pct(ts.prop_with_output)}</td>
        <td>${pct(ts.prop_same_dimensions)}</td>
        <td>${pct(ts.mean_edit_accuracy)}</td>
        <td>${pct(ts.mean_preservation_accuracy)}</td>
        <td>${pct(ts.mean_iou_2)}</td>
      </tr>`;
    }
    wrap.appendChild(tbl2);
  }
  return wrap;
}

// ── Init ────────────────────────────────────────────────────────────────────
onTaskChange();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Eval stats helpers (translate eval.py format → JS-expected format)
# ---------------------------------------------------------------------------

def _translate_problem_stats(record: dict) -> dict:
    """Translate a problem_stats.jsonl record into the format expected by the JS eval view."""
    out   = record.get("output", {})
    ct    = out.get("cie76_threshold", {})
    ep    = out.get("edit_pixels", 0)
    pp    = out.get("preservation_pixels", 0)
    total = ep + pp
    result = {
        "has_output":      record.get("has_output", False),
        "same_dimensions": record.get("correct_output_size", False),
        "n_changed":       ep,
    }
    for t in range(11):
        ts  = ct.get(str(t), {})
        ecp = ts.get("edit_correct_pixels", 0)
        pcp = ts.get("preservation_correct_pixels", 0)
        result[f"prop_le_{t}_all"]       = (ecp + pcp) / total if total else 0.0
        result[f"prop_le_{t}_changed"]   = ts.get("edit_accuracy",         0.0)
        result[f"prop_le_{t}_unchanged"] = ts.get("preservation_accuracy", 0.0)
        result[f"iou_{t}"]               = ts.get("iou", 0.0)
        result[f"changed_pixels_{t}"]    = ts.get("changed_pixels")
    return result


def _aggregate_problem_stats(records: list, task_filter: str = "") -> dict:
    """Aggregate problem_stats.jsonl records into the format the JS eval view expects."""
    if not records:
        return {"error": "No matching records found"}
    n             = len(records)
    n_with_output = sum(1 for r in records if r.get("has_output", False))
    n_same_dim    = sum(1 for r in records if r.get("correct_output_size", False))
    result = {
        "prop_with_output":     n_with_output / n,
        "prop_same_dimensions": n_same_dim    / n,
    }
    out_records = [r for r in records if r.get("has_output", False)
                   and "cie76_threshold" in r.get("output", {})]
    for t in range(11):
        ious = [r["output"]["cie76_threshold"][str(t)]["iou"] for r in out_records]
        if ious:
            result[f"mean_agg_r0_iou_{t}"]       = sum(ious) / len(ious)
            result[f"prop_agg_r0_iou_ge_99_{t}"] = sum(1 for v in ious if v >= 0.99) / len(ious)
            result[f"prop_agg_r0_iou_ge_95_{t}"] = sum(1 for v in ious if v >= 0.95) / len(ious)
        else:
            result[f"mean_agg_r0_iou_{t}"]       = None
            result[f"prop_agg_r0_iou_ge_99_{t}"] = None
            result[f"prop_agg_r0_iou_ge_95_{t}"] = None

    # CIE76 mean (avg over thresholds 0–10) — the primary benchmark metric
    mean_recs = [r for r in records if r.get("has_output", False)
                 and "cie76_mean" in r.get("output", {})]
    def _mean(vals):
        return sum(vals) / len(vals) if vals else None
    result["cie76_mean_iou"]                  = _mean([r["output"]["cie76_mean"]["iou"]                  for r in mean_recs])
    result["cie76_mean_edit_accuracy"]        = _mean([r["output"]["cie76_mean"]["edit_accuracy"]        for r in mean_recs])
    result["cie76_mean_preservation_accuracy"]= _mean([r["output"]["cie76_mean"]["preservation_accuracy"] for r in mean_recs])

    if not task_filter:
        task_names = sorted(set(r.get("task", "") for r in records))
        tasks = {}
        for tn in task_names:
            t_recs = [r for r in records if r.get("task") == tn]
            t_out  = [r for r in t_recs  if r.get("has_output", False)
                      and "cie76_threshold" in r.get("output", {})]
            nt = len(t_recs)
            iou2   = [r["output"]["cie76_threshold"]["2"]["iou"]              for r in t_out] if t_out else []
            ea2    = [r["output"]["cie76_threshold"]["2"]["edit_accuracy"]     for r in t_out
                      if "edit_accuracy"        in r["output"]["cie76_threshold"].get("2", {})] if t_out else []
            pa2    = [r["output"]["cie76_threshold"]["2"]["preservation_accuracy"] for r in t_out
                      if "preservation_accuracy" in r["output"]["cie76_threshold"].get("2", {})] if t_out else []
            tasks[tn] = {
                "n_problems":              nt,
                "prop_with_output":        len(t_out) / nt if nt else 0,
                "prop_same_dimensions":    sum(1 for r in t_recs if r.get("correct_output_size")) / nt if nt else 0,
                "mean_edit_accuracy":      sum(ea2) / len(ea2) if ea2 else None,
                "mean_preservation_accuracy": sum(pa2) / len(pa2) if pa2 else None,
                "mean_iou_2":              sum(iou2) / len(iou2) if iou2 else None,
            }
        result["tasks"] = tasks
    return result


# {results_dir: (mtime, records, index_by_key)} — invalidated when mtime changes.
_PROBLEM_STATS_CACHE: dict = {}


def _load_problem_stats_jsonl(results_dir: str) -> list:
    """Read all records from problem_stats.jsonl in the results root.

    Cached per (results_dir, mtime). The current production JSONL is ~35 MB /
    ~16 K records; parsing it linearly on every /eval_problem request was the
    dominant per-request cost (~260 ms each) before this cache.
    """
    from pathlib import Path as _Path
    path = _Path(results_dir) / "problem_stats.jsonl"
    if not path.exists():
        return []
    mtime = path.stat().st_mtime
    cached = _PROBLEM_STATS_CACHE.get(results_dir)
    if cached and cached[0] == mtime:
        return cached[1]

    records: list = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    index = {(r.get("model"), r.get("benchmark"), r.get("task"), r.get("idx")): r
             for r in records}
    _PROBLEM_STATS_CACHE[results_dir] = (mtime, records, index)
    return records


def _problem_stats_record(results_dir: str, model: str, benchmark: str,
                          task: str, idx: int) -> dict:
    """O(1) lookup of one problem_stats.jsonl record. Returns {} if absent."""
    _load_problem_stats_jsonl(results_dir)              # builds cache + index
    cached = _PROBLEM_STATS_CACHE.get(results_dir)
    return cached[2].get((model, benchmark, task, idx), {}) if cached else {}


# ── In-memory LRU for live-computed cie76 maps ─────────────────────────────
# The 4 ΔE-threshold image fetches for one problem all need the same
# cie76_map and `changed` mask. Without this cache, the heavy LAB conversion
# (~120 ms) would re-run for each threshold. With it, the first /eval_image
# fetch for a (model, bench, task, idx) pays the cost; the next 3 (one per
# threshold) hit the cache.
#
# Memory: each cie76_map is float32 H×W (~4 MB at 1024²) + a uint8 changed
# mask (~1 MB). Cap at 32 entries → ~160 MB worst case.
_CIE76_LRU: "OrderedDict[tuple, tuple]" = OrderedDict()
_CIE76_LRU_LOCK = threading.Lock()
_CIE76_LRU_MAX  = 32


def _get_cie76_map(model: str, benchmark: str, task: str, idx: int):
    """Return (cie76_map, changed_mask) for one problem, computing if needed.

    Returns (None, None) if there's no model output PNG on disk for this
    (model, benchmark, task, idx).
    """
    key = (model, benchmark, task, idx)
    with _CIE76_LRU_LOCK:
        if key in _CIE76_LRU:
            _CIE76_LRU.move_to_end(key)
            return _CIE76_LRU[key]

    # Heavy work outside the lock — concurrent requests for *different* keys
    # can run in parallel; concurrent requests for the *same* key may both
    # compute (acceptable race — the result is deterministic and the second
    # write just re-stamps the LRU entry).
    ev      = _get_eval()
    num_str = f"{idx:03d}"
    num4    = f"{idx:04d}"

    if not (_BENCHMARKS_ROOT and _MODEL_OUTPUTS_DIR):
        return (None, None)
    bench_task = Path(_BENCHMARKS_ROOT) / benchmark / task
    out_path   = Path(_MODEL_OUTPUTS_DIR) / model / benchmark / task / f"{num4}_output.png"
    if not out_path.exists():
        return (None, None)

    I_img  = ev.Image.open(bench_task / f"{num_str}_input.png").convert("RGB")
    A_img  = ev.Image.open(bench_task / f"{num_str}_answer.png").convert("RGB")
    O_raw  = ev.Image.open(out_path).convert("RGB")
    O_norm = ev.normalize_output(O_raw, A_img)
    I = ev.np.array(I_img)
    A = ev.np.array(A_img)
    O = ev.np.array(O_norm)
    _, cie76_map, changed = ev.compute_problem_stats(I, A, O)

    with _CIE76_LRU_LOCK:
        _CIE76_LRU[key] = (cie76_map, changed)
        _CIE76_LRU.move_to_end(key)
        while len(_CIE76_LRU) > _CIE76_LRU_MAX:
            _CIE76_LRU.popitem(last=False)
    return (cie76_map, changed)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class _Handler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self._serve_index()
        elif parsed.path == "/gen":
            self._serve_gen(parsed.query)
        elif parsed.path == "/bench_list":
            self._serve_bench_list()
        elif parsed.path == "/bench_index":
            self._serve_bench_index(parsed.query)
        elif parsed.path == "/bench_get":
            self._serve_bench_get(parsed.query)
        elif parsed.path == "/eval_index":
            self._serve_eval_index()
        elif parsed.path == "/eval_problem":
            self._serve_eval_problem(parsed.query)
        elif parsed.path == "/eval_image":
            self._serve_eval_image(parsed.query)
        elif parsed.path == "/eval_stats":
            self._serve_eval_stats(parsed.query)
        else:
            self.send_error(404)

    def _serve_index(self):
        opt_parts = [
            f'<option value="{n}">{n.replace("_", " ").title()}</option>'
            for _, n in TASKS
        ]
        tgf_pairs = _tgf_task_keys()
        for key, label in tgf_pairs:
            opt_parts.append(f'<option value="{key}">{label}</option>')
        task_options = "\n".join(opt_parts)
        task_params: dict = {}
        for _, name in TASKS:
            try:
                mod = _load_module(name)
                task_params[name] = {k: [str(v) for v in vs]
                                     for k, vs in mod.PARAMETERS.items()}
            except Exception:
                task_params[name] = {}
        for key, _ in tgf_pairs:
            task_params[key] = {}

        # Map TGB folder names (e.g. "bar_chart_add_bar") to display labels
        # ("Bar Chart · Add Bar") for the Benchmark-tab task dropdown.
        tgb_task_labels: dict[str, str] = {}
        for g in _TGF_GRAPHS:
            import importlib as _il
            mod = _il.import_module(f"tinygrafixbench.{g}")
            for t in mod.TASKS:
                folder = f"{g}_{t}"
                tgb_task_labels[folder] = (
                    f"{g.replace('_', ' ').title()} · {t.replace('_', ' ').title()}"
                )

        html = (_HTML
                .replace("__TASK_OPTIONS__", task_options)
                .replace("__TASK_PARAMS__", json.dumps(task_params))
                .replace("__TGB_TASK_LABELS__", json.dumps(tgb_task_labels))
                .replace("__SIZE_OPTIONS__", _SIZE_OPTIONS))
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _serve_gen(self, query: str):
        args   = urllib.parse.parse_qs(query)
        task   = args.get("task",   ["translation"])[0]
        seed   = int(args.get("seed", ["42"])[0])
        params = json.loads(args.get("params", ["{}"])[0])
        W      = int(args.get("W", [str(_DEFAULT_W)])[0])
        H      = int(args.get("H", [str(_DEFAULT_H)])[0])
        # Clamp to multiples of 64 in [64, 2048]
        W = max(64, min(2048, (W // 64) * 64))
        H = max(64, min(2048, (H // 64) * 64))

        pal_choice = args.get("palette", ["random"])[0]
        bg_choice  = args.get("background", ["random"])[0]

        try:
            if task.startswith("tgf:"):
                graph, tname = task[4:].split(".", 1)
                tgf_mod = importlib.import_module(f"tinygrafixbench.{graph}")
                input_fig, answer_fig, instruction = tgf_mod.generate_task(seed, tname)
                result = {
                    "instruction": instruction,
                    "input_b64":   _fig_to_b64(input_fig),
                    "answer_b64":  _fig_to_b64(answer_fig),
                    "params":      {},
                }
                self._send_json(json.dumps(result).encode())
                return

            mod = _load_module(task)

            # Resolve palette/bg choices (defer to a small RNG so "random" is
            # reproducible per seed, independent of the color-split stream).
            choice_rng = _random.Random(seed ^ 0xBEEF)
            if pal_choice == "standard":
                palette = STANDARD_PALETTE
            elif pal_choice == "nonstandard":
                palette = NONSTANDARD_PALETTE
            else:
                palette = STANDARD_PALETTE if choice_rng.random() < 0.5 else NONSTANDARD_PALETTE

            if bg_choice == "striped":
                use_striped = True
            elif bg_choice == "solid":
                use_striped = False
            else:  # "random"
                use_striped = choice_rng.random() < 0.5

            # Colors and background use the SAME helpers as generate_benchmark,
            # so the visualizer and the benchmark produce byte-identical scenes
            # for matching (seed, W, H, palette, striped) configurations.
            bg_rgb, holdout_rgb, colors = _color_split(palette, seed)
            bg = (_striped_bg(bg_rgb, holdout_rgb, seed)
                  if use_striped else BackgroundSpec(colors=[bg_rgb]))

            prob = mod.generate(seed=seed, bg_spec=bg, W=W, H=H,
                                obj_colors=colors, **params)
            result = {
                "instruction": prob.instruction,
                "input_b64":   _img_to_b64(prob.input_image),
                "answer_b64":  _img_to_b64(prob.answer_image),
                "params":      prob.metadata.get("params", {}),
            }
        except Exception:
            import traceback
            result = {"error": traceback.format_exc()}

        self._send_json(json.dumps(result).encode())

    def _serve_bench_list(self):
        if _BENCHMARKS_ROOT:
            root       = Path(_BENCHMARKS_ROOT)
            benchmarks = sorted(d.name for d in root.iterdir()
                                if d.is_dir() and (d / "problems.jsonl").exists())
            result = {"available": bool(benchmarks), "benchmarks": benchmarks}
        elif _BENCHMARK_DIR:
            result = {"available": True,
                      "benchmarks": [Path(_BENCHMARK_DIR).name]}
        else:
            result = {"available": False, "benchmarks": []}
        self._send_json(json.dumps(result).encode())

    def _serve_bench_index(self, query: str = ""):
        args       = urllib.parse.parse_qs(query)
        bench_name = args.get("benchmark", [""])[0]

        if bench_name and _BENCHMARKS_ROOT:
            bench_dir = os.path.join(_BENCHMARKS_ROOT, bench_name)
        elif _BENCHMARK_DIR:
            bench_dir = _BENCHMARK_DIR
        else:
            self._send_json(json.dumps({"available": False, "problems": []}).encode())
            return

        jsonl_path = os.path.join(bench_dir, "problems.jsonl")
        problems   = []
        if os.path.exists(jsonl_path):
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            problems.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        self._send_json(json.dumps({"available": True, "problems": problems}).encode())

    def _serve_bench_get(self, query: str):
        args       = urllib.parse.parse_qs(query)
        task       = args.get("task",       [""])[0]
        problem_id = int(args.get("problem_id", ["0"])[0])
        bench_name = args.get("benchmark",  [""])[0]

        if bench_name and _BENCHMARKS_ROOT:
            bench_dir = os.path.join(_BENCHMARKS_ROOT, bench_name)
        else:
            bench_dir = _BENCHMARK_DIR

        try:
            task_dir    = os.path.join(bench_dir, task)
            input_path  = os.path.join(task_dir, f"{problem_id:03d}_input.png")
            answer_path = os.path.join(task_dir, f"{problem_id:03d}_answer.png")
            meta_path   = os.path.join(task_dir, f"{problem_id:03d}.json")
            with open(meta_path) as f:
                meta = json.load(f)
            result = {
                "instruction": meta["instruction"],
                "input_b64":   _file_to_b64(input_path),
                "answer_b64":  _file_to_b64(answer_path),
                "params":      meta.get("params", {}),
                "seed":        meta.get("seed", 0),
            }
        except Exception:
            import traceback
            result = {"error": traceback.format_exc()}

        self._send_json(json.dumps(result).encode())

    def _serve_eval_index(self):
        """Return {available, models, tree: {model: {benchmark: {task: [idx,...]}}}}.

        Source of truth: model_outputs/ (which models × benchmarks have been
        run) intersected with benchmarks/ (which tasks each benchmark contains
        and what input indices exist). --results carries cached stats and
        cached diagnostic images served by /eval_image; it is not the index
        source after the model_outputs/ ⟷ eval_outputs/ split.
        """
        from pathlib import Path as _Path
        if _MODEL_OUTPUTS_DIR and _BENCHMARKS_ROOT:
            out_root   = _Path(_MODEL_OUTPUTS_DIR)
            bench_root = _Path(_BENCHMARKS_ROOT)
            tree = {}
            for model_dir in sorted(out_root.iterdir()):
                if not model_dir.is_dir(): continue
                tree[model_dir.name] = {}
                for bench_dir in sorted(model_dir.iterdir()):
                    if not bench_dir.is_dir(): continue
                    b_src = bench_root / bench_dir.name
                    if not b_src.exists(): continue
                    tree[model_dir.name][bench_dir.name] = {}
                    for task_dir in sorted(b_src.iterdir()):
                        if not task_dir.is_dir(): continue
                        idxs = sorted(
                            int(p.stem.replace("_input", ""))
                            for p in task_dir.glob("*_input.png")
                        )
                        if idxs:
                            tree[model_dir.name][bench_dir.name][task_dir.name] = idxs
            models = sorted(tree.keys())
            result = {"available": bool(models), "models": models, "tree": tree}
        else:
            result = {"available": False, "models": [], "tree": {}}

        body = json.dumps(result).encode()
        self._send_json(body)

    def _serve_eval_problem(self, query: str):
        """Return stats + image URLs for one problem (no inline image bytes).

        After the lazy-URL refactor, image bytes are served by /eval_image and
        the browser fetches them lazily as cards scroll into view. This
        endpoint is now ~1 KB per response (was ~1 MB) so a 96-problem task
        load drops from ~96 MB of upfront JSON to ~96 KB.

        Stats source of truth is `problem_stats.jsonl` (cached + indexed).
        The pre-`make eval` fallback still computes stats live — slow but
        correct for a model whose JSONL hasn't been refreshed yet.
        """
        args      = urllib.parse.parse_qs(query)
        model     = args.get("model",     [""])[0]
        benchmark = args.get("benchmark", [""])[0]
        task      = args.get("task",      [""])[0]
        idx       = int(args.get("idx",   ["0"])[0])

        try:
            num4 = f"{idx:04d}"

            raw_record = (_problem_stats_record(_RESULTS_DIR, model, benchmark, task, idx)
                          if _RESULTS_DIR else {})

            if raw_record:
                stats = _translate_problem_stats(raw_record)
                has_output = stats.get("has_output", False)
            else:
                # Live fallback for the (model, bench, task, idx) tuples not
                # covered by problem_stats.jsonl yet. Mirrors the old eager
                # path but only fires on cache miss.
                stats, has_output = self._live_problem_stats(model, benchmark, task, idx, num4)

            def _url(kind: str) -> str:
                return "/eval_image?" + urllib.parse.urlencode({
                    "model":     model,
                    "benchmark": benchmark,
                    "task":      task,
                    "idx":       idx,
                    "kind":      kind,
                })

            result = {
                "stats":          stats,
                "input_url":      _url("input"),
                "answer_url":     _url("answer"),
                "output_url":     _url("output")     if has_output else None,
                "normalized_url": _url("normalized") if has_output else None,
                "diff_urls":      ({str(t): _url(f"diff_de_{t}") for t in _LIVE_DIFF_THRESHOLDS}
                                   if has_output else {}),
            }
        except Exception:
            import traceback
            result = {"error": traceback.format_exc()}

        self._send_json(json.dumps(result).encode())

    def _live_problem_stats(self, model: str, benchmark: str, task: str,
                            idx: int, num4: str):
        """Slow path: compute stats from input/answer/output PNGs when
        problem_stats.jsonl doesn't have a record yet. Returns (stats, has_output).
        """
        ev = _get_eval()
        num_str = f"{idx:03d}"
        bench_task = Path(_BENCHMARKS_ROOT) / benchmark / task
        out_task   = Path(_MODEL_OUTPUTS_DIR) / model / benchmark / task if _MODEL_OUTPUTS_DIR else None
        out_path   = (out_task / f"{num4}_output.png") if out_task else None
        has_output = out_path is not None and out_path.exists()

        I_img = ev.Image.open(bench_task / f"{num_str}_input.png").convert("RGB")
        A_img = ev.Image.open(bench_task / f"{num_str}_answer.png").convert("RGB")
        I = ev.np.array(I_img); A = ev.np.array(A_img)
        changed = ev.np.any(I != A, axis=-1)
        ep = int(changed.sum()); pp = int((~changed).sum()); total = ep + pp

        stats: dict = {
            "has_output":      has_output,
            "same_dimensions": False,
            "n_changed":       ep,
        }
        if has_output:
            O_raw  = ev.Image.open(out_path).convert("RGB")
            stats["same_dimensions"] = (O_raw.size == A_img.size)
            O_norm = ev.normalize_output(O_raw, A_img)
            O = ev.np.array(O_norm)
            stats_raw, cie76_map, changed = ev.compute_problem_stats(I, A, O)
            # Pre-warm the CIE76 LRU so the subsequent /eval_image?kind=
            # diff_de_* requests for this problem don't pay another
            # ~120 ms LAB conversion. Cheap (one OrderedDict insert) and
            # only matters in the slow-path JSONL-miss case where the
            # user is already serving live compute.
            with _CIE76_LRU_LOCK:
                key = (model, benchmark, task, idx)
                _CIE76_LRU[key] = (cie76_map, changed)
                _CIE76_LRU.move_to_end(key)
                while len(_CIE76_LRU) > _CIE76_LRU_MAX:
                    _CIE76_LRU.popitem(last=False)
            ct = stats_raw.get("cie76_threshold", {})
            for t in range(11):
                ts  = ct.get(str(t), {})
                ecp = ts.get("edit_correct_pixels", 0)
                pcp = ts.get("preservation_correct_pixels", 0)
                stats[f"prop_le_{t}_all"]       = (ecp + pcp) / total if total else 0.0
                stats[f"prop_le_{t}_changed"]   = ts.get("edit_accuracy",         0.0)
                stats[f"prop_le_{t}_unchanged"] = ts.get("preservation_accuracy", 0.0)
                stats[f"iou_{t}"]               = ts.get("iou", 0.0)
        return stats, has_output

    def _serve_eval_image(self, query: str):
        """Serve one PNG image for an eval problem.

        Reads the disk cache (populated by `make eval`) when present, else
        computes-and-caches. The 4 ΔE diff images for one problem share an
        in-memory cie76_map LRU so the heavy LAB conversion runs at most
        once per problem until the LRU evicts.

        Query: model, benchmark, task, idx, kind
        kind ∈ {input, answer, output, normalized, diff_de_<t>} where
        t ∈ _LIVE_DIFF_THRESHOLDS.
        """
        args      = urllib.parse.parse_qs(query)
        model     = args.get("model",     [""])[0]
        benchmark = args.get("benchmark", [""])[0]
        task      = args.get("task",      [""])[0]
        try:
            idx   = int(args.get("idx",   ["0"])[0])
        except ValueError:
            self.send_error(400, "bad idx"); return
        kind  = args.get("kind", [""])[0]

        # Path-traversal guard: model/benchmark/task/kind all flow into
        # ``Path()`` constructions that prefix the configured roots, so a
        # ``..``-laced query param would escape them. ``make viz`` binds
        # ``0.0.0.0`` so this is a real (if low-blast-radius) attack
        # surface introduced by serving these as URL params instead of
        # inlining everything in the JSONL. Reject anything with a path
        # separator or a parent-dir segment up front.
        for name, val in (("model", model), ("benchmark", benchmark),
                          ("task", task), ("kind", kind)):
            if not val or ".." in val or "/" in val or "\\" in val or "\x00" in val:
                self.send_error(400, f"invalid {name}"); return

        try:
            num_str = f"{idx:03d}"
            num4    = f"{idx:04d}"

            if kind == "input" or kind == "answer":
                if not _BENCHMARKS_ROOT:
                    self.send_error(404, "no --benchmarks"); return
                self._serve_png_file(Path(_BENCHMARKS_ROOT) / benchmark / task / f"{num_str}_{kind}.png")
                return

            if kind == "output":
                if not _MODEL_OUTPUTS_DIR:
                    self.send_error(404, "no --model-outputs"); return
                self._serve_png_file(Path(_MODEL_OUTPUTS_DIR) / model / benchmark / task / f"{num4}_output.png")
                return

            if kind == "normalized":
                self._serve_normalized(model, benchmark, task, num_str, num4)
                return

            if kind.startswith("diff_de_"):
                try:
                    t = int(kind[len("diff_de_"):])
                except ValueError:
                    self.send_error(400, f"bad threshold in {kind}"); return
                if t not in _LIVE_DIFF_THRESHOLDS:
                    self.send_error(400, f"unsupported threshold: {t}"); return
                self._serve_diff_de(model, benchmark, task, idx, num4, t)
                return

            self.send_error(400, f"unknown kind: {kind}")
        except Exception:
            import traceback
            self.send_error(500, traceback.format_exc())

    def _serve_normalized(self, model, benchmark, task, num_str, num4):
        # 1. eval cache (only populated when normalize_output had to resize/crop)
        if _RESULTS_DIR:
            cache = Path(_RESULTS_DIR) / model / benchmark / task / f"{num4}_normalized_output.png"
            if cache.exists():
                self._serve_png_file(cache); return

        # 2. No cache file → either there's no resize needed (common) or eval
        #    hasn't been run. Decide by actually doing the normalize.
        ev = _get_eval()
        if not (_BENCHMARKS_ROOT and _MODEL_OUTPUTS_DIR):
            self.send_error(404, "no --benchmarks/--model-outputs"); return
        bench_task = Path(_BENCHMARKS_ROOT) / benchmark / task
        out_path   = Path(_MODEL_OUTPUTS_DIR) / model / benchmark / task / f"{num4}_output.png"
        if not out_path.exists():
            self.send_error(404, "no model output"); return
        # ``.convert("RGB")`` to mirror what _get_cie76_map does and what
        # eval.py's _process_one_problem does — keeps mode-handling
        # consistent across the codebase. ``normalize_output`` only reads
        # ``A_img.size`` today (mode-independent), so this is defensive
        # against future changes that might touch pixel data.
        A_img  = ev.Image.open(bench_task / f"{num_str}_answer.png").convert("RGB")
        O_raw  = ev.Image.open(out_path).convert("RGB")
        O_norm = ev.normalize_output(O_raw, A_img)
        if O_norm is O_raw:
            # No resize → normalized == raw output, just serve the raw bytes
            self._serve_png_file(out_path); return

        # Resize was needed and we don't have a cache. Encode once, write to
        # disk if --results (for next time), and serve from memory either way
        # — no need for a save → re-read round-trip.
        buf = io.BytesIO(); O_norm.save(buf, format="PNG"); body = buf.getvalue()
        if _RESULTS_DIR:
            cache_dir = Path(_RESULTS_DIR) / model / benchmark / task
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / f"{num4}_normalized_output.png").write_bytes(body)
        self._serve_png_bytes(body)

    def _serve_diff_de(self, model, benchmark, task, idx, num4, t):
        # 1. eval cache (now populated by default `make eval`)
        if _RESULTS_DIR:
            cache = Path(_RESULTS_DIR) / model / benchmark / task / f"{num4}_diff_cie76_{t}.png"
            if cache.exists():
                self._serve_png_file(cache); return

        # 2. Compute via the in-memory cie76_map LRU (shared across all 4
        # threshold requests for this problem). Save to disk for next time
        # if --results is set.
        cie76_map, changed = _get_cie76_map(model, benchmark, task, idx)
        if cie76_map is None:
            self.send_error(404, "no model output for this problem"); return
        ev  = _get_eval()
        img = ev.make_eval_image(changed, cie76_map, t)
        buf = io.BytesIO(); img.save(buf, format="PNG"); body = buf.getvalue()
        if _RESULTS_DIR:
            cache_dir = Path(_RESULTS_DIR) / model / benchmark / task
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / f"{num4}_diff_cie76_{t}.png").write_bytes(body)
        self._serve_png_bytes(body)

    def _serve_png_file(self, path: Path):
        """Serve a PNG from disk with Cache-Control + Content-Length headers."""
        if not path.exists():
            self.send_error(404, f"missing: {path.name}"); return
        with open(path, "rb") as f:
            body = f.read()
        self._serve_png_bytes(body)

    def _serve_png_bytes(self, body: bytes):
        self.send_response(200)
        self.send_header("Content-Type",   "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control",  "public, max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def _serve_eval_stats(self, query: str):
        """Return aggregate stats for model+benchmark (optionally filtered by task)."""
        args      = urllib.parse.parse_qs(query)
        model     = args.get("model",     [""])[0]
        benchmark = args.get("benchmark", [""])[0]
        task      = args.get("task",      [""])[0]

        try:
            from pathlib import Path as _Path
            if _RESULTS_DIR:
                # Read and aggregate from problem_stats.jsonl
                all_records = _load_problem_stats_jsonl(_RESULTS_DIR)
                records = [
                    r for r in all_records
                    if r.get("model") == model and r.get("benchmark") == benchmark
                    and (not task or r.get("task") == task)
                ]
                result = _aggregate_problem_stats(records, task)
            else:
                # Live computation — requires --model-outputs and --benchmarks
                ev = _get_eval()
                bench_root = _Path(_BENCHMARKS_ROOT) / benchmark
                out_root   = _Path(_MODEL_OUTPUTS_DIR) / model / benchmark
                tasks_to_run = [task] if task else sorted(
                    d.name for d in bench_root.iterdir() if d.is_dir()
                )
                records = []
                for t in tasks_to_run:
                    bench_task = bench_root / t
                    out_task   = out_root / t
                    if not bench_task.exists():
                        continue
                    for inp in sorted(bench_task.glob("*_input.png")):
                        num_str = inp.stem.replace("_input", "")
                        idx_val = int(num_str)
                        num4    = f"{idx_val:04d}"
                        I_img = ev.Image.open(inp).convert("RGB")
                        A_pil = ev.Image.open(bench_task / f"{num_str}_answer.png").convert("RGB")
                        I = ev.np.array(I_img)
                        A = ev.np.array(A_pil)
                        changed = ev.np.any(I != A, axis=-1)
                        ep = int(changed.sum())
                        pp = int((~changed).sum())
                        out_path = out_task / f"{num4}_output.png"
                        O_raw  = (ev.Image.open(out_path).convert("RGB")
                                  if (out_task.exists() and out_path.exists()) else None)
                        O_norm = ev.normalize_output(O_raw, A_pil) if O_raw else None
                        O      = ev.np.array(O_norm) if O_norm else None
                        rec = {
                            "model": model, "benchmark": benchmark, "task": t,
                            "idx": idx_val,
                            "has_output": O_raw is not None,
                            "correct_output_size": (list(O_raw.size) == list(A_pil.size)) if O_raw else False,
                            "output": ev._zero_stats(ep, pp),
                        }
                        if O is not None:
                            stats_raw, _, _ = ev.compute_problem_stats(I, A, O)
                            rec["output"] = {"output_size": list(O_raw.size), **stats_raw}
                        records.append(rec)
                result = _aggregate_problem_stats(records, task)
        except Exception:
            import traceback
            result = {"error": traceback.format_exc()}

        self._send_json(json.dumps(result).encode())

    def _send_json(self, body: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global _BENCHMARK_DIR, _MODEL_OUTPUTS_DIR, _BENCHMARKS_ROOT, _RESULTS_DIR

    port = 8765
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1]); i += 2
        elif args[i] == "--benchmark" and i + 1 < len(args):
            _BENCHMARK_DIR = args[i + 1]; i += 2
        elif args[i] == "--model-outputs" and i + 1 < len(args):
            _MODEL_OUTPUTS_DIR = args[i + 1]; i += 2
        elif args[i] == "--benchmarks" and i + 1 < len(args):
            _BENCHMARKS_ROOT = args[i + 1]; i += 2
        elif args[i] == "--eval-outputs" and i + 1 < len(args):
            _RESULTS_DIR = args[i + 1]; i += 2
        else:
            i += 1

    # Auto-detect repo-root directories when not explicitly passed.
    _ROOT = _HERE.parent
    if not _BENCHMARK_DIR and not _BENCHMARKS_ROOT:
        d = str(_ROOT / "benchmarks")
        if os.path.isdir(d):
            _BENCHMARKS_ROOT = d
    if not _MODEL_OUTPUTS_DIR:
        d = str(_ROOT / "model_outputs")
        if os.path.isdir(d):
            _MODEL_OUTPUTS_DIR = d
    if not _RESULTS_DIR:
        d = str(_ROOT / "eval_outputs")
        if os.path.isdir(d):
            _RESULTS_DIR = d

    if _BENCHMARK_DIR:
        print(f"Benchmark dir:    {_BENCHMARK_DIR}")
    if _MODEL_OUTPUTS_DIR:
        print(f"Model outputs:    {_MODEL_OUTPUTS_DIR}")
    if _BENCHMARKS_ROOT:
        print(f"Benchmarks root:  {_BENCHMARKS_ROOT}")
    if _RESULTS_DIR:
        print(f"Results dir:      {_RESULTS_DIR}  (pre-computed eval — Eval tab reads from disk)")

    class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    server = _Server(("0.0.0.0", port), _Handler)
    print(f"PaintBench Visualizer  →  http://localhost:{port}")
    print("Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
