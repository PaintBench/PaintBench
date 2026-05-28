"""Generate a self-contained report.html from aggregate_stats.jsonl.

Usage:
    python src/report.py \
        --input  eval_outputs/aggregate_stats.jsonl \
        --output report.html
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

BENCH_ORDER = ["PaintBench", "TinyGrafixBench"]


def _model_sort_key(model: str, pb_bench: dict) -> tuple:
    """Best PaintBench cie76_mean IoU first; models without PaintBench data last."""
    entry = pb_bench.get(model)
    if entry and "cie76_mean" in entry:
        return (0, -entry["cie76_mean"]["mean_iou"], model)
    return (1, 0, model)


def load(path: str, exclude_models: tuple[str, ...] = ()) -> dict:
    bench_level:          dict = defaultdict(dict)
    category_level:       dict = defaultdict(lambda: defaultdict(dict))
    visual_condition_level: dict = defaultdict(lambda: defaultdict(dict))
    task_level:           dict = defaultdict(lambda: defaultdict(dict))
    mode_level:           dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    cat_order:            dict = defaultdict(list)
    cat_to_tasks:         dict = defaultdict(lambda: defaultdict(list))
    vcond_order:          dict = defaultdict(list)
    task_order:           dict = defaultdict(list)
    mode_order:           dict = defaultdict(lambda: defaultdict(list))
    models_seen:  set  = set()
    benches_seen: list = []

    def metrics(row: dict) -> dict:
        m: dict = {
            "n_problems":            row.get("n_problems"),
            "n_tasks":               row.get("n_tasks"),  # aggregate rows only; bootstrap unit
            "n_with_output":         row.get("n_with_output"),
            "n_correct_output_size": row.get("n_correct_output_size"),
        }
        for t in range(11):
            k = f"cie76_{t}"
            if k in row:
                m[k] = row[k]
        if "cie76_mean" in row:
            m["cie76_mean"] = row["cie76_mean"]
        return m

    with open(path) as f:
        for line in f:
            row  = json.loads(line)
            mdl  = row["model"]
            if any(pat in mdl for pat in exclude_models):
                continue
            bnch = row["benchmark"]
            lvl  = row["level"]
            models_seen.add(mdl)
            if bnch not in benches_seen:
                benches_seen.append(bnch)

            m = metrics(row)
            if lvl == "benchmark":
                bench_level[bnch][mdl] = m
            elif lvl == "category":
                cat = row["category"]
                if cat not in cat_order[bnch]:
                    cat_order[bnch].append(cat)
                category_level[bnch][cat][mdl] = m
            elif lvl == "visual_condition":
                vcond = row["visual_condition"]
                if vcond not in vcond_order[bnch]:
                    vcond_order[bnch].append(vcond)
                visual_condition_level[bnch][vcond][mdl] = m
            elif lvl == "task":
                task     = row["task"]
                task_cat = row.get("category")
                if task not in task_order[bnch]:
                    task_order[bnch].append(task)
                # Only categorized tasks appear in the grouped layout.
                # Diagnostic tasks (e.g. preservation, category=None) get a
                # task_level entry but are not added to cat_to_tasks.
                if task_cat is not None and task not in cat_to_tasks[bnch][task_cat]:
                    cat_to_tasks[bnch][task_cat].append(task)
                task_level[bnch][task][mdl] = m
            elif lvl == "mode":
                task = row["task"]
                mode = row["mode"]
                if mode not in mode_order[bnch][task]:
                    mode_order[bnch][task].append(mode)
                mode_level[bnch][task][mode][mdl] = m

    # Sort modes alphabetically; put "baseline" first.
    for bnch in mode_order:
        for task in mode_order[bnch]:
            modes = mode_order[bnch][task]
            mode_order[bnch][task] = (
                (["baseline"] if "baseline" in modes else []) +
                sorted(m for m in modes if m != "baseline")
            )

    pb_bench = bench_level.get("PaintBench", {})
    models = sorted(models_seen, key=lambda m: _model_sort_key(m, pb_bench))
    benchmarks = [b for b in BENCH_ORDER if b in benches_seen]

    return {
        "models":                 models,
        "benchmarks":             benchmarks,
        "bench_level":            {b: dict(v) for b, v in bench_level.items()},
        "category_level":         {b: {c: dict(v) for c, v in cs.items()}
                                   for b, cs in category_level.items()},
        "visual_condition_level": {b: {c: dict(v) for c, v in cs.items()}
                                   for b, cs in visual_condition_level.items()},
        "task_level":             {b: {t: dict(v) for t, v in ts.items()}
                                   for b, ts in task_level.items()},
        "mode_level":             {b: {t: {mo: dict(v) for mo, v in ms.items()}
                                       for t, ms in ts.items()}
                                   for b, ts in mode_level.items()},
        "cat_order":              {b: list(v) for b, v in cat_order.items()},
        "cat_to_tasks":           {b: {c: list(v) for c, v in cs.items()}
                                   for b, cs in cat_to_tasks.items()},
        "vcond_order":            {b: list(v) for b, v in vcond_order.items()},
        "task_order":             {b: list(v) for b, v in task_order.items()},
        "mode_order":             {b: {t: list(v) for t, v in ts.items()}
                                   for b, ts in mode_order.items()},
    }


CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: system-ui, -apple-system, sans-serif;
  background: #f4f5f7;
  color: #1a1a1a;
  padding: 28px 32px;
}
h1 {
  font-size: 24px;
  font-weight: 700;
  margin-bottom: 20px;
  color: #111;
}
h2 {
  font-size: 17px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #555;
  margin: 36px 0 10px;
  padding-bottom: 6px;
  border-bottom: 2px solid #ddd;
}
h3 {
  font-size: 13px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #888;
  margin: 20px 0 8px;
}
.controls {
  display: flex;
  gap: 32px;
  align-items: center;
  background: #fff;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 14px 22px;
  margin-bottom: 28px;
  flex-wrap: wrap;
  box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
.ctrl { display: flex; align-items: center; gap: 9px; }
.ctrl label {
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .05em;
  color: #666;
}
select {
  font-size: 13px;
  padding: 5px 10px;
  border: 1px solid #ccc;
  border-radius: 5px;
  background: #fff;
  cursor: pointer;
  color: #222;
}
select:focus { outline: 2px solid #4a7fc1; }

.table-wrap { overflow-x: auto; margin-bottom: 6px; }

table {
  width: 100%;
  border-collapse: collapse;
  background: #fff;
  border: 1px solid #ddd;
  border-radius: 6px;
  overflow: hidden;
  font-size: 14px;
  box-shadow: 0 1px 4px rgba(0,0,0,.05);
}
thead th {
  background: #eef0f4;
  padding: 9px 14px;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .05em;
  color: #555;
  border-bottom: 2px solid #d4d7de;
  border-right: 1px solid #d4d7de;
  white-space: nowrap;
  text-align: center;
}
thead th.lbl { text-align: left; min-width: 200px; }
thead th:last-child { border-right: none; }

tbody td {
  padding: 7px 14px;
  border-bottom: 1px solid #ebebeb;
  border-right: 1px solid #ebebeb;
  text-align: center;
  white-space: nowrap;
}
tbody td.lbl { text-align: left; }
tbody td:last-child { border-right: none; }
tbody tr:last-child td { border-bottom: none; }

tr.bench-row td {
  background: #e8edf8;
  font-weight: 700;
  font-size: 13px;
}
tr.cat-row td {
  background: #d8e2f5;
  font-weight: 700;
  font-size: 13px;
}
tr.cat-row.expandable { cursor: pointer; }
tr.cat-row.expandable:hover td { background: #c8d4ec; }
tr.task-row.expandable { cursor: pointer; }
tr.task-row.expandable:hover td { background: #f5f7fb; }
.toggle { display: inline-block; width: 14px; font-size: 10px; color: #999; }
tr.mode-row td { background: #fafbfd; }
tr.mode-row td.lbl {
  padding-left: 38px;
  font-size: 13px;
  color: #666;
}
td.best { font-weight: 700; color: #1a6b2a; }
tr.bench-row td.best { color: #1a3a8a; }
tr.cat-row td.best { color: #1a4580; }
td.na { color: #ccc; }
.ci { font-size: 11px; color: #888; font-weight: 400; }
.n-badge {
  font-size: 11px;
  color: #aaa;
  font-weight: 400;
  margin-left: 5px;
}
"""

JS = r"""
const PCT_METRICS = new Set(["mean_edit_accuracy", "mean_preservation_accuracy", "mean_iou"]);
const COUNT_METRICS = { n_with_output: true, n_correct_output_size: true };

// Map percentage metrics to their bootstrap-CI sibling fields. Rendering of
// the [lo, hi] bracket is conditional on both fields being present in the
// row block — so `make stats` (which writes CIs by default) and
// `make stats NO_CI=1` (which omits the ci95_* fields) both render
// correctly without the report needing to know which mode produced the
// input file.
const CI_MAP = {
  mean_edit_accuracy:         ["ci95_edit_accuracy_low",         "ci95_edit_accuracy_high"],
  mean_preservation_accuracy: ["ci95_preservation_accuracy_low", "ci95_preservation_accuracy_high"],
  mean_iou:                   ["ci95_iou_low",                   "ci95_iou_high"],
};

function pct(v) { return (v * 100).toFixed(1) + "%"; }

function findBest(vals) {
  const valid = vals.filter(v => v !== null);
  if (!valid.length) return new Set();
  const max = Math.max(...valid);
  const s = new Set();
  vals.forEach((v, i) => { if (v === max) s.add(i); });
  return s;
}

function cellInfo(modelMetrics, de, metric) {
  if (!modelMetrics) return { html: "—", val: null };
  if (COUNT_METRICS[metric]) {
    const v  = modelMetrics[metric];
    const np = modelMetrics.n_problems;
    if (v === null || v === undefined) return { html: "—", val: null };
    const html = np ? `${v} / ${np}` : `${v}`;
    return { html, val: v };
  }
  if (!modelMetrics[de]) return { html: "—", val: null };
  const v = modelMetrics[de][metric];
  if (v === null || v === undefined) return { html: "—", val: null };
  let html = PCT_METRICS.has(metric) ? pct(v) : v.toFixed(1);
  const ciKeys = CI_MAP[metric];
  if (ciKeys) {
    const lo = modelMetrics[de][ciKeys[0]];
    const hi = modelMetrics[de][ciKeys[1]];
    if (lo != null && hi != null) {
      html += ` <span class="ci">[${pct(lo)}, ${pct(hi)}]</span>`;
    }
  }
  return { html, val: v };
}

function buildRow(rowCls, label, badge, rowDataByModel, de, metric, extraAttrs) {
  const cells = DATA.models.map(m => cellInfo(rowDataByModel[m] || null, de, metric));
  const best  = findBest(cells.map(c => c.val));
  let html = `<tr class="${rowCls}" ${extraAttrs || ""}>`;
  const badgeStr = badge ? `<span class="n-badge">${badge}</span>` : "";
  html += `<td class="lbl">${label}${badgeStr}</td>`;
  cells.forEach((c, i) => {
    const cls = c.val === null ? "na" : best.has(i) ? "best" : "";
    html += `<td class="${cls}">${c.html}</td>`;
  });
  return html + `</tr>`;
}

function _nBadge(rowDataForFirstModel) {
  const d  = rowDataForFirstModel || {};
  const nt = d.n_tasks;
  const np = d.n_problems;
  if (nt !== undefined && nt !== null) {
    return np ? `n=${nt} tasks · ${np} problems` : `n=${nt} tasks`;
  }
  return np ? `n=${np}` : "";
}

// ── Toggle helpers ──────────────────────────────────────────────────────────

// taskTids: array of task tids belonging to this category, used to also
// collapse any expanded task→mode subrows when the category closes.
function toggleCat(cid, taskTids) {
  const rows = document.querySelectorAll(`tr[data-cid="${cid}"]`);
  const tog  = document.getElementById(`tog-${cid}`);
  const open = tog.textContent.trim() === "▼";
  tog.textContent = open ? "▶" : "▼";
  rows.forEach(r => { r.style.display = open ? "none" : "table-row"; });
  if (open) {
    taskTids.forEach(tid => {
      document.querySelectorAll(`tr[data-tid="${tid}"]`).forEach(mr => {
        mr.style.display = "none";
      });
      const tt = document.getElementById(`tog-${tid}`);
      if (tt) tt.textContent = "▶";
    });
  }
}

function toggleTask(tid) {
  const rows = document.querySelectorAll(`tr[data-tid="${tid}"]`);
  const tog  = document.getElementById(`tog-${tid}`);
  const open = tog.textContent.trim() === "▼";
  tog.textContent = open ? "▶" : "▼";
  rows.forEach(r => { r.style.display = open ? "none" : "table-row"; });
}

// ── Render helpers ──────────────────────────────────────────────────────────

function _header(firstColLabel) {
  let html = `<table><thead><tr><th class="lbl">${firstColLabel}</th>`;
  DATA.models.forEach(m => { html += `<th>${m}</th>`; });
  return html + `</tr></thead><tbody>`;
}

function _benchRow(benchmark, cie76, metric) {
  const bl = DATA.bench_level[benchmark] || {};
  return buildRow("bench-row", "Overall", _nBadge(bl[DATA.models[0]]), bl, cie76, metric, "");
}

function _taskRows(benchmark, task, cie76, metric, extraRowAttrs) {
  const tl       = DATA.task_level[benchmark]?.[task] || {};
  const modes    = (DATA.mode_order[benchmark] || {})[task] || [];
  const hasModes = modes.length > 0;
  const tid      = `${benchmark}__${task}`.replace(/[^a-zA-Z0-9]/g, "_");
  const toggle   = hasModes
    ? `<span class="toggle" id="tog-${tid}">▶</span> `
    : `<span class="toggle"> </span> `;
  const onclick  = hasModes ? `onclick="toggleTask('${tid}')"` : "";
  const rowCls   = "task-row" + (hasModes ? " expandable" : "");
  let html = buildRow(rowCls, toggle + task, _nBadge(tl[DATA.models[0]]), tl, cie76, metric,
                      `${extraRowAttrs} ${onclick}`.trim());
  // Mode rows start hidden; toggleTask reveals them.
  modes.forEach(mode => {
    const ml = DATA.mode_level[benchmark]?.[task]?.[mode] || {};
    html += buildRow("mode-row", mode, _nBadge(ml[DATA.models[0]]), ml, cie76, metric,
                     `data-tid="${tid}" style="display:none"`);
  });
  return html;
}

// ── Per-benchmark renderers ─────────────────────────────────────────────────

function renderCategoryTable(benchmark, cie76, metric) {
  const catOrd = DATA.cat_order[benchmark]    || [];
  const c2t    = DATA.cat_to_tasks[benchmark] || {};
  let html = _header("Category / Task / Mode") + _benchRow(benchmark, cie76, metric);

  catOrd.forEach(cat => {
    const cl     = DATA.category_level[benchmark]?.[cat] || {};
    const cid    = `${benchmark}__${cat}`.replace(/[^a-zA-Z0-9]/g, "_");
    const tasks  = c2t[cat] || [];
    const tids   = tasks.map(t => `${benchmark}__${t}`.replace(/[^a-zA-Z0-9]/g, "_"));
    const tidsJs = "[" + tids.map(t => `'${t}'`).join(",") + "]";
    const lbl    = `<span class="toggle" id="tog-${cid}">▶</span> ${cat.replace(/_/g, " ")}`;
    html += buildRow("cat-row expandable", lbl, _nBadge(cl[DATA.models[0]]), cl, cie76, metric,
                     `onclick="toggleCat('${cid}', ${tidsJs})"`);
    // Task rows start hidden; toggleCat reveals them.
    tasks.forEach(task => {
      html += _taskRows(benchmark, task, cie76, metric,
                        `data-cid="${cid}" style="display:none"`);
    });
  });
  return `<div class="table-wrap">` + html + `</tbody></table></div>`;
}

function renderFlatTable(benchmark, cie76, metric) {
  let html = _header("Task / Mode") + _benchRow(benchmark, cie76, metric);
  (DATA.task_order[benchmark] || []).forEach(task => {
    html += _taskRows(benchmark, task, cie76, metric, "");
  });
  return `<div class="table-wrap">` + html + `</tbody></table></div>`;
}

function renderVisualConditionTable(benchmark, cie76, metric) {
  const vcondOrd = DATA.vcond_order[benchmark] || [];
  if (!vcondOrd.length) return "";
  let html = _header("Visual condition");
  vcondOrd.forEach(vcond => {
    const cl = DATA.visual_condition_level[benchmark]?.[vcond] || {};
    html += buildRow("task-row", vcond, _nBadge(cl[DATA.models[0]]), cl, cie76, metric, "");
  });
  return `<h3>Visual conditions</h3><div class="table-wrap">` + html + `</tbody></table></div>`;
}

function renderAll() {
  const cie76  = document.getElementById("cie76-sel").value;
  const metric = document.getElementById("metric-sel").value;
  let html = "";
  DATA.benchmarks.forEach(b => {
    html += `<h2>${b}</h2>`;
    const hasCats = (DATA.cat_order[b] || []).length > 0;
    html += hasCats
      ? renderCategoryTable(b, cie76, metric)
      : renderFlatTable(b, cie76, metric);
    html += renderVisualConditionTable(b, cie76, metric);
  });
  document.getElementById("report").innerHTML = html;
}

function onMetricChange() {
  const isCount = COUNT_METRICS[document.getElementById("metric-sel").value];
  document.getElementById("cie76-sel").disabled = !!isCount;
  renderAll();
}

document.getElementById("cie76-sel").addEventListener("change", renderAll);
document.getElementById("metric-sel").addEventListener("change", onMetricChange);
renderAll();
"""

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PaintBench Report</title>
<style>__CSS__</style>
</head>
<body>
<h1>PaintBench Evaluation Report</h1>

<div class="controls">
  <div class="ctrl">
    <label for="cie76-sel">CIE76 Threshold</label>
    <select id="cie76-sel">
      <option value="cie76_0">CIE76 = 0 (exact match)</option>
      <option value="cie76_1">CIE76 = 1</option>
      <option value="cie76_2">CIE76 = 2</option>
      <option value="cie76_3">CIE76 = 3</option>
      <option value="cie76_4">CIE76 = 4</option>
      <option value="cie76_5">CIE76 = 5</option>
      <option value="cie76_6">CIE76 = 6</option>
      <option value="cie76_7">CIE76 = 7</option>
      <option value="cie76_8">CIE76 = 8</option>
      <option value="cie76_9">CIE76 = 9</option>
      <option value="cie76_10">CIE76 = 10</option>
      <option disabled>──────────────</option>
      <option value="cie76_mean" selected>CIE76 mean (avg 0–10)</option>
    </select>
  </div>
  <div class="ctrl">
    <label for="metric-sel">Metric</label>
    <select id="metric-sel">
      <option value="mean_edit_accuracy">Edit Accuracy</option>
      <option value="mean_preservation_accuracy">Preservation Accuracy</option>
      <option value="mean_iou" selected>Mean IoU</option>
      <option disabled>──────────────</option>
      <option value="n_with_output">Output Count</option>
      <option value="n_correct_output_size">Correct Size Count</option>
    </select>
  </div>
</div>

<div id="report"></div>

<script>
const DATA = __DATA__;
__JS__
</script>
</body>
</html>
"""


def generate_html(data: dict) -> str:
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return (HTML_TEMPLATE
            .replace("__CSS__",  CSS)
            .replace("__DATA__", data_json)
            .replace("__JS__",   JS))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="eval_outputs/aggregate_stats.jsonl")
    parser.add_argument("--output", default="report.html")
    parser.add_argument(
        "--exclude-models", default="",
        help="Comma-separated substrings; rows whose model name contains "
             "any of them are dropped from the rendered HTML. Underlying "
             "aggregate_stats.jsonl is untouched.",
    )
    args = parser.parse_args()

    exclude = tuple(s.strip() for s in args.exclude_models.split(",") if s.strip())
    data = load(args.input, exclude)
    html = generate_html(data)
    Path(args.output).write_text(html, encoding="utf-8")
    n_models = len(data["models"])
    suffix = f", excluded substrings: {list(exclude)}" if exclude else ""
    print(f"Wrote {args.output}  ({Path(args.output).stat().st_size // 1024} KB, "
          f"{n_models} models{suffix})")


if __name__ == "__main__":
    main()
