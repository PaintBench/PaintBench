"""PaintBench — benchmark generation script.

PaintBench: 20 tasks × 8 visual conditions × 12 problems-per-visual-condition = 1920 problems.
Each problem's mode and visual_condition are stored in its .json metadata; task folders use
the plain task name (e.g. ``removal/``, ``pattern/``).

A ``preservation/`` diagnostic folder is also generated alongside the main tasks.
It reuses the same 96 inputs as ``removal/`` (same seeds → same images) with
answer = input, and is excluded from main benchmark scoring in stats.py.

Visual conditions
-----------------
Eight parameter configurations, each varying exactly one axis from the baseline:

  baseline    — 1024²,  standard palette,    solid bg,   n = n_default
  horizontal  — 1024×576, standard palette,  solid bg,   n = n_default
  vertical    — 576×1024, standard palette,  solid bg,   n = n_default
  nonstandard — 1024²,  nonstandard palette, solid bg,   n = n_default
  striped     — 1024²,  standard palette,    striped bg, n = n_default
  n_med       — 1024²,  standard palette,    solid bg,   n = n_med
  n_high      — 1024²,  standard palette,    solid bg,   n = n_high
  n_xhigh     — 1024²,  standard palette,    solid bg,   n = n_xhigh

n-level tables (default / med / high / xhigh):
  regular tasks       — 3 / 10 / 25 / 60
  counting            — 5 / 10 / 25 / 60
  comparison/ordering — 3 /  5 /  7 /  9
  pattern             — 1 /  3 /  6 / 10

Seed design
-----------
Every problem gets its own independent seed encoding (task, visual_condition, mode, slot).
Seeds are SHA-256-based and immune to Python hash randomisation.

Problems per task
-----------------
_N_PROBLEMS = 12 problems per (task, visual condition), split evenly across modes.
For a task with k modes: _N_PROBLEMS // k seed-slots per (visual_condition, mode) group.
LCM(1, 2, 3) = 6 divides 12, so tasks with up to 3 modes produce whole numbers.

Within each task folder problems are ordered by visual_condition first, then mode within
each visual_condition group.  The ``visual_condition``, ``mode``, and ``seed`` fields in each
.json identify the problem fully and enable downstream slicing or matched analysis.

Parallelism
-----------
Seed search and rendering are parallelised across all (visual_condition × mode × slot)
combinations for each task at once.  With --jobs N the work is dispatched to a
process pool; output is deterministic regardless of N.

Usage
-----
    python src/generate_benchmark.py --paintbench       [--output DIR] [--jobs N]
    python src/generate_benchmark.py --tinygrafixbench  [--output DIR] [--jobs N]
    python src/generate_benchmark.py --paintbench --tinygrafixbench
    # default output: benchmarks/

    # Single-problem preview
    python src/generate_benchmark.py --task removal --mode attribute --seed 42
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import hashlib
import importlib
import json
import os
import random
import shutil
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from core.background import BackgroundSpec
from core.colors import STANDARD_PALETTE, NONSTANDARD_PALETTE

# ── Task registry ──────────────────────────────────────────────────────────────

TASKS: list[tuple[str, str]] = [
    ("tasks.translation",      "translation"),
    ("tasks.rotation",         "rotation"),
    ("tasks.reflection",       "reflection"),
    ("tasks.scaling",          "scaling"),
    ("tasks.shearing",         "shearing"),
    ("tasks.construction",     "construction"),
    ("tasks.removal",          "removal"),
    ("tasks.copying",          "copying"),
    ("tasks.border",           "border"),
    ("tasks.cropping",         "cropping"),
    ("tasks.recolor",          "recolor"),
    ("tasks.flood_fill",       "flood_fill"),
    ("tasks.blending",         "blending"),
    ("tasks.gradient",         "gradient"),
    ("tasks.point_operations", "point_operations"),
    ("tasks.comparison",       "comparison"),
    ("tasks.ordering",         "ordering"),
    ("tasks.pattern",          "pattern"),
    ("tasks.counting",         "counting"),
    ("tasks.legend",           "legend"),
]

TASK_CATEGORIES: dict[str, str] = {
    "translation":      "geometric_transformation",
    "rotation":         "geometric_transformation",
    "reflection":       "geometric_transformation",
    "scaling":          "geometric_transformation",
    "shearing":         "geometric_transformation",
    "construction":     "structural_manipulation",
    "removal":          "structural_manipulation",
    "copying":          "structural_manipulation",
    "border":           "structural_manipulation",
    "cropping":         "structural_manipulation",
    "recolor":          "color_change",
    "flood_fill":       "color_change",
    "blending":         "color_change",
    "gradient":         "color_change",
    "point_operations": "color_change",
    "comparison":       "symbolic_reasoning",
    "ordering":         "symbolic_reasoning",
    "pattern":          "symbolic_reasoning",
    "counting":         "symbolic_reasoning",
    "legend":           "symbolic_reasoning",
}

# ── Visual conditions ──────────────────────────────────────────────────────────

# 8 visual conditions: each varies exactly one axis from baseline.
VISUAL_CONDITIONS: list[dict] = [
    {"name": "baseline",    "W": 1024, "H": 1024, "palette": None,          "striped": False, "n_level": 0},
    {"name": "horizontal",  "W": 1024, "H": 576,  "palette": None,          "striped": False, "n_level": 0},
    {"name": "vertical",    "W": 576,  "H": 1024, "palette": None,          "striped": False, "n_level": 0},
    {"name": "nonstandard", "W": 1024, "H": 1024, "palette": "nonstandard", "striped": False, "n_level": 0},
    {"name": "striped",     "W": 1024, "H": 1024, "palette": None,          "striped": True,  "n_level": 0},
    {"name": "n_med",       "W": 1024, "H": 1024, "palette": None,          "striped": False, "n_level": 1},
    {"name": "n_high",      "W": 1024, "H": 1024, "palette": None,          "striped": False, "n_level": 2},
    {"name": "n_xhigh",     "W": 1024, "H": 1024, "palette": None,          "striped": False, "n_level": 3},
]

# n-value lookup: index = n_level (0 = default, 1 = med, 2 = high, 3 = xhigh)
_N_DEFAULT  = [3, 10, 25, 60]
_N_PATTERN  = [1,  3,  6, 10]
_N_CMP_ORD  = [3,  5,  7,  9]
_N_COUNTING = [5, 10, 25, 60]  # default raised to 5: n=3 gives only a binary removal choice

# 12 problems per (task, visual condition); modes split this evenly.
# LCM(1, 2, 3) = 6 divides 12 — tasks with up to 3 modes produce whole numbers.
_N_PROBLEMS     = 12
_N_TGF_PROBLEMS = 30   # TinyGrafixBench: unchanged

# Single-mode PaintBench tasks (blending, border, comparison, ...) and all TGF
# tasks have no mode dimension. We label them "default" in the on-disk JSON
# (and downstream in problem_stats / aggregate_stats / HF dataset) rather than
# leaving the field as JSON null / Python None — uniform labelling means
# filter calls and results tables don't need to special-case the missing-mode
# row. Coercion happens only at the JSON-write boundary; the seed-hash key
# already substitutes "" for None in _search_slot (``mode_str = mode or ""``
# before the _make_seed call), so the on-disk relabel doesn't touch the hash
# input — scenes for any given (task, vcond, slot) tuple are byte-identical
# to those generated before this constant existed.
_NO_MODE_LABEL = "default"

# Set from CLI (--jobs) in main().  Each worker runs one seed search or render;
# determinism is preserved regardless of worker count.
_JOBS: int = 1


def _parallel_map(fn, items: list) -> list:
    """Run fn over items, preserving input order.  Sequential when _JOBS <= 1."""
    items = list(items)
    if _JOBS <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with cf.ProcessPoolExecutor(max_workers=_JOBS) as ex:
        return list(ex.map(fn, items))

# ── Seeds ──────────────────────────────────────────────────────────────────────

def _make_seed(task: str, cond_name: str, mode: str, slot: int, attempt: int) -> int:
    """SHA-256 seed unique to one (task, visual_condition, mode, slot)."""
    key = f"paintbench|{task}|{cond_name}|{mode}|{slot}|{attempt}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)

# ── Condition helpers ──────────────────────────────────────────────────────────

def _n_levels_for(task_name: str) -> list[int]:
    if task_name in ("comparison", "ordering"):
        return _N_CMP_ORD
    if task_name == "pattern":
        return _N_PATTERN
    if task_name == "counting":
        return _N_COUNTING
    return _N_DEFAULT


def _visual_condition_config(cond: dict, task_name: str) -> tuple:
    """Resolve (visual_condition, task) → (W, H, palette_dict, n, striped)."""
    pal = NONSTANDARD_PALETTE if cond["palette"] == "nonstandard" else STANDARD_PALETTE
    n   = _n_levels_for(task_name)[cond["n_level"]]
    return cond["W"], cond["H"], pal, n, cond["striped"]

# ── Color / background helpers ─────────────────────────────────────────────────

def _color_split(palette: dict, seed: int) -> tuple[tuple, tuple, list]:
    """Deterministic shuffle → (bg_rgb, holdout_rgb, obj_colors)."""
    rng   = random.Random(seed ^ 0xC0FFEE)
    items = list(palette.items())
    rng.shuffle(items)
    return items[0][1], items[1][1], [rgb for _, rgb in items[2:]]


def _striped_bg(bg_rgb: tuple, holdout_rgb: tuple, seed: int) -> BackgroundSpec:
    rng       = random.Random(seed ^ 0x571ACED)
    waveform  = rng.choice(["line", "sine", "square", "triangle", "sawtooth"])
    rotation  = rng.choice([0.0, 45.0, 90.0])
    band_w    = rng.choice([0.06, 0.08, 0.10])
    amplitude = rng.uniform(0.01, 0.04) if waveform != "line" else 0.0
    frequency = rng.uniform(4.0,  10.0) if waveform != "line" else 1.0
    return BackgroundSpec(
        colors=[bg_rgb, holdout_rgb],
        band_width=band_w, waveform=waveform, rotation=rotation,
        amplitude=amplitude, frequency=frequency,
    )

# ── Validation ─────────────────────────────────────────────────────────────────

def _valid(prob) -> bool:
    return prob is not None and not prob.error

# ── Single-problem generation ──────────────────────────────────────────────────

def _gen(mod, mode: str | None, seed: int,
         W: int, H: int, palette: dict, n: int, striped: bool = False):
    """Generate one problem; returns None on failure.

    Expected validation failures (RuntimeError / ValueError) print a dot.
    Unexpected exceptions (programming bugs) print a full traceback so they
    are not silently swallowed during a long parallel run.
    """
    try:
        bg_rgb, holdout_rgb, obj_colors = _color_split(palette, seed)
        bg_spec = (_striped_bg(bg_rgb, holdout_rgb, seed)
                   if striped else BackgroundSpec(colors=[bg_rgb]))
        kwargs  = {"n_min": n, "n_max": n}
        if mode is not None:
            kwargs["mode"] = mode
        return mod.generate(seed=seed, bg_spec=bg_spec, W=W, H=H,
                            obj_colors=obj_colors, **kwargs)
    except (RuntimeError, ValueError):
        print(".", end="", flush=True)
        return None
    except Exception:
        import traceback
        traceback.print_exc()
        return None

# ── Saving ─────────────────────────────────────────────────────────────────────

def _to_json(v):
    if isinstance(v, tuple): return list(v)
    if isinstance(v, dict):  return {k: _to_json(vv) for k, vv in v.items()}
    if isinstance(v, list):  return [_to_json(x) for x in v]
    return v


def _save(prob, prefix: str, meta: dict) -> None:
    prob.input_image.save(f"{prefix}_input.png")
    prob.answer_image.save(f"{prefix}_answer.png")
    with open(f"{prefix}.json", "w") as f:
        json.dump({"instruction": prob.instruction,
                   **_to_json(prob.metadata), **meta}, f, indent=2)

# ── Seed search ───────────────────────────────────────────────────────────────

def _search_slot(args) -> int | None:
    """Worker: find the lowest-attempt seed valid for one (task, visual_condition, mode, slot)."""
    task_name, cond, mode, slot = args
    mod = importlib.import_module(f"tasks.{task_name}")
    W, H, pal, n, striped = _visual_condition_config(cond, task_name)
    mode_str = mode or ""
    for attempt in range(200):
        seed = _make_seed(task_name, cond["name"], mode_str, slot, attempt)
        if _valid(_gen(mod, mode, seed, W, H, pal, n, striped)):
            return seed
    return None

# ── Render ────────────────────────────────────────────────────────────────────

def _render_main(args) -> dict | None:
    task_name, mode, task_dir, pid, seed, cond = args
    W, H, pal, n, striped = _visual_condition_config(cond, task_name)
    mod  = importlib.import_module(f"tasks.{task_name}")
    prob = _gen(mod, mode, seed, W, H, pal, n, striped)
    if not _valid(prob):
        return None
    prefix   = os.path.join(task_dir, f"{pid:03d}")
    category = TASK_CATEGORIES[task_name]
    _save(prob, prefix, {
        "category":   category,
        "task":       task_name,
        "mode":       mode or _NO_MODE_LABEL,
        "visual_condition":  cond["name"],
        "problem_id": pid,
        "seed":       seed,
        "n":          n,
        "W":          W,
        "H":          H,
    })
    return {
        "category":    category,
        "task":        task_name,
        "mode":        mode or _NO_MODE_LABEL,
        "visual_condition":   cond["name"],
        "problem_id":  pid,
        "instruction": prob.instruction,
    }

# ── Preservation render ───────────────────────────────────────────────────────

_PRESERVATION_INSTRUCTION = "Generate the exact same image as the input."


def _render_preservation(args) -> dict | None:
    """Write one preservation problem by copying the removal input (answer = input)."""
    task_dir, pid, removal_input, seed, mode, cond_name, n, W, H = args
    prefix = os.path.join(task_dir, f"{pid:03d}")
    try:
        shutil.copy2(removal_input, f"{prefix}_input.png")
        shutil.copy2(removal_input, f"{prefix}_answer.png")
    except OSError:
        return None
    with open(f"{prefix}.json", "w") as f:
        json.dump({
            "instruction": _PRESERVATION_INSTRUCTION,
            "task":        "preservation",
            "mode":        mode or _NO_MODE_LABEL,
            "visual_condition":   cond_name,
            "problem_id":  pid,
            "seed":        seed,
            "n":           n,
            "W":           W,
            "H":           H,
        }, f, indent=2)
    return {
        "task":        "preservation",
        "mode":        mode or _NO_MODE_LABEL,
        "visual_condition":   cond_name,
        "problem_id":  pid,
        "instruction": _PRESERVATION_INSTRUCTION,
    }


def generate_preservation(paintbench_dir: str) -> list[dict]:
    """Generate preservation task inside a PaintBench directory.

    Copies the removal/ input images (answer = input) — no re-rendering.
    Not included in main benchmark scoring.
    Requires removal/ to already exist under paintbench_dir.
    """
    removal_dir = os.path.join(paintbench_dir, "removal")
    if not os.path.isdir(removal_dir):
        print("  [preservation] skipping — removal/ not found; run generate_paintbench first")
        return []

    cond_by_name = {c["name"]: c for c in VISUAL_CONDITIONS}
    pres_dir     = os.path.join(paintbench_dir, "preservation")
    os.makedirs(pres_dir, exist_ok=True)

    jobs: list = []
    for fname in sorted(os.listdir(removal_dir)):
        if not fname.endswith(".json"):
            continue
        pid_str = fname[:-5]                        # strip ".json"
        if not pid_str.isdigit():
            continue
        with open(os.path.join(removal_dir, fname)) as f:
            removal_meta = json.load(f)
        seed      = removal_meta.get("seed")
        mode      = removal_meta.get("mode")
        cond_name = removal_meta.get("visual_condition", "baseline")
        cond      = cond_by_name.get(cond_name)
        if seed is None or cond is None:
            continue
        W, H, _, n, _ = _visual_condition_config(cond, "removal")
        removal_input  = os.path.join(removal_dir, f"{pid_str}_input.png")
        jobs.append((pres_dir, int(pid_str), removal_input, seed, mode, cond_name, n, W, H))

    print(f"  [preservation] copying {len(jobs)} problems...", end=" ", flush=True)
    results  = _parallel_map(_render_preservation, jobs)
    ok       = sum(1 for r in results if r is not None)
    all_meta = [r for r in results if r is not None]
    print(f"{ok}/{len(jobs)}")
    return all_meta


# ── Targeted regeneration ─────────────────────────────────────────────────────

def _parse_regenerate_spec(spec: str) -> list[tuple[str, int]]:
    """Parse a --regenerate argument like 'removal:11,pattern:5'."""
    pairs: list[tuple[str, int]] = []
    for raw in spec.split(","):
        entry = raw.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(f"--regenerate entry {entry!r} missing ':' (expected task:id)")
        task, idx_str = entry.rsplit(":", 1)
        try:
            idx = int(idx_str)
        except ValueError as exc:
            raise ValueError(f"--regenerate entry {entry!r}: id must be an integer") from exc
        pairs.append((task.strip(), idx))
    return pairs


def _regenerate_problems(output_dir: str, problems: list[tuple[str, int]]) -> int:
    """Regenerate specific (task, problem_id) pairs by replaying recorded seeds.

    Reads visual_condition/mode/seed from the existing .json sidecar and re-renders.
    No seed search; only targeted problems are touched.
    """
    task_names   = {t for _, t in TASKS}
    cond_by_name = {c["name"]: c for c in VISUAL_CONDITIONS}
    n_ok = 0
    for task_name, pid in problems:
        if task_name not in task_names:
            print(f"  [SKIP] {task_name}/{pid:03d}: unknown task")
            continue
        task_dir  = os.path.join(output_dir, task_name)
        json_path = os.path.join(task_dir, f"{pid:03d}.json")
        if not os.path.exists(json_path):
            print(f"  [SKIP] {task_name}/{pid:03d}: no .json (problem doesn't exist)")
            continue
        with open(json_path) as f:
            meta = json.load(f)
        if "seed" not in meta:
            print(f"  [SKIP] {task_name}/{pid:03d}: no seed in metadata")
            continue
        cond = cond_by_name.get(meta.get("visual_condition", "baseline"))
        if cond is None:
            print(f"  [SKIP] {task_name}/{pid:03d}: unknown visual_condition {meta.get('visual_condition')!r}")
            continue
        result = _render_main(
            (task_name, meta.get("mode"), task_dir, pid, meta["seed"], cond)
        )
        if result is None:
            print(f"  [FAIL] {task_name}/{pid:03d}: render failed (seed={meta['seed']})")
            continue
        print(f"  [OK]   {task_name}/{pid:03d}  seed={meta['seed']} visual_condition={meta.get('visual_condition')}")
        n_ok += 1
    return n_ok

# ── PaintBench ─────────────────────────────────────────────────────────────────

def generate_paintbench(output_dir: str) -> None:
    """Generate PaintBench: 20 tasks × 8 visual conditions × 12 problems-per-visual-condition = 1920 problems,
    plus a ``preservation/`` diagnostic folder (96 problems, excluded from aggregate scoring).

    Each task gets one folder named by task (e.g. ``removal/``).  For a task
    with k modes, each (visual_condition, mode) group gets _N_PROBLEMS // k seed-slots.
    All seed searches and renders for a task are dispatched in one parallel batch.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_meta: list[dict] = []

    for mod_name, task_name in TASKS:
        mod   = importlib.import_module(mod_name)
        modes = mod.PARAMETERS.get("mode") or [None]
        if _N_PROBLEMS % len(modes) != 0:
            raise ValueError(
                f"{task_name}: _N_PROBLEMS={_N_PROBLEMS} not divisible by {len(modes)} modes"
            )
        n_seeds  = _N_PROBLEMS // len(modes)  # slots per (visual_condition, mode) group
        task_dir = os.path.join(output_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)

        # Build all (visual_condition, mode, slot) combos and dispatch seed search in one batch
        combos = [
            (ci, cond, mi, mode, slot)
            for ci, cond in enumerate(VISUAL_CONDITIONS)
            for mi, mode in enumerate(modes)
            for slot in range(n_seeds)
        ]
        print(f"  [{task_name}] searching {len(combos)} seeds...", end="", flush=True)
        found = _parallel_map(
            _search_slot,
            [(task_name, cond, mode, slot) for _, cond, _, mode, slot in combos],
        )

        # Build render jobs; pid = cond_index × _N_PROBLEMS + mode_index × n_seeds + slot.
        # Failed seed slots are aggregated into one summary line per task instead
        # of one print per failure, so the per-task output stays single-line.
        render_jobs = []
        fail_counts: dict[str, int] = defaultdict(int)
        for (ci, cond, mi, mode, slot), seed in zip(combos, found):
            if seed is None:
                # Modeless tasks omit the "/mode" suffix so the summary key
                # reads "nonstandard" rather than "nonstandard/".
                key = f"{cond['name']}/{mode}" if mode else cond["name"]
                fail_counts[key] += 1
                continue
            pid = ci * _N_PROBLEMS + mi * n_seeds + slot
            render_jobs.append((task_name, mode, task_dir, pid, seed, cond))

        if fail_counts:
            n_failed = sum(fail_counts.values())
            parts = ", ".join(f"{k}×{v}" for k, v in sorted(fail_counts.items()))
            print(f" [skip {n_failed}/{len(combos)}: {parts}]", end="", flush=True)

        print(" rendering...", end=" ", flush=True)
        results  = _parallel_map(_render_main, render_jobs)
        ok       = sum(1 for r in results if r is not None)
        all_meta.extend(r for r in results if r is not None)
        print(f"{ok}/{len(render_jobs)}")

    all_meta.extend(generate_preservation(output_dir))

    with open(os.path.join(output_dir, "problems.jsonl"), "w") as f:
        for item in sorted(all_meta, key=lambda x: (x["task"], x["problem_id"])):
            f.write(json.dumps(item) + "\n")
    print(f"\nPaintBench: {len(all_meta)} problems → {output_dir}")

# ── TinyGrafixBench ────────────────────────────────────────────────────────────

_TGF_GRAPHS = ["bar_chart", "heatmap", "line_chart", "network", "scatter_plot"]


def _tgf_seed(graph: str, task: str, slot: int) -> int:
    key = f"tinygrafixbench|{graph}|{task}|{slot}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)


def _render_tgf(args) -> dict | None:
    import matplotlib.pyplot as plt
    graph, task, folder, task_dir, pid, seed = args
    try:
        mod = importlib.import_module(f"tinygrafixbench.{graph}")
        input_fig, answer_fig, instruction = mod.generate_task(seed, task)
    except (RuntimeError, ValueError):
        print(".", end="", flush=True)
        return None
    except Exception:
        import traceback
        traceback.print_exc()
        return None
    prefix = os.path.join(task_dir, f"{pid:03d}")
    input_fig.savefig(f"{prefix}_input.png")
    answer_fig.savefig(f"{prefix}_answer.png")
    plt.close(input_fig)
    plt.close(answer_fig)
    # Chart type is the category; the chart-edit subtask is the task.
    # eval.py uses the folder name (bar_chart_edit_bars) as the task key;
    # stats.py strips the chart prefix to recover the subtask label.
    with open(f"{prefix}.json", "w") as f:
        json.dump({
            "category":    graph,
            "task":        folder,
            "mode":        _NO_MODE_LABEL,
            "problem_id":  pid,
            "seed":        seed,
            "instruction": instruction,
        }, f, indent=2)
    return {
        "category":    graph,
        "task":        folder,
        "mode":        _NO_MODE_LABEL,
        "problem_id":  pid,
        "instruction": instruction,
    }


def generate_tinygrafixbench(output_dir: str) -> None:
    """Generate TinyGrafixBench: _N_TGF_PROBLEMS problems per (chart, task).

    Layout and seed design are unchanged from the original; _N_TGF_PROBLEMS
    is independent of PaintBench's _N_PROBLEMS.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_meta: list[dict] = []

    for graph in _TGF_GRAPHS:
        mod = importlib.import_module(f"tinygrafixbench.{graph}")
        for task in mod.TASKS:
            folder   = f"{graph}_{task}"
            task_dir = os.path.join(output_dir, folder)
            os.makedirs(task_dir, exist_ok=True)

            jobs = [(graph, task, folder, task_dir, pid, _tgf_seed(graph, task, pid))
                    for pid in range(_N_TGF_PROBLEMS)]
            results = _parallel_map(_render_tgf, jobs)

            ok = sum(1 for r in results if r is not None)
            all_meta.extend(r for r in results if r is not None)
            print(f"  {folder}:{ok}", flush=True)

    with open(os.path.join(output_dir, "problems.jsonl"), "w") as f:
        for item in all_meta:
            f.write(json.dumps(item) + "\n")
    print(f"\nTinyGrafixBench: {len(all_meta)} problems → {output_dir}")

# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="PaintBench generation script")
    ap.add_argument("--paintbench",      action="store_true", help="Generate PaintBench")
    ap.add_argument("--tinygrafixbench", action="store_true", help="Generate TinyGrafixBench")
    ap.add_argument("--output", default=None,
                    help="Base output directory (default: benchmarks/). "
                         "Each benchmark is saved as a named subfolder.")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                    help="Parallel worker processes (default: all CPUs). "
                         "Output is deterministic regardless of value.")
    ap.add_argument("--regenerate", default=None,
                    help="Regenerate specific problems by replaying recorded seeds. "
                         "Format: <task>:<id>[,<task>:<id>...] — e.g. 'removal:11,pattern:5'. "
                         "Requires --output to point at the PaintBench directory.")
    ap.add_argument("--task",  default=None, help="Task name for single-problem preview")
    ap.add_argument("--mode",  default=None, help="Task mode for preview")
    ap.add_argument("--seed",  type=int, default=42)
    ap.add_argument("--save",  default=None, help="Save preview images to this directory")
    args = ap.parse_args()

    global _JOBS
    _JOBS = max(1, args.jobs)

    if args.regenerate:
        if not args.output:
            ap.error("--regenerate requires --output (path to the PaintBench directory)")
        try:
            pairs = _parse_regenerate_spec(args.regenerate)
        except ValueError as exc:
            ap.error(str(exc))
        if not pairs:
            ap.error("--regenerate spec is empty")
        print(f"Regenerating {len(pairs)} problem(s) in {args.output}")
        n_ok = _regenerate_problems(args.output, pairs)
        print(f"\nDone: {n_ok}/{len(pairs)} regenerated successfully")
        return

    if args.paintbench or args.tinygrafixbench:
        base = args.output or "benchmarks"
        if args.paintbench:
            generate_paintbench(os.path.join(base, "PaintBench"))
        if args.tinygrafixbench:
            generate_tinygrafixbench(os.path.join(base, "TinyGrafixBench"))
        return

    if not args.task:
        ap.error("specify at least one of --paintbench, --tinygrafixbench, --task, or --regenerate")

    # Single-problem preview using baseline visual_condition defaults
    mod = importlib.import_module(f"tasks.{args.task}")
    baseline = VISUAL_CONDITIONS[0]
    W, H, pal, n, _ = _visual_condition_config(baseline, args.task)
    bg_rgb, _, obj_colors = _color_split(pal, args.seed)
    bg_spec = BackgroundSpec(colors=[bg_rgb])
    kwargs: dict = {"n_min": n, "n_max": n}
    if args.mode:
        kwargs["mode"] = args.mode
    prob = mod.generate(seed=args.seed, bg_spec=bg_spec, W=W, H=H,
                        obj_colors=obj_colors, **kwargs)
    print(f"Task:        {args.task}")
    print(f"Instruction: {prob.instruction}")
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        prob.input_image.save(os.path.join(args.save, "input.png"))
        prob.answer_image.save(os.path.join(args.save, "answer.png"))
        print(f"Saved → {args.save}")


if __name__ == "__main__":
    main()
