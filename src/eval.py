"""Evaluate model outputs using pointwise CIE76 (Lab Euclidean distance).
Pure numpy — no scipy/skimage/torch required.

Per-problem output (problem_stats.jsonl):
  input_size                                        — [W, H] of the input image
  output.output_size                                — [W, H] of model output (null if missing)
  output.edit_pixels                                — pixels that changed from input → answer
  output.preservation_pixels                        — pixels unchanged from input → answer
  output.cie76_threshold.{0..10}.edit_correct_pixels
  output.cie76_threshold.{0..10}.preservation_correct_pixels
  output.cie76_threshold.{0..10}.edit_accuracy
  output.cie76_threshold.{0..10}.preservation_accuracy
  output.cie76_threshold.{0..10}.iou               — edit_correct / (edit + preservation_incorrect)
  output.cie76_threshold.{0..10}.changed_pixels    — pixels where ΔE(output, input) > t

Caching:
  Per-problem stats are cached as
  ``<results>/<model>/<benchmark>/<task>/<NNNN>_stats.json`` sidecars.
  Reruns reuse the sidecar when its mtime is ≥ all source mtimes
  (input/answer/output PNGs + the per-problem metadata JSON). Pass
  ``--overwrite`` to invalidate. With ``--save-images`` (the default)
  the cache also requires the diagnostic diff PNGs to be present
  *when a model output exists*, so a ``make eval-quick`` → ``make eval``
  upgrade triggers a full rerun for the missing images (but the
  inverse is cache-clean). Problems with no model output cache
  on the sidecar mtime alone — there are no diff PNGs to gate on.

Usage:
    python src/eval.py \\
        --benchmarks benchmarks \\
        --model-outputs model_outputs \\
        --eval-outputs eval_outputs
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────

THRESHOLDS = list(range(11))   # CIE76 = 0, 1, 2, ..., 10

# Subset of THRESHOLDS that get a per-problem ΔE diff PNG written to disk
# under --save-images. Matches the four the JS Eval card displays
# (`_DE_THRESHOLDS = [0, 2, 5, 10, 50]` in src/visualize.py minus 50, which
# eval doesn't compute). Stats for ALL 11 thresholds still go into the
# sidecar / problem_stats.jsonl — this only narrows the diagnostic PNGs.
IMAGE_THRESHOLDS: Tuple[int, ...] = (0, 2, 5, 10)

_CC = np.array([  0, 128,   0], dtype=np.uint8)   # changed   & correct   (green)
_CU = np.array([  0,   0, 255], dtype=np.uint8)   # unchanged & correct   (blue)
_IC = np.array([255,   0,   0], dtype=np.uint8)   # changed   & incorrect (red)
_IU = np.array([255, 165,   0], dtype=np.uint8)   # unchanged & incorrect (orange)


# ── Colour conversion ─────────────────────────────────────────────────────────

def _rgb_to_lab(img: np.ndarray) -> np.ndarray:
    """Convert (H, W, 3) uint8 RGB array to CIE L*a*b*. Pure numpy."""
    rgb = img.astype(np.float32) / 255.0
    # sRGB → linear light
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    # Linear RGB → XYZ (D65 illuminant)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    xyz = linear @ M.T                                                  # (H, W, 3)
    xyz /= np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)     # D65 white point
    # XYZ → L*a*b*
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), (29 / 6) ** 2 / 3 * xyz + 4 / 29)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1).astype(np.float32)


# ── Normalization / visualization ─────────────────────────────────────────────

def normalize_output(out: Image.Image, ans: Image.Image) -> Image.Image:
    """Resize and centre-crop output to match answer dimensions (NEAREST interpolation)."""
    W_a, H_a = ans.size
    W_o, H_o = out.size
    if (W_o, H_o) == (W_a, H_a):
        return out
    scale   = max(W_a / W_o, H_a / H_o)
    nW      = math.ceil(W_o * scale)
    nH      = math.ceil(H_o * scale)
    resized = out.resize((nW, nH), Image.NEAREST)
    x       = (nW - W_a) // 2
    y       = (nH - H_a) // 2
    return resized.crop((x, y, x + W_a, y + H_a))


def make_eval_image(changed: np.ndarray, cie76: np.ndarray, threshold: float) -> Image.Image:
    """Colour-coded diff image: green=CC, blue=CU, red=IC, orange=IU."""
    correct = cie76 <= threshold
    out     = np.empty((*changed.shape, 3), dtype=np.uint8)
    out[ changed &  correct] = _CC
    out[~changed &  correct] = _CU
    out[ changed & ~correct] = _IC
    out[~changed & ~correct] = _IU
    return Image.fromarray(out)


# ── Per-problem statistics ────────────────────────────────────────────────────

def compute_problem_stats(
    I: np.ndarray,
    A: np.ndarray,
    O: np.ndarray,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """Return pixel counts + per-threshold stats, the CIE76 map, and the changed mask."""
    changed = np.any(I != A, axis=-1)
    edit_pixels         = int(changed.sum())
    preservation_pixels = int((~changed).sum())

    OIA_lab   = _rgb_to_lab(np.stack([O, I, A]))       # (3, H, W, 3) — one pass
    cie76_map       = np.sqrt(((OIA_lab[0] - OIA_lab[2]) ** 2).sum(axis=-1))  # ΔE(O, A)
    cie76_input_map = np.sqrt(((OIA_lab[0] - OIA_lab[1]) ** 2).sum(axis=-1))  # ΔE(O, I)

    cie76_threshold: dict = {}
    for t in THRESHOLDS:
        correct = cie76_map <= t
        edit_correct_pixels         = int((changed  & correct).sum())
        preservation_correct_pixels = int((~changed & correct).sum())
        preservation_incorrect_pixels = preservation_pixels - preservation_correct_pixels
        iou_denom = edit_pixels + preservation_incorrect_pixels
        cie76_threshold[str(t)] = {
            "edit_correct_pixels":         edit_correct_pixels,
            "preservation_correct_pixels": preservation_correct_pixels,
            "edit_accuracy":         edit_correct_pixels         / edit_pixels         if edit_pixels         else 1.0,
            "preservation_accuracy": preservation_correct_pixels / preservation_pixels if preservation_pixels else 1.0,
            "iou":                   edit_correct_pixels / iou_denom                  if iou_denom           else 1.0,
            "changed_pixels":        int((cie76_input_map > t).sum()),
        }

    n = len(THRESHOLDS)
    cie76_mean = {
        f: sum(cie76_threshold[str(t)][f] for t in THRESHOLDS) / n
        for f in ["edit_correct_pixels", "preservation_correct_pixels",
                  "edit_accuracy", "preservation_accuracy", "iou", "changed_pixels"]
    }

    return {"edit_pixels": edit_pixels, "preservation_pixels": preservation_pixels,
            "cie76_threshold": cie76_threshold, "cie76_mean": cie76_mean}, cie76_map, changed


def _zero_stats(edit_pixels: int, preservation_pixels: int) -> dict:
    """Stats dict with zero correct pixels/accuracy for missing model output."""
    iou_zero = 1.0 if edit_pixels == 0 and preservation_pixels == 0 else 0.0
    return {
        "edit_pixels":         edit_pixels,
        "preservation_pixels": preservation_pixels,
        "cie76_threshold": {
            str(t): {
                "edit_correct_pixels":         0,
                "preservation_correct_pixels": 0,
                "edit_accuracy":               0.0,
                "preservation_accuracy":       0.0,
                "iou":                         iou_zero,
                "changed_pixels":              0,
            }
            for t in THRESHOLDS
        },
        "cie76_mean": {
            "edit_correct_pixels":         0,
            "preservation_correct_pixels": 0,
            "edit_accuracy":               0.0,
            "preservation_accuracy":       0.0,
            "iou":                         iou_zero,
            "changed_pixels":              0,
        },
    }


# ── Cache helpers ─────────────────────────────────────────────────────────────
# Per-problem sidecar JSONs at ``<results>/<model>/<benchmark>/<task>/<NNNN>_stats.json``
# mirror the inference cache pattern (see ``src/inference.py``: skipped/cached
# results carry a "skipped": true flag and short-circuit before the worker
# pool dispatch). Cheap mtime check handles "model output regenerated since
# last eval"; validation happens at load time in _load_cached_record.

_SIDECAR_SUFFIX = "_stats.json"


def _sidecar_path(results_task_dir: Path, num4: str) -> Path:
    return results_task_dir / f"{num4}{_SIDECAR_SUFFIX}"


def _newest_mtime(*paths: Path) -> float:
    """Max mtime across present paths; missing paths skipped (treated as
    older than any present file)."""
    best = 0.0
    for p in paths:
        try:
            m = p.stat().st_mtime
        except FileNotFoundError:
            continue
        if m > best:
            best = m
    return best


def _is_cached(
    sidecar_path: Path,
    sources: List[Path],
    save_images_marker: Optional[Path],
) -> bool:
    """Cache hit when sidecar exists, has nonzero size, and is newer
    than every present source file (input/answer/output PNGs + per-
    problem JSON).

    Plan-time check only: matches ``inference.py``'s
    ``_build_global_plan`` cheap-stat pattern. JSON-decode validation
    happens at load time in :func:`_load_cached_record`; truncated /
    corrupt sidecars fall through there and get rerun (with the same
    "(N forecast-cached sidecar(s) was/were corrupt and rerun)" note
    that ``inference.py``'s ``_print_corrupt_cache_note`` emits for
    PNGs).

    ``save_images_marker`` is the canonical diff-PNG that ``--save-images``
    would write; pass ``None`` when ``--save-images`` is off. When passed
    and missing, the cache is treated as a miss so the diff images get
    rendered. (The inverse — sidecar from a prior ``--save-images`` run,
    current run is no-images — is a cache hit: the sidecar suffices.)
    """
    try:
        st = sidecar_path.stat()
    except FileNotFoundError:
        return False
    if st.st_size == 0:
        return False

    src_mtime = _newest_mtime(*sources)
    if src_mtime > st.st_mtime:
        return False

    if save_images_marker is not None and not save_images_marker.exists():
        return False

    return True


def _save_images_marker(results_task_dir: Path, num4: str) -> Path:
    """Canonical diff-PNG used as the ``--save-images`` cache marker.
    Picked as the *last* IMAGE_THRESHOLDS diff so a partial save
    (interrupted between threshold writes) falls through and gets redone."""
    return results_task_dir / f"{num4}_diff_cie76_{IMAGE_THRESHOLDS[-1]}.png"


# ── Processing ────────────────────────────────────────────────────────────────

def _process_one_problem(job: dict) -> dict:
    bench_task_dir   = Path(job["bench_task_dir"])
    results_task_dir = Path(job["results_task_dir"])
    out_task_dir     = Path(job["out_task_dir"]) if job["out_task_dir"] else None
    num_str, idx     = job["num_str"], job["idx"]
    num4             = f"{idx:04d}"

    input_path  = bench_task_dir / f"{num_str}_input.png"
    answer_path = bench_task_dir / f"{num_str}_answer.png"
    out_path    = (out_task_dir / f"{num4}_output.png") if out_task_dir else None

    bg_colors        = None
    mode             = ""
    visual_condition = ""
    meta_path        = bench_task_dir / f"{num_str}.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if "bg_color" in meta:
            bg_colors = [meta["bg_color"]]
        elif "bg_colors" in meta:
            bg_colors = meta["bg_colors"]
        # "default" is the canonical mode-less label (mirrors _NO_MODE_LABEL in
        # generate_benchmark.py); the fallback only fires for pre-default-label
        # sidecars on disk.
        mode             = meta.get("mode") or "default"
        visual_condition = meta.get("visual_condition") or ""

    I_img = Image.open(input_path).convert("RGB")
    A_img = Image.open(answer_path).convert("RGB")
    I, A  = np.array(I_img), np.array(A_img)

    O_raw  = Image.open(out_path).convert("RGB") if (out_path and out_path.exists()) else None
    O_norm = normalize_output(O_raw, A_img)      if O_raw is not None else None
    O      = np.array(O_norm)                    if O_norm is not None else None

    changed = np.any(I != A, axis=-1)
    edit_pixels         = int(changed.sum())
    preservation_pixels = int((~changed).sum())

    has_output          = O is not None
    correct_output_size = (list(O_raw.size) == list(A_img.size)) if has_output else None

    result: dict = {
        "model":               job["model"],
        "benchmark":           job["benchmark"],
        "task":                job["task"],
        "mode":                mode,
        "visual_condition":    visual_condition,
        "idx":                 idx,
        "input_size":          list(I_img.size),
        "bg_colors":           bg_colors,
        "has_output":          has_output,
        "correct_output_size": correct_output_size,
        "output": {"output_size": None, **_zero_stats(edit_pixels, preservation_pixels)},
    }

    if O is not None:
        stats, cie76_map, changed = compute_problem_stats(I, A, O)
        result["output"] = {"output_size": list(O_raw.size), **stats}
        if job["save_images"]:
            # Only the thresholds the visualizer actually displays — keeps
            # eval_outputs/ at ~20 KB / problem instead of ~55 KB.
            for t in IMAGE_THRESHOLDS:
                make_eval_image(changed, cie76_map, t).save(
                    results_task_dir / f"{num4}_diff_cie76_{t}.png"
                )

    if job["save_images"]:
        # Only cache the normalized output when normalize_output had to
        # resize/crop (it returns its `out` argument unchanged when the
        # model output is already at answer dimensions, so this skips a
        # redundant write for ~all problems where the model honours the
        # requested size).
        # NB: input/answer live in --benchmarks and output in --model-outputs,
        # so we don't copy them — the visualizer reads from the source dirs.
        if O_norm is not None and O_norm is not O_raw:
            O_norm.save(results_task_dir / f"{num4}_normalized_output.png")

    # Persist the per-problem record as a sidecar so the next run's cache
    # check (mtime-based) sees a fresh file. Written last so a crash mid-
    # computation doesn't leave a sidecar pointing at half-rendered diff
    # images.
    sidecar = _sidecar_path(results_task_dir, num4)
    sidecar.write_text(json.dumps(result))

    return result


def _load_cached_record(sidecar_path: Path) -> Optional[dict]:
    """Read a sidecar JSON. Returns ``None`` on any failure — the caller
    falls through to the worker pool to recompute."""
    try:
        return json.loads(sidecar_path.read_text())
    except Exception:
        return None


def _collect_jobs(
    benchmarks_root: Path,
    outputs_root: Path,
    results_root: Path,
    model_filter: Optional[str],
    benchmark_filter: Optional[str],
    task_filter: Optional[str],
    save_images: bool,
    overwrite: bool,
) -> List[dict]:
    """Collect all jobs across models/benchmarks/tasks. Annotates each
    with ``is_cached`` (sidecar present + mtime-fresh) so
    the caller can short-circuit cached jobs before dispatching them to
    the worker pool. Always creates the per-task results dir so sidecars
    have a place to land.

    On a full eval grid (12 models × 2 benchmarks × ~20 tasks × ~96
    problems = ~31k jobs) this walks ~62k files and stat()s ~93k more
    for the mtime-based ``_is_cached`` check. That's ~30s wall on warm
    cache, several minutes on cold cache — long enough that the tqdm
    progress bar (per-model granularity) is worth the trivial cost.
    """
    jobs: List[dict] = []

    model_dirs = sorted(d for d in outputs_root.iterdir()
                        if d.is_dir() and (not model_filter or d.name == model_filter))
    for model_dir in tqdm(model_dirs, desc="collecting jobs", leave=False):
        for out_bench_dir in sorted(model_dir.iterdir()):
            if not out_bench_dir.is_dir():
                continue
            bench_name = out_bench_dir.name
            if benchmark_filter and bench_name != benchmark_filter:
                continue

            bench_dir = benchmarks_root / bench_name
            if not bench_dir.exists():
                print(f"Benchmark not found: {bench_dir} — skipping")
                continue

            for task_dir in sorted(bench_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                task_name = task_dir.name
                if task_filter and task_name != task_filter:
                    continue

                out_task_dir     = out_bench_dir / task_name
                results_task_dir = results_root / model_dir.name / bench_name / task_name

                if not out_task_dir.exists():
                    out_task_dir = None
                # Always mkdir: sidecars need a home even when --save-images
                # is off. Cheap (idempotent) and unifies the create logic.
                results_task_dir.mkdir(parents=True, exist_ok=True)

                for p in sorted(task_dir.glob("*_input.png")):
                    num_str = p.stem.replace("_input", "")
                    idx     = int(num_str)
                    num4    = f"{idx:04d}"

                    input_path  = task_dir / f"{num_str}_input.png"
                    answer_path = task_dir / f"{num_str}_answer.png"
                    meta_path   = task_dir / f"{num_str}.json"
                    out_path    = (out_task_dir / f"{num4}_output.png") if out_task_dir else None

                    sources: List[Path] = [input_path, answer_path, meta_path]
                    if out_path is not None:
                        sources.append(out_path)

                    # Only require the diff-PNG marker for problems that
                    # actually have a model output PNG. Problems with no
                    # output skip the diff-PNG write loop in
                    # ``_process_one_problem`` (guarded by ``if O is not None``),
                    # so the marker would never exist for them — gating on
                    # it would force every ``make eval`` to recompute every
                    # missing-output problem, even when nothing changed.
                    out_exists = out_path is not None and out_path.exists()
                    needs_marker = save_images and out_exists
                    is_cached = (
                        not overwrite
                        and _is_cached(
                            _sidecar_path(results_task_dir, num4),
                            sources,
                            _save_images_marker(results_task_dir, num4) if needs_marker else None,
                        )
                    )

                    jobs.append({
                        "bench_task_dir":   str(task_dir),
                        "results_task_dir": str(results_task_dir),
                        "out_task_dir":     str(out_task_dir) if out_task_dir else None,
                        "num_str":          num_str,
                        "idx":              idx,
                        "model":            model_dir.name,
                        "benchmark":        bench_name,
                        "task":             task_name,
                        "save_images":      save_images,
                        "is_cached":        is_cached,
                    })

    return jobs


def _print_plan(jobs: List[dict], workers: int) -> None:
    """Print a plan summary with cached vs to-evaluate counts, mirroring
    ``src/inference.py``'s ``=== Plan ===`` block. Quick visual confirmation
    that the cache is working before any wall-time is spent."""
    total = len(jobs)
    cached = sum(1 for j in jobs if j["is_cached"])
    fresh  = total - cached

    # Per-(model, benchmark) breakdown, sorted by the same key as the jobs.
    by_model_bench: dict = defaultdict(lambda: [0, 0])  # (n_total, n_cached)
    for j in jobs:
        key = (j["model"], j["benchmark"])
        by_model_bench[key][0] += 1
        if j["is_cached"]:
            by_model_bench[key][1] += 1

    print()
    print("=== Plan ===")
    name_width = max(
        (len(f"{m}/{b}") for (m, b) in by_model_bench),
        default=0,
    )
    for (model, bench), (n, n_cached) in sorted(by_model_bench.items()):
        label = f"{model}/{bench}".ljust(name_width)
        if n_cached == 0:
            note = f"{n} problems"
        elif n_cached == n:
            note = f"{n} problems — all cached"
        else:
            note = f"{n} problems, {n_cached} cached, evaluating {n - n_cached}"
        print(f"  {label}  ({note})")

    print()
    if fresh > 0:
        print(f"  {fresh} problems to evaluate (cached: {cached}/{total})")
    else:
        print(f"  All {total} problems cached — nothing to recompute.")
    print()
    parallelism_note = f"{workers} workers" if workers > 1 else "serial"
    print(f"=== Running × {parallelism_note} ===")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate model outputs with pointwise CIE76 colour distance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmarks",    default="benchmarks")
    parser.add_argument("--model-outputs", default="model_outputs")
    parser.add_argument("--eval-outputs",  default="eval_outputs")
    parser.add_argument("--model",     help="Process only this model")
    parser.add_argument("--benchmark", help="Process only this benchmark")
    parser.add_argument("--task",      help="Process only this task")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--save-images", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Cache per-problem diagnostic PNGs (4 ΔE diff "
                             "PNGs at thresholds 0/2/5/10, plus the "
                             "normalized output when it differs from raw) "
                             "alongside the sidecar. The visualizer reads "
                             "these for the Eval tab. Pass --no-save-images "
                             "for a stats-only fast eval (`make eval-quick`).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute every problem, overwriting cached "
                             "<NNNN>_stats.json sidecars. By default eval is "
                             "incremental: a sidecar with mtime ≥ all source "
                             "PNGs (input/answer/output) is reused, so reruns "
                             "after a cancel / partial inference only redo the "
                             "problems whose model output changed. Pass "
                             "--overwrite when you change eval logic (CIE76 "
                             "math, threshold list, normalization, ...) and "
                             "want to invalidate the cache without manually "
                             "deleting --eval-outputs.")
    args = parser.parse_args()

    results_root = Path(args.eval_outputs)
    results_root.mkdir(parents=True, exist_ok=True)

    jobs = _collect_jobs(
        benchmarks_root  = Path(args.benchmarks),
        outputs_root     = Path(args.model_outputs),
        results_root     = results_root,
        model_filter     = args.model,
        benchmark_filter = args.benchmark,
        task_filter      = args.task,
        save_images      = args.save_images,
        overwrite        = args.overwrite,
    )
    print(f"Found {len(jobs)} problems")

    _print_plan(jobs, args.workers)

    cached_jobs = [j for j in jobs if j["is_cached"]]
    fresh_jobs  = [j for j in jobs if not j["is_cached"]]

    # Cached records: just load the sidecar. Single-threaded — this is
    # fast enough that the worker-pool overhead (process spawn + IPC for
    # each load) would dominate. tqdm bar because at full grid this is
    # ~31k JSON reads and ~1-3min wall, long enough that silent loading
    # looks like a hang otherwise.
    cached_records: List[dict] = []
    n_corrupt_cache = 0
    cached_iter = tqdm(cached_jobs, desc="loading cached", unit="problems",
                       leave=False) if cached_jobs else cached_jobs
    for j in cached_iter:
        sidecar = _sidecar_path(Path(j["results_task_dir"]), f"{j['idx']:04d}")
        rec = _load_cached_record(sidecar)
        if rec is None:
            # Sidecar passed the _is_cached check earlier but failed to
            # load now (race / fs flakiness). Promote to fresh so the
            # worker pool recomputes it.
            n_corrupt_cache += 1
            j["is_cached"] = False
            fresh_jobs.append(j)
        else:
            cached_records.append(rec)
    if n_corrupt_cache:
        plural = "" if n_corrupt_cache == 1 else "s"
        was = "was" if n_corrupt_cache == 1 else "were"
        print(f"  ({n_corrupt_cache} forecast-cached sidecar{plural} {was} corrupt and rerun)")

    if fresh_jobs:
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                fresh_records = list(tqdm(
                    ex.map(_process_one_problem, fresh_jobs),
                    total=len(fresh_jobs),
                    desc="evaluating", unit="problems", leave=False,
                ))
        else:
            fresh_records = [_process_one_problem(j) for j in tqdm(
                fresh_jobs, desc="evaluating", unit="problems", leave=False)]
    else:
        fresh_records = []

    # Final write: stable sort by (model, benchmark, task, mode, visual_condition, idx)
    # so the jsonl ordering doesn't churn between cached-only and mixed
    # runs (otherwise diffs of problem_stats.jsonl across cache-rebuild
    # vs cache-hit reruns are noisy).
    records = cached_records + fresh_records
    records.sort(key=lambda r: (r["model"], r["benchmark"], r["task"], r.get("mode", ""), r.get("visual_condition", ""), r["idx"]))

    jsonl_path = results_root / "problem_stats.jsonl"
    with open(jsonl_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    n_cached = len(cached_records)
    n_fresh  = len(fresh_records)
    print(
        f"Wrote {len(records)} records → {jsonl_path}  "
        f"(cached: {n_cached}, recomputed: {n_fresh})"
    )


if __name__ == "__main__":
    main()
