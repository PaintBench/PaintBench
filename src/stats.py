"""Compute aggregate stats from problem_stats.jsonl.

Granularity rules per benchmark:

  PaintBench
    - Single-mode tasks  → level="task" only
    - Multi-mode tasks   → level="mode" per mode + level="task"
    - level="category"   → equal task weight within each of 4 categories
    - level="visual_condition"  → equal task weight per visual condition (8 conditions)
    - level="benchmark"  → equal task weight across all 20 tasks

  TinyGrafixBench
    - level="task"       → one row per chart subtask (no mode aggregation)
    - level="category"   → equal subtask weight per chart type (5 chart types)
    - level="benchmark"  → equal category weight

Unless --no-ci is passed, all cie76_t and cie76_mean entries include a
95% percentile-bootstrap CI for mean IoU, mean edit accuracy, and mean
preservation accuracy. The bootstrap unit matches the aggregation level
so the CI answers the same inferential question as the displayed point
estimate:

  PaintBench (4 categories × 5 tasks)
    - level="task" / "mode"               → resample problems
                                            (per-problem bootstrap)
    - level="category"                    → 2-level hierarchical:
                                            resample problems within each task,
                                            macro across the 5 tasks
    - level="visual_condition"            → 2-level hierarchical over the (up to 20)
                                            tasks contributing to that condition
    - level="benchmark"                    → 3-level hierarchical:
                                            resample problems within each task,
                                            macro across the 5 tasks per category,
                                            macro across the 4 categories (matches
                                            the doubly-macro point estimate)

  TinyGrafixBench (5 charts × 4 subtasks)
    - level="task"                         → resample problems
    - level="category"                    → 2-level hierarchical over the 4 subtasks
                                            within a chart
    - level="benchmark"                    → 3-level hierarchical:
                                            resample problems within each subtask,
                                            macro across the 4 subtasks per chart,
                                            macro across the 5 charts (matches the
                                            doubly-macro point estimate)

PB and TGB share the same bench-level estimator shape (doubly-macro:
categories × inner-units). With uniform inner-unit count per category —
production: 5 tasks/cat for PB, 4 subtasks/chart for TGB — the
3-level bootstrap reduces numerically to a flat task-macro, but the
3-level form is required for correctness under unequal inner-unit
counts (e.g. task drops) and matches the design intent that categories
are the equal-weight unit at bench level. See ``_bootstrap_ci_tasks``
(2-level), ``_bootstrap_ci_subtasks_then_tasks`` (3-level), and the
``test_*_macro_ci_*`` / ``test_*_bench_ci_*`` regression guards in
``tests/test_stats.py``.

Per-task counts are reported alongside per-problem counts on aggregated
rows (``n_tasks`` next to ``n_problems``) so readers can tell the bootstrap
sample size apart from total problem volume.

Usage:
    python src/stats.py \\
        --input  eval_outputs/problem_stats.jsonl \\
        --output eval_outputs/aggregate_stats.jsonl

    # Fast iteration without CIs (~3s vs ~6min with B=10000):
    python src/stats.py --no-ci ...

    # Per-model parallelism for the bootstrap CI step defaults to
    # os.cpu_count() (~30s on a 12-core box vs ~6min serial). Override via
    # --workers N, or via the Makefile: `make stats JOBS=N` (matches
    # `make generate` / `make eval` JOBS=N convention). Pass --workers 1
    # to disable parallelism (e.g. in a sandbox that blocks multiprocessing).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import multiprocessing as mp
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Single source of truth: mirrors the definition in generate_benchmark.py.
# Keep these two in sync if categories ever change.
from generate_benchmark import TASK_CATEGORIES

THRESHOLDS = list(range(11))   # CIE76 = 0, 1, 2, ..., 10

TASK_NAMES: list[str] = sorted([
    "translation", "rotation", "reflection", "scaling", "shearing",
    "construction", "removal", "copying", "border", "cropping",
    "recolor", "flood_fill", "blending", "gradient", "point_operations",
    "comparison", "ordering", "pattern", "counting", "legend",
], key=len, reverse=True)

TGF_TASK_NAMES: list[str] = sorted([
    "bar_chart", "heatmap", "line_chart", "network", "scatter_plot",
], key=len, reverse=True)

_MEAN_FIELDS = ["mean_edit_accuracy", "mean_preservation_accuracy", "mean_iou", "mean_changed_pixels"]
_STD_PAIRS   = [("mean_edit_accuracy",         "std_edit_accuracy"),
                ("mean_preservation_accuracy", "std_preservation_accuracy"),
                ("mean_iou",                   "std_iou"),
                ("mean_changed_pixels",        "std_changed_pixels")]


def task_of(task_mode: str, task_names: list[str] = TASK_NAMES) -> str:
    for t in task_names:
        if task_mode == t or task_mode.startswith(t + "_"):
            return t
    return task_mode


# ── Stats helpers ──────────────────────────────────────────────────────────────

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


# ── Bootstrap CIs ──────────────────────────────────────────────────────────────
#
# RNG seed sharing: every helper below builds its own `np.random.default_rng(seed)`
# from the same seed passed in by the caller. So within a single stats.py run
# the bootstrap streams across calls (different thresholds, different levels,
# different metrics within a call) are *correlated* — they replay the same
# index sequences whenever the array shape coincides. This does NOT bias any
# individual CI (each is a valid percentile bootstrap of its own input), but
# it does mean cross-threshold or cross-level CI comparisons aren't strictly
# independent. For the intended use (point estimates with CIs on a single
# artifact, displayed independently) this is fine. If decorrelated streams
# ever matter, derive a SeedSequence once in main() and spawn child seeds
# per call site.

def _percentiles_2_5_97_5(boot_means: np.ndarray) -> np.ndarray:
    """Apply (2.5, 97.5) percentiles along axis 0.

    boot_means may be 1-D (single metric) → returns shape (2,) of [lo, hi];
    or 2-D (n_boot, n_metrics) → returns shape (2, n_metrics) of [[los], [his]].
    """
    return np.percentile(boot_means, [2.5, 97.5], axis=0)


def _bootstrap_ci_problems(per_problem_values: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Percentile bootstrap 95% CI for the mean(s) of per-problem values.

    Resamples problems with replacement. Use for task / mode level CIs,
    where the only sampling unit is the per-problem score.

    Args:
      per_problem_values : shape (n_problems,) for a single metric, or
                           (n_problems, n_metrics) to bootstrap multiple
                           metrics jointly (same resampling, different
                           per-problem values — much cheaper than calling
                           this once per metric).
      n_boot             : number of bootstrap iterations.
      seed               : RNG seed.

    Returns:
      shape (2,) of [lo, hi]              if input was 1-D
      shape (2, n_metrics) of [los, his]  if input was 2-D

    Fully vectorized: one (n_boot, n_problems) index gather, then a
    single `mean(axis=1)` and one `percentile` call. Constant in n_boot
    and n_metrics (no Python loops over either).
    """
    arr = np.asarray(per_problem_values, dtype=float)
    if arr.size == 0 or n_boot <= 0:
        return np.zeros((2,) if arr.ndim == 1 else (2, arr.shape[1]))
    n   = arr.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    # arr[idx]: (n_boot, n_problems[, n_metrics]) → mean over axis=1 →
    #          (n_boot,) for 1-D input  /  (n_boot, n_metrics) for 2-D
    boot_means = arr[idx].mean(axis=1)
    return _percentiles_2_5_97_5(boot_means)


def _bootstrap_ci_tasks(per_task_values: list[np.ndarray], n_boot: int, seed: int) -> np.ndarray:
    """Hierarchical percentile bootstrap 95% CI for the macro-mean over tasks.

    Use for 2-level macro estimators: PaintBench category /
    visual_condition, and TGF category. PB and TGB benchmark rows use the
    3-level form (``_bootstrap_ci_subtasks_then_tasks``) since their bench
    estimators are doubly-macro through categories.

    Each bootstrap iteration:
      1. Resample problems within each task with replacement (preserves
         per-task n and within-task structure).
      2. Compute each task's resampled mean.
      3. Take the macro-mean across tasks.

    Treats the set of tasks as FIXED — the benchmark IS the 20 tasks, not
    a sample from a population. The CI reflects "if I regenerated different
    random problems within the same tasks, where would the macro-mean land?"
    See module docstring and ``tests/test_stats.py::test_*_macro_ci_*`` for
    the regression guards.

    Args:
      per_task_values : list of per-task arrays, each shape (n_problems_t,)
                        for a single metric or (n_problems_t, n_metrics) to
                        bootstrap multiple metrics jointly. Tasks with
                        n_problems_t == 0 are dropped.
      n_boot, seed    : bootstrap params.

    Returns:
      shape (2,) of [lo, hi]              if per-task arrays were 1-D
      shape (2, n_metrics) of [los, his]  if per-task arrays were 2-D

    Note on small k: the percentile bootstrap is lumpy when k is small.
    PaintBench categories (k=5 tasks) and TGF chart categories (k=4
    subtasks) are in this regime — the 2.5/97.5 percentiles often just
    bracket the extreme task means rather than producing a smoothly
    interpolated interval. Read the resulting CI as a *range* rather
    than a precise interval at category level. Benchmark level uses the
    3-level form (this helper isn't called there) — k=4 categories for
    PB, k=5 charts for TGB at the outer macro step, with the inner step
    smoothing within-category variance into per-category means. The
    point estimate is unaffected by k.

    Vectorized: for each task we build one (n_boot, n_t) index matrix and
    one gather+mean call; the outer task loop is len(tasks) ≪ n_boot
    iterations of Python. ~100× faster than a per-iter Python loop.
    """
    arrs = [np.asarray(v, dtype=float) for v in per_task_values if len(v) > 0]
    if not arrs or n_boot <= 0:
        # Need to infer output shape from any non-empty array, else 1-D.
        sample = next((np.asarray(v) for v in per_task_values if len(v) > 0), None)
        if sample is None or sample.ndim == 1:
            return np.zeros((2,))
        return np.zeros((2, sample.shape[1]))

    rng = np.random.default_rng(seed)
    # boot_task_means: per-task bootstrap means → stack into one array.
    boot_task_means = []
    for arr in arrs:
        n   = arr.shape[0]
        idx = rng.integers(0, n, size=(n_boot, n))
        boot_task_means.append(arr[idx].mean(axis=1))   # (n_boot,) or (n_boot, n_metrics)
    # Stack along a new task axis, then mean over it → (n_boot,) or (n_boot, n_metrics)
    boot_macro = np.stack(boot_task_means, axis=0).mean(axis=0)
    return _percentiles_2_5_97_5(boot_macro)


def _bootstrap_ci_subtasks_then_tasks(
    per_task_subtask_values: list[list[np.ndarray]], n_boot: int, seed: int,
) -> np.ndarray:
    """Three-level hierarchical percentile bootstrap 95% CI for a doubly-macro
    mean: macro over tasks of (macro over subtasks of per-subtask means).

    Used by both PaintBench's and TinyGrafixBench's BENCHMARK-level rows,
    whose point estimates are doubly-macro by design:
      - PB:   ``mean_over_categories(mean_over_tasks(per_task_mean))``
              (4 categories × 5 tasks each)
      - TGB:  ``mean_over_charts(mean_over_subtasks(per_subtask_mean))``
              (5 charts × 4 subtasks each)
    In either case the outer unit is a "category" (chart for TGB) and the
    inner unit is a "task" (subtask for TGB) contributing equally to its
    parent. The matching CI sampling unit is *per-inner-unit problem
    resampling* (not the category-pooled problem list), so the CI is
    centered on the same doubly-macro mean as the displayed point
    estimate. Pooling problems across inner units within a category
    would silently regress to a problem-weighted per-category mean — the
    same pooled-vs-macro mismatch ``_bootstrap_ci_tasks`` exists to avoid
    at one level shallower.

    Each bootstrap iteration:
      1. Resample problems within each inner unit (task/subtask) with replacement.
      2. Compute each inner unit's resampled mean.
      3. Macro-average across inner units within each outer unit (category/chart)
         → per-category mean.
      4. Macro-average across outer units → bench mean.

    Args:
      per_task_subtask_values : list[list[arr]]. Outer list: categories
                                (charts for TGB, categories for PB). Inner
                                list: inner units within that category
                                (tasks for PB, subtasks for TGB).
                                Each arr: shape (n_problems_ts,) or
                                (n_problems_ts, n_metrics).
                                The parameter name retains TGB-flavoured
                                naming for historical reasons; conceptually
                                it's outer-then-inner per-problem lists.
      n_boot, seed            : bootstrap params.

    Returns:
      shape (2,) of [lo, hi]              if inner arrs were 1-D
      shape (2, n_metrics) of [los, his]  if inner arrs were 2-D

    See ``tests/test_stats.py::test_tgf_benchmark_ci_uses_subtask_hierarchical*``
    for the discriminating regression guard.
    """
    # Filter out empty subtasks; drop tasks that have no remaining subtasks.
    cleaned: list[list[np.ndarray]] = []
    sample = None
    for subtask_arrs in per_task_subtask_values:
        kept = [np.asarray(v, dtype=float) for v in subtask_arrs if len(v) > 0]
        if kept:
            cleaned.append(kept)
            if sample is None:
                sample = kept[0]
    if not cleaned or n_boot <= 0:
        if sample is None or sample.ndim == 1:
            return np.zeros((2,))
        return np.zeros((2, sample.shape[1]))

    rng = np.random.default_rng(seed)
    boot_task_means = []
    for subtask_arrs in cleaned:
        boot_subtask_means = []
        for arr in subtask_arrs:
            n   = arr.shape[0]
            idx = rng.integers(0, n, size=(n_boot, n))
            boot_subtask_means.append(arr[idx].mean(axis=1))   # (n_boot,) or (n_boot, n_metrics)
        # Macro across subtasks within this task → (n_boot,) or (n_boot, n_metrics)
        boot_task_means.append(np.stack(boot_subtask_means, axis=0).mean(axis=0))
    # Macro across tasks → (n_boot,) or (n_boot, n_metrics)
    boot_bench = np.stack(boot_task_means, axis=0).mean(axis=0)
    return _percentiles_2_5_97_5(boot_bench)


_CI_METRIC_KEYS    = ["edit_accuracy", "preservation_accuracy", "iou"]
_CI_FIELD_PREFIXES = ["ci95_edit_accuracy", "ci95_preservation_accuracy", "ci95_iou"]


def _extract_per_problem_matrix(problems: list[dict], block: str, threshold_key: str | None) -> np.ndarray:
    """Build a (n_problems, 3) array of per-problem (edit_acc, pres_acc, iou)
    so all three metrics can be bootstrapped jointly in a single resampling.
    """
    if block == "cie76_threshold":
        rows = [
            [p["output"][block][threshold_key][k] for k in _CI_METRIC_KEYS]
            for p in problems
        ]
    else:
        rows = [
            [p["output"][block][k] for k in _CI_METRIC_KEYS]
            for p in problems
        ]
    return np.asarray(rows, dtype=float)


def _add_problem_cis(entry: dict, problems: list[dict], block: str, threshold_key: str | None,
                     n_boot: int, seed: int) -> None:
    """In-place: add ci95_*_low/high fields to entry by per-problem bootstrap.

    Vectorizes across the 3 metrics: one resampling, three percentile
    outputs.
    """
    if n_boot <= 0 or not problems:
        return
    matrix = _extract_per_problem_matrix(problems, block, threshold_key)   # (n_p, 3)
    los_his = _bootstrap_ci_problems(matrix, n_boot, seed)                 # (2, 3)
    for i, prefix in enumerate(_CI_FIELD_PREFIXES):
        entry[f"{prefix}_low"]  = float(los_his[0, i])
        entry[f"{prefix}_high"] = float(los_his[1, i])


def _add_task_cis(entry: dict, per_task_problems: list[list[dict]] | None,
                  block: str, threshold_key: str | None,
                  n_boot: int, seed: int) -> None:
    """In-place: add ci95_*_low/high fields to entry by 2-level hierarchical
    task-bootstrap.

    Vectorizes across the 3 metrics: per-task arrays carry all 3 columns,
    one resampling per task, three percentile outputs.
    """
    if n_boot <= 0 or not per_task_problems:
        return
    per_task_matrices = [
        _extract_per_problem_matrix(probs, block, threshold_key)
        for probs in per_task_problems if probs
    ]
    if not per_task_matrices:
        return
    los_his = _bootstrap_ci_tasks(per_task_matrices, n_boot, seed)          # (2, 3)
    for i, prefix in enumerate(_CI_FIELD_PREFIXES):
        entry[f"{prefix}_low"]  = float(los_his[0, i])
        entry[f"{prefix}_high"] = float(los_his[1, i])


def _add_subtask_task_cis(entry: dict,
                          per_task_subtask_problems: list[list[list[dict]]] | None,
                          block: str, threshold_key: str | None,
                          n_boot: int, seed: int) -> None:
    """In-place: add ci95_*_low/high fields to entry by 3-level hierarchical
    bootstrap (inner-unit resample → per-category macro → bench macro).

    Used by both PaintBench's and TinyGrafixBench's benchmark rows, whose
    displayed point estimates are doubly-macro (inner-unit-macro per
    category, category-macro across categories). PB's inner unit is
    "task" within each of 4 categories; TGB's is "subtask" within each of
    5 charts. See ``_bootstrap_ci_subtasks_then_tasks`` for the math.
    """
    if n_boot <= 0 or not per_task_subtask_problems:
        return
    per_task_subtask_matrices = [
        [_extract_per_problem_matrix(probs, block, threshold_key)
         for probs in subtask_problem_lists if probs]
        for subtask_problem_lists in per_task_subtask_problems
        if subtask_problem_lists
    ]
    per_task_subtask_matrices = [task for task in per_task_subtask_matrices if task]
    if not per_task_subtask_matrices:
        return
    los_his = _bootstrap_ci_subtasks_then_tasks(per_task_subtask_matrices, n_boot, seed)
    for i, prefix in enumerate(_CI_FIELD_PREFIXES):
        entry[f"{prefix}_low"]  = float(los_his[0, i])
        entry[f"{prefix}_high"] = float(los_his[1, i])


# ── Aggregation ────────────────────────────────────────────────────────────────

def compute_cie76_stats(problems: list[dict], n_boot: int = 0, seed: int = 0) -> dict:
    """Within-task per-problem (micro) cie76 stats.

    If n_boot > 0, also writes ci95_<metric>_low/high fields per threshold +
    cie76_mean using a per-problem percentile bootstrap.
    """
    result = {}
    for t in THRESHOLDS:
        key = str(t)
        edit_accs      = [p["output"]["cie76_threshold"][key]["edit_accuracy"]         for p in problems]
        pres_accs      = [p["output"]["cie76_threshold"][key]["preservation_accuracy"] for p in problems]
        ious           = [p["output"]["cie76_threshold"][key]["iou"]                   for p in problems]
        changed_pixels = [p["output"]["cie76_threshold"][key].get("changed_pixels", 0) for p in problems]
        entry = {
            "mean_edit_accuracy":         _mean(edit_accs),
            "std_edit_accuracy":          _std(edit_accs),
            "mean_preservation_accuracy": _mean(pres_accs),
            "std_preservation_accuracy":  _std(pres_accs),
            "mean_iou":                   _mean(ious),
            "std_iou":                    _std(ious),
            "mean_changed_pixels":        _mean(changed_pixels),
            "std_changed_pixels":         _std(changed_pixels),
        }
        _add_problem_cis(entry, problems, "cie76_threshold", key, n_boot, seed)
        result[f"cie76_{t}"] = entry

    # cie76_mean: aggregate the per-problem mean already computed in eval.py
    edit_accs      = [p["output"]["cie76_mean"]["edit_accuracy"]                  for p in problems]
    pres_accs      = [p["output"]["cie76_mean"]["preservation_accuracy"]          for p in problems]
    ious           = [p["output"]["cie76_mean"]["iou"]                            for p in problems]
    changed_pixels = [p["output"]["cie76_mean"].get("changed_pixels", 0)          for p in problems]
    entry = {
        "mean_edit_accuracy":         _mean(edit_accs),
        "std_edit_accuracy":          _std(edit_accs),
        "mean_preservation_accuracy": _mean(pres_accs),
        "std_preservation_accuracy":  _std(pres_accs),
        "mean_iou":                   _mean(ious),
        "std_iou":                    _std(ious),
        "mean_changed_pixels":        _mean(changed_pixels),
        "std_changed_pixels":         _std(changed_pixels),
    }
    _add_problem_cis(entry, problems, "cie76_mean", None, n_boot, seed)
    result["cie76_mean"] = entry
    return result


def compute_cie76_stats_macro(task_cie76_list: list[dict],
                              task_problem_lists: list[list[dict]] | None = None,
                              task_subtask_problem_lists: list[list[list[dict]]] | None = None,
                              n_boot: int = 0, seed: int = 0) -> dict:
    """Equal-task-weighted (macro-average) mean/std (+ optional hierarchical CI).

    Counterpart to ``compute_cie76_stats`` (within-task per-problem stats —
    the micro-average within a task). At aggregate levels (category /
    visual_condition / benchmark) the displayed mean is a macro-average
    across tasks; this helper computes that mean and (when n_boot > 0)
    the matching hierarchical bootstrap CI.

    CI dispatching (exactly one bootstrap path runs per call):

    - ``task_problem_lists`` set → **2-level hierarchical**: resample
      problems within each task, macro across tasks. Use for any row whose
      point estimate is a single macro over per-task means: PaintBench
      category / visual_condition; TGF category. Wraps
      ``_bootstrap_ci_tasks``.
    - ``task_subtask_problem_lists`` set → **3-level hierarchical**:
      resample problems within each inner unit, macro across inner units
      per category, macro across categories. Use for rows whose point
      estimate is a *doubly* macro mean — PaintBench benchmark (estimator
      ``mean_over_categories(mean_over_tasks(per_task_mean))``) and
      TGF benchmark (estimator
      ``mean_over_charts(mean_over_subtasks(per_subtask_mean))``).
      Wraps ``_bootstrap_ci_subtasks_then_tasks``.

    Passing neither (or n_boot=0) skips the CI step. Passing both is a
    caller bug — only the 3-level path runs in that case.

    Args:
      task_cie76_list            : pre-aggregated per-task cie76 dicts
                                   (point estimates; always used).
      task_problem_lists         : 2-level bootstrap input.
      task_subtask_problem_lists : 3-level bootstrap input (PB & TGB bench).
      n_boot, seed               : bootstrap params.
    """
    result = {}
    for t in THRESHOLDS:
        key = f"cie76_{t}"
        entry = _aggregate_block(task_cie76_list, key)
        if task_subtask_problem_lists is not None:
            _add_subtask_task_cis(entry, task_subtask_problem_lists,
                                  "cie76_threshold", str(t), n_boot, seed)
        else:
            _add_task_cis(entry, task_problem_lists,
                          "cie76_threshold", str(t), n_boot, seed)
        result[key] = entry
    entry = _aggregate_block(task_cie76_list, "cie76_mean")
    if task_subtask_problem_lists is not None:
        _add_subtask_task_cis(entry, task_subtask_problem_lists,
                              "cie76_mean", None, n_boot, seed)
    else:
        _add_task_cis(entry, task_problem_lists,
                      "cie76_mean", None, n_boot, seed)
    result["cie76_mean"] = entry
    return result


def _aggregate_block(task_cie76_list: list[dict], key: str) -> dict:
    """One cie76_t (or cie76_mean) block, task-macro mean + std."""
    entry: dict = {}
    for field in _MEAN_FIELDS:
        entry[field] = _mean([s[key][field] for s in task_cie76_list])
    for src, dst in _STD_PAIRS:
        entry[dst] = _std([s[key][src] for s in task_cie76_list])
    return entry


def base_counts(problems: list[dict]) -> dict:
    return {
        "n_problems":            len(problems),
        "n_with_output":         sum(1 for p in problems if p.get("has_output", True)),
        "n_correct_output_size": sum(1 for p in problems if p.get("correct_output_size") is True),
    }


def mode_row(model, benchmark, task, mode_label, problems, n_boot=0, seed=0) -> dict:
    return {"level": "mode", "model": model, "benchmark": benchmark,
            "task": task, "mode": mode_label,
            **base_counts(problems), **compute_cie76_stats(problems, n_boot, seed)}


def task_row(model, benchmark, task, problems, n_boot=0, seed=0) -> dict:
    return {"level": "task", "model": model, "benchmark": benchmark,
            "task": task,
            **base_counts(problems), **compute_cie76_stats(problems, n_boot, seed)}


def tgf_task_row(model, category, task_label, problems, n_boot=0, seed=0) -> dict:
    """TGF leaf row: one chart subtask (level='task')."""
    return {"level": "task", "model": model, "benchmark": "TinyGrafixBench",
            "category": category, "task": task_label,
            **base_counts(problems), **compute_cie76_stats(problems, n_boot, seed)}


def category_row(model, benchmark, category, all_problems, task_cie76_list,
                 task_problem_lists=None, n_boot=0, seed=0) -> dict:
    return {"level": "category", "model": model, "benchmark": benchmark,
            "category": category,
            "n_tasks": len(task_cie76_list),
            **base_counts(all_problems),
            **compute_cie76_stats_macro(task_cie76_list,
                                        task_problem_lists=task_problem_lists,
                                        n_boot=n_boot, seed=seed)}


def visual_condition_row(model, benchmark, visual_condition, all_problems, task_cie76_list,
                         task_problem_lists=None, n_boot=0, seed=0) -> dict:
    return {"level": "visual_condition", "model": model, "benchmark": benchmark,
            "visual_condition": visual_condition,
            "n_tasks": len(task_cie76_list),
            **base_counts(all_problems),
            **compute_cie76_stats_macro(task_cie76_list,
                                        task_problem_lists=task_problem_lists,
                                        n_boot=n_boot, seed=seed)}


def bench_row(model, benchmark, all_problems, task_cie76_list,
              task_problem_lists=None, task_subtask_problem_lists=None,
              n_boot=0, seed=0) -> dict:
    return {"level": "benchmark", "model": model, "benchmark": benchmark,
            "n_tasks": len(task_cie76_list),
            **base_counts(all_problems),
            **compute_cie76_stats_macro(task_cie76_list,
                                        task_problem_lists=task_problem_lists,
                                        task_subtask_problem_lists=task_subtask_problem_lists,
                                        n_boot=n_boot, seed=seed)}


# ── Per-benchmark processors ───────────────────────────────────────────────────

def _process_paintbench(
    model: str,
    task_mode_map: dict[tuple[str, str], list[dict]],
    n_boot: int = 0,
    seed: int = 0,
) -> list[dict]:
    """Emit mode/task/category/visual_condition/benchmark rows for PaintBench."""
    benchmark = "PaintBench"

    # Group (task, mode) keys by task; for PaintBench key[0] is the task name.
    task_to_keys: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in sorted(task_mode_map):
        task_to_keys[key[0]].append(key)

    rows: list[dict] = []
    all_problems: list[dict] = []
    task_vcond_problems: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    cat_task_cie76:        dict[str, list[dict]]       = defaultdict(list)
    cat_task_problems:     dict[str, list[list[dict]]] = defaultdict(list)
    cat_problems:          dict[str, list[dict]]       = defaultdict(list)

    for task, keys in tqdm(sorted(task_to_keys.items()), desc=f"  {model} PaintBench",
                           leave=False, position=_WORKER_POSITION + 1):
        task_problems = [p for key in keys for p in task_mode_map[key]]
        is_main = task in TASK_CATEGORIES

        # Compute once; reuse for the task row and (if is_main) roll-ups.
        cie76 = compute_cie76_stats(task_problems, n_boot, seed)

        if is_main:
            all_problems.extend(task_problems)
            for p in task_problems:
                task_vcond_problems[task][p.get("visual_condition") or ""].append(p)
            cat = TASK_CATEGORIES[task]
            cat_task_cie76[cat].append(cie76)
            cat_task_problems[cat].append(task_problems)
            cat_problems[cat].extend(task_problems)

        if len(keys) > 1:
            for (task_str, mode_str) in keys:
                rows.append(mode_row(model, benchmark, task, mode_str,
                                     task_mode_map[(task_str, mode_str)],
                                     n_boot=n_boot, seed=seed))

        rows.append({"level": "task", "model": model, "benchmark": benchmark, "task": task,
                     "category": TASK_CATEGORIES.get(task),   # None for diagnostic tasks
                     **base_counts(task_problems), **cie76})

    # Category rows (4 categories, equal task weight within each). Also
    # accumulates per-category macro cie76 dicts + per-category task-problem
    # lists for the bench row's 3-level hierarchical bootstrap below
    # (matches the doubly-macro point estimate, symmetric with TGB).
    bench_cat_cie76:    list[dict]             = []
    bench_cat_problems: list[list[list[dict]]] = []
    for cat in sorted(cat_task_cie76):
        cat_macro = compute_cie76_stats_macro(
            cat_task_cie76[cat],
            task_problem_lists=cat_task_problems[cat],
            n_boot=n_boot, seed=seed,
        )
        rows.append({"level": "category", "model": model, "benchmark": benchmark,
                     "category": cat,
                     "n_tasks": len(cat_task_cie76[cat]),
                     **base_counts(cat_problems[cat]),
                     **cat_macro})
        bench_cat_cie76.append(cat_macro)
        bench_cat_problems.append(cat_task_problems[cat])

    # Visual-condition rows (8 visual conditions, equal task weight)
    all_vconditions = sorted({c for tc in task_vcond_problems.values() for c in tc if c})
    for vcond in all_vconditions:
        vcond_task_cie76:    list[dict]       = []
        vcond_task_problems: list[list[dict]] = []
        vcond_probs:         list[dict]       = []
        for task in sorted(task_vcond_problems):
            probs = task_vcond_problems[task].get(vcond, [])
            if probs:
                vcond_task_cie76.append(compute_cie76_stats(probs, n_boot, seed))
                vcond_task_problems.append(probs)
                vcond_probs.extend(probs)
        rows.append(visual_condition_row(model, benchmark, vcond, vcond_probs,
                                         vcond_task_cie76,
                                         task_problem_lists=vcond_task_problems,
                                         n_boot=n_boot, seed=seed))

    # Benchmark row: equal category weight, where each per-category mean is
    # itself an equal-task-weight macro. The point estimate is
    # `mean_over_categories(mean_over_tasks(per-task mean))` — a *doubly*
    # macro mean, symmetric with the TGB benchmark row. With uniform task
    # count per category (production: 5 tasks/category × 4 categories), this
    # is numerically identical to a flat 20-task macro; but the 3-level
    # form is robust to task drops (preserves equal-category-weight even if
    # some tasks fail to render / infer) and matches the design intent that
    # categories are the equal-weight unit at bench level.
    rows.append(bench_row(model, benchmark, all_problems, bench_cat_cie76,
                          task_subtask_problem_lists=bench_cat_problems,
                          n_boot=n_boot, seed=seed))
    return rows


def _process_tinygrafixbench(
    model: str,
    task_mode_map: dict[tuple[str, str], list[dict]],
    n_boot: int = 0,
    seed: int = 0,
) -> list[dict]:
    """Emit task/category/benchmark rows for TinyGrafixBench.

    TGF folders are named '{chart}_{subtask}' (e.g. 'bar_chart_edit_bars').
      level='task'      → one row per subtask (no mode aggregation)
      level='category'  → one row per chart type (equal subtask weight)
      level='benchmark' → one row (equal category weight)

    Benchmark-level CI uses a 3-level hierarchical bootstrap matching the
    doubly-macro point estimate: resample problems within each subtask,
    macro across subtasks per chart, macro across charts. See
    ``_bootstrap_ci_subtasks_then_tasks`` for the math and
    ``test_tgf_benchmark_ci_uses_subtask_hierarchical_not_chart_pool`` for
    the centering regression guard. Pooling problems across subtasks
    within a chart before the resample (the previous implementation) would
    silently produce a problem-weighted per-chart mean — internally
    inconsistent with the doubly-macro estimator at unequal subtask n.
    """
    benchmark = "TinyGrafixBench"

    chart_to_keys: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in sorted(task_mode_map):
        chart_to_keys[task_of(key[0], TGF_TASK_NAMES)].append(key)

    rows: list[dict] = []
    bench_chart_cie76:            list[dict]             = []
    bench_chart_subtask_problems: list[list[list[dict]]] = []   # 3-level for bench CI
    all_problems:                 list[dict]             = []

    for chart, keys in tqdm(sorted(chart_to_keys.items()), desc=f"  {model} TinyGrafixBench",
                            leave=False, position=_WORKER_POSITION + 1):
        chart_task_cie76:    list[dict]       = []
        chart_task_problems: list[list[dict]] = []
        chart_problems:      list[dict]       = []

        for (task_str, mode_str) in keys:
            probs = task_mode_map[(task_str, mode_str)]
            prefix     = chart + "_"
            task_label = task_str[len(prefix):] if task_str.startswith(prefix) else task_str
            cie76 = compute_cie76_stats(probs, n_boot, seed)
            chart_task_cie76.append(cie76)
            chart_task_problems.append(probs)
            chart_problems.extend(probs)
            rows.append({"level": "task", "model": model, "benchmark": benchmark,
                         "category": chart, "task": task_label,
                         **base_counts(probs), **cie76})

        all_problems.extend(chart_problems)
        # Category-level cie76 = equal subtask weight; used for benchmark roll-up.
        chart_cie76 = compute_cie76_stats_macro(chart_task_cie76,
                                                task_problem_lists=chart_task_problems,
                                                n_boot=n_boot, seed=seed)
        bench_chart_cie76.append(chart_cie76)
        # Keep per-subtask problem lists (not flattened per chart) so the
        # bench-level CI can do a 3-level hierarchical bootstrap matching
        # the doubly-macro point estimate.
        bench_chart_subtask_problems.append(chart_task_problems)
        rows.append(category_row(model, benchmark, chart, chart_problems,
                                 chart_task_cie76,
                                 task_problem_lists=chart_task_problems,
                                 n_boot=n_boot, seed=seed))

    # Benchmark row: equal category (chart) weight, where each per-chart
    # mean is itself an equal-subtask-weight macro. The point estimate is
    # `mean_over_charts(mean_over_subtasks(per_subtask_mean))` — a *doubly*
    # macro mean. The matching CI must therefore use 3-level hierarchical
    # bootstrap: resample problems within each subtask, macro across
    # subtasks within each chart, macro across charts. Pooling problems
    # across subtasks within a chart (the previous code path) would silently
    # produce a problem-weighted per-chart resample mean — internally
    # inconsistent with the doubly-macro estimator at any unequal subtask n
    # (today: equal n=30 holds in steady state, but render/inference drops
    # can break it; the variance is also slightly off even at equal n).
    rows.append(bench_row(model, benchmark, all_problems, bench_chart_cie76,
                          task_subtask_problem_lists=bench_chart_subtask_problems,
                          n_boot=n_boot, seed=seed))
    return rows


def process_paintbench(model: str, task_mode_map: dict[tuple[str, str], list[dict]],
                       n_boot: int = 0, seed: int = 0) -> list[dict]:
    return _process_paintbench(model, task_mode_map, n_boot=n_boot, seed=seed)


def process_tinygrafixbench(model: str, task_mode_map: dict[tuple[str, str], list[dict]],
                            n_boot: int = 0, seed: int = 0) -> list[dict]:
    return _process_tinygrafixbench(model, task_mode_map, n_boot=n_boot, seed=seed)


# ── Main ───────────────────────────────────────────────────────────────────────

def _process_one_model(args: tuple) -> list[dict]:
    """Worker: aggregate all rows for one model. Used by both serial and
    parallel paths in ``main()``. Returning a flat list keeps the parent's
    merge trivial and avoids inter-worker ordering concerns.

    args = (model, pb_map, tgf_map, n_boot, seed)
    """
    model, pb_map, tgf_map, n_boot, seed = args
    rows: list[dict] = []
    if pb_map:
        rows.extend(process_paintbench(model, pb_map, n_boot=n_boot, seed=seed))
    if tgf_map:
        rows.extend(process_tinygrafixbench(model, tgf_map, n_boot=n_boot, seed=seed))
    return rows


# Per-worker tqdm row position. Parent's outer "models" bar takes position 0;
# workers each take position WORKER_POSITION+1 so their inner per-task bars
# don't clobber each other or the parent's bar. Default 0 in the parent /
# serial path (where there's only one tqdm call active at a time at that
# level, so position 1 for the inner bar is fine).
_WORKER_POSITION: int = 0


def _init_worker(counter, lock) -> None:
    """ProcessPoolExecutor initializer for the parallel CI path.

    Assigns each worker a sequential index via a shared counter (so each
    worker owns a dedicated tqdm row) and wires up tqdm's cross-process
    write lock (so concurrent worker bars don't garble each other's
    output even when they update simultaneously).
    """
    global _WORKER_POSITION
    with counter.get_lock():
        _WORKER_POSITION = counter.value
        counter.value += 1
    tqdm.set_lock(lock)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input",  default="eval_outputs/problem_stats.jsonl")
    parser.add_argument("--output", default="eval_outputs/aggregate_stats.jsonl")
    parser.add_argument("--no-ci", action="store_true",
                        help="Skip the bootstrap CI step. Output rows omit ci95_* fields. "
                             "~5s instead of ~30-60s. Use for fast iteration when only "
                             "point estimates matter (e.g. debugging aggregation logic).")
    parser.add_argument("--n-boot", type=int, default=10_000,
                        help="Bootstrap iterations for 95%% CIs. Ignored with --no-ci.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for the bootstrap. Determines reproducibility of "
                             "the CIs across runs on the same input.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="Per-model parallelism for the bootstrap CI step. Each "
                             "model's rows are fully independent (no cross-model state), "
                             "so spreading the 12 models across N processes scales "
                             "near-linearly to N≤n_models. Default = os.cpu_count() "
                             "(matches eval.py / generate_benchmark.py defaults). On a "
                             "12-core machine with the default the full --n-boot 10000 "
                             "pipeline drops from ~6min serial to ~30-60s. Pass "
                             "--workers 1 to opt out (e.g. when running inside an "
                             "environment that blocks multiprocessing, like the Cursor "
                             "IDE on macOS).")
    args = parser.parse_args()

    n_boot  = 0 if args.no_ci else args.n_boot
    seed    = args.seed
    workers = max(1, args.workers)
    # --no-ci is ~3s of pure aggregation work; spawning N workers would cost
    # ~1-2s × N just for startup, with no parallelizable bootstrap to amortize
    # over. Force serial in that combo so `make stats NO_CI=1 JOBS=12` is the
    # same speed as `make stats NO_CI=1`.
    if n_boot == 0 and workers > 1:
        workers = 1

    # Load all records grouped by (model, benchmark, task, mode)
    data: dict[tuple, list[dict]] = defaultdict(list)
    with open(args.input) as f:
        for line in f:
            p = json.loads(line)
            data[(p["model"], p["benchmark"], p["task"], p.get("mode") or "")].append(p)

    models = sorted({k[0] for k in data})

    # Build per-model (pb_map, tgf_map) tuples once in the parent; serialize
    # only what each worker actually needs (no risk of pickling the global
    # ``data`` dict twice).
    def _maps_for(model: str) -> tuple[dict, dict]:
        pb_map  = {(task, mode): data[(model, "PaintBench",      task, mode)]
                   for (m, b, task, mode) in data
                   if m == model and b == "PaintBench"}
        tgf_map = {(task, mode): data[(model, "TinyGrafixBench", task, mode)]
                   for (m, b, task, mode) in data
                   if m == model and b == "TinyGrafixBench"}
        return pb_map, tgf_map

    jobs = [(model, *_maps_for(model), n_boot, seed) for model in models]

    rows: list[dict] = []
    if workers > 1 and len(jobs) > 1:
        # Per-model parallelism. Each worker takes one of the 12 models, runs
        # its full bootstrap CI computation, and returns the model's rows.
        # ``ex.map`` preserves input order → output row order is the same as
        # the serial path (models in alphabetical order), so the resulting
        # aggregate_stats.jsonl is byte-identical regardless of ``--workers``.
        #
        # Each worker gets a dedicated tqdm row via the counter+lock dance
        # in _init_worker: the parent's outer "models" bar takes row 0;
        # workers occupy rows 1..N, each showing its current model's per-task
        # progress. Without the shared lock + position assignment, all
        # workers' inner bars would clobber each other in row 0.
        n_workers = min(workers, len(jobs))
        ctx = mp.get_context("spawn")
        counter = ctx.Value("i", 0)
        lock = ctx.RLock()
        tqdm.set_lock(lock)
        with cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                                    initializer=_init_worker,
                                    initargs=(counter, lock)) as ex:
            for model_rows in tqdm(ex.map(_process_one_model, jobs),
                                   total=len(jobs), desc="models", position=0):
                rows.extend(model_rows)
        # Push the cursor below the highest worker row so subsequent prints
        # (e.g. the "Wrote N rows" line) don't overwrite a worker's old
        # position. tqdm leaves the screen with cursor at the highest active
        # bar, which is the parent's "models" bar at row 0.
        print("\n" * n_workers, end="")
    else:
        for job in tqdm(jobs, desc="models", position=0):
            rows.extend(_process_one_model(job))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    ci_tag = "without CIs (--no-ci)" if n_boot == 0 else f"with bootstrap CIs (B={n_boot}, seed={seed})"
    workers_tag = "" if workers == 1 else f", {workers} workers"
    print(f"Wrote {len(rows)} rows → {args.output}  ({ci_tag}{workers_tag})")


if __name__ == "__main__":
    main()
