"""Aggregation produces the expected row shape and hierarchy.

stats.py emits mode/task/category/visual_condition/benchmark rows for PaintBench,
and task/category/benchmark rows for TinyGrafixBench (subtask → task,
chart type → category). When `n_boot > 0` (the production default), every
cie76_t and cie76_mean block also carries 95% percentile-bootstrap CIs for
mean IoU / edit_accuracy / preservation_accuracy. We pump synthetic
per-problem records through and check the rollup schema + the CI invariants,
not specific numeric values (which eval.py already covers).

CI tests use n_boot=1000 for speed (vs the production default 10000) —
assertions check structural invariants (field presence, bracketing, width
sign) not exact percentile values, so the smaller B doesn't matter.
"""
from __future__ import annotations

import stats

# Smaller B for test speed. Production default is 10_000.
N_BOOT_TEST = 1000
SEED_TEST = 0


def _fake_problem(threshold_iou: float = 0.5) -> dict:
    """Minimal record matching what eval.py writes to problem_stats.jsonl,
    filled with pointwise-constant stats for easy hand-checking."""
    return {
        "has_output":          True,
        "correct_output_size": True,
        "output": {
            "cie76_threshold": {
                str(t): {
                    "edit_accuracy":         0.8,
                    "preservation_accuracy": 0.9,
                    "iou":                   threshold_iou,
                }
                for t in range(11)
            },
            "cie76_mean": {
                "edit_accuracy":         0.8,
                "preservation_accuracy": 0.9,
                "iou":                   threshold_iou,
            },
        },
    }


def _fake_problem_vcond(visual_condition: str, threshold_iou: float = 0.5) -> dict:
    p = _fake_problem(threshold_iou)
    p["visual_condition"] = visual_condition
    return p


# ── PaintBench level structure ────────────────────────────────────────────────

def test_paintbench_multi_mode_emits_mode_task_category_benchmark_rows():
    """A task with 2 modes → 2 mode rows + 1 task row + 1 category row + 1 benchmark row."""
    problems = [_fake_problem() for _ in range(3)]
    task_mode_map = {
        ("translation", "align"):  problems,
        ("translation", "amount"): problems,
    }
    rows = stats.process_paintbench(model="m", task_mode_map=task_mode_map)
    levels = [r["level"] for r in rows]
    assert levels.count("mode")      == 2
    assert levels.count("task")      == 1
    assert levels.count("category")  == 1
    assert levels.count("benchmark") == 1

    mode_rows = [r for r in rows if r["level"] == "mode"]
    assert {r["mode"] for r in mode_rows} == {"align", "amount"}

    cat_row = next(r for r in rows if r["level"] == "category")
    assert cat_row["category"] == "geometric_transformation"


def test_paintbench_single_mode_suppresses_mode_row():
    """A task with one mode → task + category + benchmark (no mode row)."""
    problems = [_fake_problem()]
    task_mode_map = {("construction", ""): problems}
    rows = stats.process_paintbench(model="m", task_mode_map=task_mode_map)
    levels = [r["level"] for r in rows]
    assert "mode" not in levels
    assert levels.count("task")      == 1
    assert levels.count("category")  == 1
    assert levels.count("benchmark") == 1
    assert next(r for r in rows if r["level"] == "task")["task"] == "construction"


def test_paintbench_category_rows_equal_weighted():
    """Category rows equal-weight per-task stats regardless of per-task problem count.

    task1 (blending, color_change): 3 problems iou=1.0
    task2 (construction, structural_manipulation): 1 problem iou=0.0
    Same category → two separate category rows, each with their own mean.
    Different category, so the benchmark tests equal-category-weight.
    """
    task_mode_map = {
        ("blending",     ""): [_fake_problem(1.0)] * 3,
        ("construction", ""): [_fake_problem(0.0)] * 3,
        ("gradient",     ""): [_fake_problem(1.0)] * 3,  # also color_change
    }
    rows = stats.process_paintbench(model="m", task_mode_map=task_mode_map)
    cat_rows = {r["category"]: r for r in rows if r["level"] == "category"}
    assert "color_change" in cat_rows
    assert "structural_manipulation" in cat_rows
    # color_change has blending (iou=1.0) and gradient (iou=1.0) → mean=1.0
    assert abs(cat_rows["color_change"]["cie76_mean"]["mean_iou"] - 1.0) < 1e-9
    # structural_manipulation has only construction (iou=0.0) → mean=0.0
    assert abs(cat_rows["structural_manipulation"]["cie76_mean"]["mean_iou"] - 0.0) < 1e-9


def test_paintbench_category_equal_task_weight():
    """Category mean is equal-task-weighted, not problem-count-weighted.

    blending: 3 problems iou=1.0
    gradient:  1 problem iou=0.0
    Equal-task-weighted color_change mean = 0.5; problem-weighted = 0.75.
    """
    task_mode_map = {
        ("blending",  ""): [_fake_problem(1.0)] * 3,
        ("gradient",  ""): [_fake_problem(0.0)],
    }
    rows = stats.process_paintbench(model="m", task_mode_map=task_mode_map)
    cat_row = next(r for r in rows if r["level"] == "category")
    assert cat_row["category"] == "color_change"
    assert abs(cat_row["cie76_mean"]["mean_iou"] - 0.5) < 1e-9


def test_paintbench_visual_condition_rows_equal_weighted():
    """Visual-condition rows equal-weight per-task stats regardless of per-task problem count.

    task1/baseline: 3 problems with iou=1.0
    task2/baseline: 1 problem  with iou=0.0
    Equal-weighted mean = 0.5; problem-count-weighted mean would be 0.75.
    """
    task_mode_map = {
        ("blending",      ""): [_fake_problem_vcond("baseline", 1.0)] * 3,
        ("construction",  ""): [_fake_problem_vcond("baseline", 0.0)],
    }
    rows = stats.process_paintbench(model="m", task_mode_map=task_mode_map)
    cond_rows = [r for r in rows if r["level"] == "visual_condition"]
    assert len(cond_rows) == 1
    assert cond_rows[0]["visual_condition"] == "baseline"
    assert abs(cond_rows[0]["cie76_mean"]["mean_iou"] - 0.5) < 1e-9


def test_paintbench_visual_condition_rows_one_per_vcond():
    """One visual_condition row is emitted per distinct visual_condition name."""
    task_mode_map = {
        ("blending", ""): [
            _fake_problem_vcond("baseline"),
            _fake_problem_vcond("n_high"),
            _fake_problem_vcond("striped"),
        ],
    }
    rows = stats.process_paintbench(model="m", task_mode_map=task_mode_map)
    cond_names = {r["visual_condition"] for r in rows if r["level"] == "visual_condition"}
    assert cond_names == {"baseline", "n_high", "striped"}


def test_row_counts_reflect_input():
    problems = [_fake_problem() for _ in range(7)]
    rows = stats.process_paintbench(
        model="m", task_mode_map={("construction", ""): problems}
    )
    task_row = next(r for r in rows if r["level"] == "task")
    assert task_row["n_problems"]            == 7
    assert task_row["n_with_output"]         == 7
    assert task_row["n_correct_output_size"] == 7


# ── TinyGrafixBench level structure ──────────────────────────────────────────

def test_tinygrafixbench_task_and_category_levels():
    """TGF: subtasks → level='task', chart types → level='category'."""
    task_mode_map = {
        ("bar_chart_edit_bars",       ""): [_fake_problem()],
        ("bar_chart_add_annotations", ""): [_fake_problem()],
    }
    rows = stats.process_tinygrafixbench(model="m", task_mode_map=task_mode_map)
    levels = [r["level"] for r in rows]
    assert levels.count("task")      == 2
    assert levels.count("category")  == 1
    assert levels.count("benchmark") == 1
    assert "mode" not in levels

    task_rows = [r for r in rows if r["level"] == "task"]
    assert {r["task"] for r in task_rows} == {"edit_bars", "add_annotations"}
    assert all(r["category"] == "bar_chart" for r in task_rows)

    cat_row = next(r for r in rows if r["level"] == "category")
    assert cat_row["category"] == "bar_chart"


def test_tinygrafixbench_no_visual_condition_rows():
    """TGF problems carry no visual_condition field → no visual_condition rows emitted."""
    task_mode_map = {("bar_chart_edit_bars", ""): [_fake_problem()]}
    rows = stats.process_tinygrafixbench(model="m", task_mode_map=task_mode_map)
    assert not any(r["level"] == "visual_condition" for r in rows)


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

_CI_FIELDS = [
    ("ci95_edit_accuracy_low",          "ci95_edit_accuracy_high"),
    ("ci95_preservation_accuracy_low",  "ci95_preservation_accuracy_high"),
    ("ci95_iou_low",                    "ci95_iou_high"),
]


def test_ci_fields_present_in_all_row_levels():
    """Every cie76 stats block has CI bounds for iou, edit accuracy, and preservation accuracy."""
    rows = stats.process_paintbench(
        model="m",
        task_mode_map={("blending", ""): [_fake_problem()] * 5},
        n_boot=N_BOOT_TEST, seed=SEED_TEST,
    )
    for row in rows:
        for key in [f"cie76_{t}" for t in range(11)] + ["cie76_mean"]:
            for lo, hi in _CI_FIELDS:
                assert lo in row[key], f"{row['level']}/{key} missing {lo}"
                assert hi in row[key], f"{row['level']}/{key} missing {hi}"


def test_ci_fields_omitted_when_n_boot_zero():
    """With n_boot=0 (--no-ci), ci95_* fields must be entirely absent (not null)."""
    rows = stats.process_paintbench(
        model="m",
        task_mode_map={("blending", ""): [_fake_problem(iou) for iou in [0.2, 0.5, 0.8]]},
        n_boot=0,
    )
    for row in rows:
        for key in [f"cie76_{t}" for t in range(11)] + ["cie76_mean"]:
            for lo, hi in _CI_FIELDS:
                assert lo not in row[key], f"{row['level']}/{key} unexpectedly has {lo} with n_boot=0"
                assert hi not in row[key], f"{row['level']}/{key} unexpectedly has {hi} with n_boot=0"


def test_ci_bounds_contain_mean():
    """For non-degenerate data: ci_low ≤ mean ≤ ci_high for all three metrics.

    Uses iou values with variance per problem so the bootstrap distribution
    is non-degenerate. Other metrics (edit_accuracy, preservation_accuracy)
    are constant in our synthetic problems → CI collapses to a point, which
    still satisfies ci_low ≤ mean ≤ ci_high.
    """
    problems = [_fake_problem(iou) for iou in [0.2, 0.5, 0.7, 0.9, 0.4]]
    rows = stats.process_paintbench(
        model="m",
        task_mode_map={("blending", ""): problems},
        n_boot=N_BOOT_TEST, seed=SEED_TEST,
    )
    metric_triples = [
        ("mean_edit_accuracy",         "ci95_edit_accuracy_low",         "ci95_edit_accuracy_high"),
        ("mean_preservation_accuracy", "ci95_preservation_accuracy_low", "ci95_preservation_accuracy_high"),
        ("mean_iou",                   "ci95_iou_low",                   "ci95_iou_high"),
    ]
    for row in rows:
        for key in [f"cie76_{t}" for t in range(11)] + ["cie76_mean"]:
            blk = row[key]
            for mean_f, lo_f, hi_f in metric_triples:
                assert blk[lo_f] <= blk[mean_f] <= blk[hi_f], (
                    f"{row['level']}/{key}/{mean_f}: [{blk[lo_f]}, {blk[hi_f]}] "
                    f"doesn't contain mean {blk[mean_f]}"
                )


def test_ci_collapses_with_constant_values():
    """When all problems have identical values, each CI collapses to a point.

    True at every level under method-1 hierarchical bootstrap: per-problem
    resampling of a constant list always yields the same mean, so neither
    the per-problem (task/mode level) nor the within-task hierarchical
    (category/visual_condition/benchmark level) bootstrap can produce a
    non-degenerate interval here.
    """
    problems = [_fake_problem(0.6)] * 10
    rows = stats.process_paintbench(
        model="m",
        task_mode_map={("blending", ""): problems},
        n_boot=N_BOOT_TEST, seed=SEED_TEST,
    )
    for row in rows:
        blk = row["cie76_mean"]
        assert abs(blk["ci95_iou_low"]                    - 0.6) < 1e-6
        assert abs(blk["ci95_iou_high"]                   - 0.6) < 1e-6
        assert abs(blk["ci95_edit_accuracy_low"]          - 0.8) < 1e-6
        assert abs(blk["ci95_edit_accuracy_high"]         - 0.8) < 1e-6
        assert abs(blk["ci95_preservation_accuracy_low"]  - 0.9) < 1e-6
        assert abs(blk["ci95_preservation_accuracy_high"] - 0.9) < 1e-6


# ── Bootstrap sampling unit (hierarchical-vs-pooled regression guard) ────────
#
# At category / visual_condition / benchmark level, the hierarchical bootstrap
# resamples PROBLEMS WITHIN EACH TASK and macro-averages the per-task means.
# This preserves the equal-task-weight aggregation structure used by the
# point estimate: the macro-mean is `mean(per_task_means)`, and the CI
# reflects the distribution of that quantity under within-task problem
# resampling. See stats.py::_bootstrap_ci_tasks for the math and the
# rationale for treating tasks as fixed (the benchmark IS the 20 tasks).
#
# The regression we guard against is the "pooled per-problem bootstrap":
# concatenating all problems across tasks and resampling from the pooled
# set, which gives a CI centered on the problem-weighted (not task-weighted)
# mean. The two centerings differ whenever per-task n's are unequal or
# per-task means are unequal — both common in PaintBench.

def test_category_macro_ci_uses_within_task_hierarchical_bootstrap():
    """Category-level CI uses within-task problem resampling, macro across tasks.

    Setup:
      task1 (blending,     color_change): 3 problems with iou ∈ {0.10, 0.20, 0.30}, mean 0.20
      task2 (gradient,     color_change): 3 problems with iou ∈ {0.70, 0.80, 0.90}, mean 0.80
    Macro mean = (0.20 + 0.80) / 2 = 0.50.

    Each bootstrap iter: resample 3 problems within each task, compute per-task
    mean, then macro across tasks. Because within-task variance is nonzero,
    CI width > 0 (sanity check); because both task-means contribute equally
    via macro, the CI is centered near 0.50 (not the problem-pooled mean,
    which would also be 0.50 in this symmetric case but for asymmetric
    n's would differ).
    """
    task_mode_map = {
        ("blending", ""): [_fake_problem(v) for v in [0.10, 0.20, 0.30]],
        ("gradient", ""): [_fake_problem(v) for v in [0.70, 0.80, 0.90]],
    }
    rows = stats.process_paintbench(model="m", task_mode_map=task_mode_map,
                                    n_boot=N_BOOT_TEST, seed=SEED_TEST)
    cat_row = next(r for r in rows if r["level"] == "category")
    blk = cat_row["cie76_mean"]

    assert abs(blk["mean_iou"] - 0.5) < 1e-9, "Point estimate must remain task-macro"
    assert blk["ci95_iou_low"] <= blk["mean_iou"] <= blk["ci95_iou_high"]
    width = blk["ci95_iou_high"] - blk["ci95_iou_low"]
    assert width > 0, (
        f"CI width = {width:.4f}; expected > 0 because per-task problems have "
        f"variance (within-task resampling should produce non-degenerate task means)."
    )


def test_benchmark_macro_ci_uses_within_task_hierarchical_bootstrap():
    """Same regression guard at the benchmark level — three tasks across categories."""
    task_mode_map = {
        ("blending",     ""): [_fake_problem(v) for v in [0.00, 0.10, 0.20]],   # task mean 0.10
        ("construction", ""): [_fake_problem(v) for v in [0.40, 0.50, 0.60]],   # task mean 0.50
        ("translation",  ""): [_fake_problem(v) for v in [0.80, 0.90, 1.00]],   # task mean 0.90
    }
    rows = stats.process_paintbench(model="m", task_mode_map=task_mode_map,
                                    n_boot=N_BOOT_TEST, seed=SEED_TEST)
    bench = next(r for r in rows if r["level"] == "benchmark")
    blk = bench["cie76_mean"]

    assert abs(blk["mean_iou"] - 0.5) < 1e-9
    assert blk["ci95_iou_low"] <= blk["mean_iou"] <= blk["ci95_iou_high"]
    width = blk["ci95_iou_high"] - blk["ci95_iou_low"]
    assert width > 0


def test_macro_ci_collapses_when_within_task_variance_is_zero():
    """Method-1 invariant: if every task has constant per-problem values,
    the hierarchical macro CI collapses to a single point.

    Within-task resampling of a constant list always yields the same per-task
    mean, so the macro of those means is also always the same — degenerate
    bootstrap distribution → zero-width CI. This is the *intentional*
    behaviour of method #1: tasks are treated as fixed, only within-task
    problem sampling drives the CI."""
    task_mode_map = {
        ("blending", ""): [_fake_problem(0.2)] * 5,   # zero within-task variance
        ("gradient", ""): [_fake_problem(0.8)] * 5,   # zero within-task variance
    }
    rows = stats.process_paintbench(model="m", task_mode_map=task_mode_map,
                                    n_boot=N_BOOT_TEST, seed=SEED_TEST)
    cat_row = next(r for r in rows if r["level"] == "category")
    blk = cat_row["cie76_mean"]

    assert abs(blk["mean_iou"] - 0.5) < 1e-9
    assert abs(blk["ci95_iou_high"] - blk["ci95_iou_low"]) < 1e-9, (
        "Method-1 hierarchical bootstrap should produce zero-width CI when "
        "within-task variance is zero (tasks are treated as fixed)."
    )


def test_macro_ci_centers_on_task_macro_not_problem_pool():
    """Discriminating regression guard: method-1 (within-task hierarchical)
    centers on the TASK-MACRO mean, not the problem-pooled mean.

    A pooled per-problem bootstrap (the bug we're guarding against) would
    silently regress to centering on the problem-weighted mean. The other
    macro CI tests above use symmetric designs (equal n × symmetric means)
    where the macro and pooled centerings *coincide* at 0.5, so they don't
    discriminate the two regimes. This test breaks that symmetry with
    unequal n × zero within-task variance:

      blending: 10 problems × iou=0.2 → per-task mean 0.2
      gradient:  2 problems × iou=0.8 → per-task mean 0.8

      task-macro mean:  (0.2 + 0.8) / 2          = 0.50  ← method-1 collapses here
      problem-pooled:   (10·0.2 + 2·0.8) / 12     ≈ 0.30  ← pooled-bootstrap regression would collapse here

    Within-task variance is zero by construction, so both bootstraps would
    give zero-width CIs; the centering of the (degenerate) bracket is what
    distinguishes them.
    """
    task_mode_map = {
        ("blending", ""): [_fake_problem(0.2)] * 10,
        ("gradient", ""): [_fake_problem(0.8)] *  2,
    }
    rows = stats.process_paintbench(model="m", task_mode_map=task_mode_map,
                                    n_boot=N_BOOT_TEST, seed=SEED_TEST)
    cat_row = next(r for r in rows if r["level"] == "category")
    blk = cat_row["cie76_mean"]

    assert abs(blk["mean_iou"] - 0.5) < 1e-9, "Point estimate must be task-macro"
    # CI collapses to 0.5 (macro) under method-1; would collapse to ~0.30
    # (problem-pooled) under a buggy pooled-bootstrap regression.
    assert abs(blk["ci95_iou_low"]  - 0.5) < 1e-6, (
        f"CI low {blk['ci95_iou_low']:.4f} ≠ 0.5; collapsing to ~0.30 would "
        f"indicate the bootstrap has regressed to pooled per-problem sampling."
    )
    assert abs(blk["ci95_iou_high"] - 0.5) < 1e-6


def test_paintbench_benchmark_ci_uses_category_hierarchical_not_task_flat():
    """PaintBench benchmark row uses 3-level hierarchical bootstrap matching
    its doubly-macro point estimate (task-macro per category → category-macro
    across categories) — not a flat 20-task macro.

    The two are numerically equivalent when task count per category is
    uniform (production: 5 tasks per category × 4 categories), but the
    3-level form is the conceptually correct estimator (each category
    contributes 1/4 of the bench mean by design) and is robust to task
    drops — flat task-macro would silently up-weight categories with
    more surviving tasks.

    Setup (zero within-task variance, unequal task count per category):
      blending      (color_change)             : 1 problem  × iou=0.2 → task mean 0.2
      gradient      (color_change)             : 1 problem  × iou=0.2 → task mean 0.2
      flood_fill    (color_change)             : 1 problem  × iou=0.2 → task mean 0.2
      translation   (geometric_transformation) : 1 problem  × iou=0.8 → task mean 0.8

    Per-category macro means:
      color_change             : (0.2 + 0.2 + 0.2) / 3 = 0.20  ← matches doubly-macro
      geometric_transformation : 0.80                          (lone task in category)

    Bench mean (category-macro):
      method-1 3-level (correct):   (0.20 + 0.80) / 2 = 0.50
      flat 20-task regression:      (0.2 + 0.2 + 0.2 + 0.8) / 4 = 0.35

    Within-task variance is zero, so the CI bracket collapses to a point
    under both regimes; the centering of that point discriminates them.
    """
    task_mode_map = {
        ("blending",    ""): [_fake_problem(0.2)],
        ("gradient",    ""): [_fake_problem(0.2)],
        ("flood_fill",  ""): [_fake_problem(0.2)],
        ("translation", ""): [_fake_problem(0.8)],
    }
    rows = stats.process_paintbench(model="m", task_mode_map=task_mode_map,
                                    n_boot=N_BOOT_TEST, seed=SEED_TEST)
    bench = next(r for r in rows if r["level"] == "benchmark")
    blk = bench["cie76_mean"]

    assert abs(blk["mean_iou"] - 0.5) < 1e-9, (
        f"Bench point estimate is {blk['mean_iou']:.4f}; should be 0.50 under "
        f"category-macro (3-level). 0.35 would indicate the bench has "
        f"regressed to a flat task-macro."
    )
    # CI collapses to 0.5 under 3-level method-1; would collapse to 0.35
    # under the flat-task-macro regression path.
    assert abs(blk["ci95_iou_low"]  - 0.5) < 1e-6
    assert abs(blk["ci95_iou_high"] - 0.5) < 1e-6


def test_tgf_benchmark_ci_uses_subtask_hierarchical_not_chart_pool():
    """TGF benchmark row uses 3-level hierarchical bootstrap matching its
    doubly-macro point estimate (subtask-macro per chart → chart-macro
    across charts) — not chart-pooled-then-macro.

    A chart-pooled-per-problem bootstrap (the path the previous code
    took, equivalent to ``_bootstrap_ci_tasks`` over ``chart_problems``)
    would give a *problem-weighted* per-chart resample mean. At equal
    subtask n that's numerically the same as the subtask-macro per chart,
    but at *unequal* subtask n it diverges — same anti-pattern as the
    PaintBench-benchmark CI bug this suite originally exposed, one
    hierarchy level deeper.

    Setup (zero within-subtask variance, unequal subtask n inside one chart):
      bar_chart_edit_bars       : 10 problems × iou=0.2 → subtask mean 0.2
      bar_chart_add_annotations :  2 problems × iou=0.8 → subtask mean 0.8
      line_chart_edit_line      :  5 problems × iou=0.5 → subtask mean 0.5
                                                          (lone subtask in
                                                           line_chart, no
                                                           sub-macro effect)

    Per-chart subtask-macro means:
      bar_chart  : (0.2 + 0.8) / 2 = 0.50  ← matches doubly-macro estimator
      line_chart : 0.50

    Per-chart problem-pooled means (the regression path):
      bar_chart  : (10·0.2 + 2·0.8) / 12 ≈ 0.30
      line_chart : 0.50

    Bench mean (chart-macro):
      method-1 (correct)        : (0.50 + 0.50) / 2 = 0.50
      pooled regression         : (0.30 + 0.50) / 2 = 0.40

    Within-subtask variance is zero, so the CI bracket collapses to a point
    under both regimes; the centering of that point discriminates them.
    """
    task_mode_map = {
        ("bar_chart_edit_bars",       ""): [_fake_problem(0.2)] * 10,
        ("bar_chart_add_annotations", ""): [_fake_problem(0.8)] *  2,
        ("line_chart_edit_line",      ""): [_fake_problem(0.5)] *  5,
    }
    rows = stats.process_tinygrafixbench(model="m", task_mode_map=task_mode_map,
                                         n_boot=N_BOOT_TEST, seed=SEED_TEST)
    bench = next(r for r in rows if r["level"] == "benchmark")
    blk = bench["cie76_mean"]

    assert abs(blk["mean_iou"] - 0.5) < 1e-9, "Point estimate must be doubly-macro"
    # CI collapses to 0.5 under 3-level method-1; would collapse to ~0.40
    # under the chart-pooled regression path.
    assert abs(blk["ci95_iou_low"]  - 0.5) < 1e-6, (
        f"CI low {blk['ci95_iou_low']:.4f} ≠ 0.5; collapsing to ~0.40 would "
        f"indicate the TGF bench bootstrap has regressed to chart-pooled "
        f"per-problem sampling (pooling subtasks within a chart before the "
        f"resample). See ``_bootstrap_ci_subtasks_then_tasks`` for the fix."
    )
    assert abs(blk["ci95_iou_high"] - 0.5) < 1e-6


# ── n_tasks field on aggregate rows (bug 12) ─────────────────────────────────

def test_aggregate_rows_report_n_tasks_alongside_n_problems():
    """category/visual_condition/benchmark rows expose both the bootstrap unit count
    (n_tasks) and the total problem volume (n_problems), so readers can tell
    'CI bootstrapped from N units' apart from 'covers N problems'.

    Semantics of ``n_tasks`` per level:
      - category / visual_condition : number of tasks contributing to the row
                                      (= bootstrap unit count at that level).
      - benchmark                   : number of CATEGORIES (= bootstrap unit
                                      count at bench under the symmetric
                                      3-level scheme; matches TGB which has
                                      always reported n_charts here).
    """
    task_mode_map = {
        ("blending",  ""): [_fake_problem_vcond("baseline", 1.0)] * 4,
        ("gradient",  ""): [_fake_problem_vcond("baseline", 0.0)] * 3,
    }
    rows = stats.process_paintbench(model="m", task_mode_map=task_mode_map)

    cat_row = next(r for r in rows if r["level"] == "category")
    assert cat_row["n_tasks"]    == 2     # blending + gradient (both color_change)
    assert cat_row["n_problems"] == 7     # 4 + 3

    cond_row = next(r for r in rows if r["level"] == "visual_condition")
    assert cond_row["n_tasks"]    == 2
    assert cond_row["n_problems"] == 7

    bench = next(r for r in rows if r["level"] == "benchmark")
    # 1 category (color_change) is the bench-level bootstrap unit — matches
    # the 3-level estimator's "macro over categories" outer step.
    assert bench["n_tasks"]    == 1
    assert bench["n_problems"] == 7

    task_row = next(r for r in rows if r["level"] == "task")
    assert "n_tasks" not in task_row, "task-level rows are the bootstrap unit themselves"


def test_tinygrafixbench_aggregate_rows_report_n_tasks():
    """TGF: category n_tasks = #subtasks per chart; benchmark n_tasks = #charts."""
    task_mode_map = {
        ("bar_chart_edit_bars",       ""): [_fake_problem()] * 3,
        ("bar_chart_add_annotations", ""): [_fake_problem()] * 4,
        ("line_chart_edit_line",      ""): [_fake_problem()] * 2,
    }
    rows = stats.process_tinygrafixbench(model="m", task_mode_map=task_mode_map)

    cat_rows = {r["category"]: r for r in rows if r["level"] == "category"}
    assert cat_rows["bar_chart"]["n_tasks"]  == 2     # edit_bars + add_annotations
    assert cat_rows["line_chart"]["n_tasks"] == 1     # edit_line only

    bench = next(r for r in rows if r["level"] == "benchmark")
    assert bench["n_tasks"] == 2     # bar_chart + line_chart


# ── Preservation task ────────────────────────────────────────────────────────

def test_paintbench_preservation_excluded_from_rollup():
    """Preservation task gets a task row but is excluded from category/visual_condition/benchmark."""
    problems = [_fake_problem()]
    task_mode_map = {
        ("translation",  "align"):     problems,
        ("preservation", "attribute"): problems,
    }
    rows = stats.process_paintbench(model="m", task_mode_map=task_mode_map)
    task_names = {r["task"] for r in rows if r["level"] == "task"}
    assert "preservation" in task_names        # has its own task row
    assert "translation"  in task_names

    # Category and benchmark roll-ups must reflect only the translation problem
    cat_row   = next(r for r in rows if r["level"] == "category")
    bench_row = next(r for r in rows if r["level"] == "benchmark")
    assert cat_row["n_problems"]   == 1
    assert bench_row["n_problems"] == 1

    # Preservation must never appear in any category
    for row in rows:
        if row["level"] == "category":
            assert row.get("task") != "preservation"

