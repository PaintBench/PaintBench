# Metric

PaintBench grades a model output against a known answer image at the
pixel level, using perceptually-weighted color distance. This document
walks through the pipeline from per-problem ΔE to the headline
benchmark number.

## What we compute (`src/eval.py`)

For each `(input, instruction, answer, model_output)` quadruple, eval
computes a pointwise CIE76 distance map (Lab-space Euclidean distance
between answer and model output), then summarises it under 11 ΔE
thresholds (0 through 10 inclusive).

### Three masks

Every problem partitions pixels into two regions:

- **Edit region** — pixels that *should* change from input to answer
  (`input != answer`).
- **Preservation region** — pixels that *should not* change from input
  to answer (`input == answer`).

These two regions together cover every pixel. The size of the edit
region varies by task — `recolor` edits a lot of pixels, `border` edits
few.

### Three accuracy numbers per threshold

For each ΔE threshold `t`, a pixel is **correct** if
`ΔE(model_output, answer) <= t`. Three numbers fall out:

| Number | Formula | What it measures |
|---|---|---|
| **edit_accuracy** | `correct ∩ edit / edit` | Did the model make the intended change correctly? (Recall over the edit region.) |
| **preservation_accuracy** | `correct ∩ preservation / preservation` | Did the model leave alone what it should have left alone? (Recall over the preservation region.) |
| **IoU** | `correct ∩ edit / (edit ∪ changed-in-preservation)` | Joint correctness: penalises both incorrect-edit and spillover-into-preservation. |

IoU is the headline metric — it's the only one of the three that
simultaneously punishes underediting AND overediting, so a model that
just returns the input unchanged scores 0 on IoU even though its
preservation_accuracy is perfect.

### The four-color diff PNG

When `--save-images` is on (the default), eval also writes a per-problem
ΔE diff PNG at each of four diagnostic thresholds (0, 2, 5, 10). The
color code:

| Color | Meaning |
|---|---|
| green | changed AND correct (the model edited where it should have, with the right color) |
| blue | unchanged AND correct (preservation worked) |
| red | changed BUT incorrect (the model edited but got the color wrong, OR edited a pixel that should have been preserved) |
| orange | unchanged BUT incorrect (the model failed to edit a pixel that should have changed) |

The visualizer reads these PNGs directly in the Eval tab — they're the
fastest way to see *what* a model got wrong, not just *how much*.

## Aggregation (`src/stats.py`)

The per-problem rows in `problem_stats.jsonl` are rolled up into
hierarchical aggregates in `aggregate_stats.jsonl`, with macro-averaging
designed to keep every aggregation level interpretable as
"equal-weight-over-X".

### PaintBench rollup hierarchy

- **task** — equal problem weight within a task (12 problems per task ×
  per-mode variation)
- **mode** — for multi-mode tasks, equal problem weight per mode
- **category** — equal **task** weight within each of the 4 task
  categories (geometric / color / structural / etc.)
- **visual_condition** — equal task weight per visual condition (8
  conditions × up to 20 tasks)
- **benchmark** — equal task weight across all 20 tasks (equivalently:
  equal category weight, doubly-macro)

### TinyGrafixBench rollup hierarchy

- **task** — equal problem weight per chart-subtask (30 problems)
- **category** — equal subtask weight per chart type (4 subtasks × 5
  chart types)
- **benchmark** — equal category weight, doubly-macro

### Bootstrap confidence intervals

Every aggregate row carries a 95% percentile-bootstrap CI for mean IoU,
mean edit_accuracy, and mean preservation_accuracy. The bootstrap unit
matches the aggregation level so the CI answers the same inferential
question as the displayed point estimate:

- **task / mode** rows — per-problem bootstrap (resample problems with
  replacement)
- **category** rows — 2-level hierarchical: resample problems within
  each task, then macro over tasks
- **benchmark** rows — 3-level hierarchical: resample within task,
  macro within category, macro across categories

The 3-level form reduces numerically to a flat task-macro when every
category has the same task count (the standard config), but the
hierarchical form is required for correctness under unequal counts
(e.g. when a task is dropped from a rerun). The bootstrap defaults to
B=10,000 resamples with seed 0.

CI computation is the expensive step (~30-60 s on a 12-core box
via per-model parallelism, ~6 min serial). Pass `make stats NO_CI=1`
to skip it for fast iteration (~3 s).

`tests/test_stats.py` has regression guards covering both the macro
math and the hierarchical bootstrap.

## Choosing a ΔE threshold

The 11 thresholds (0-10) sweep from "byte-exact" to "perceptibly
close." Suggested anchors:

- **ΔE=0** — pixel-byte exact. The strictest setting; succeeds only
  when the model reproduces the answer's RGB exactly. Most useful for
  the deterministic task subset where exact replication is feasible.
- **ΔE=2** — JND (just-noticeable difference). Considered the
  threshold for "indistinguishable to a human observer" in color
  science. Generous enough to forgive PNG compression artifacts and
  small numerical drift in the rendering pipeline.
- **ΔE=5** — visible-but-similar. Different colors that a human would
  call "the same color" (e.g. two adjacent samples from a palette).
- **ΔE=10** — clearly different colors that share a perceptual family.

The paper reports across all 11 thresholds and uses pass-rate sweeps
(fraction of problems exceeding IoU=k at threshold=t) to characterize
the full failure surface — see the paper for the headline numbers.

## Why CIE76 (and not L2, SSIM, LPIPS, or an LLM judge)

- **L2 / MSE** in RGB doesn't match human perception (a 10-unit shift
  in blue is much less visible than the same shift in green). CIE76
  in Lab is the simplest fix; CIE94 / CIEDE2000 are more accurate but
  add complexity for marginal gain at the per-pixel scale we care
  about.
- **SSIM** measures structural similarity, which is the *wrong*
  notion for this benchmark — PaintBench's answers are deterministic
  pixel-exact targets, not "looks similar." A model that returns a
  blurred version of the right edit would do well on SSIM but should
  score 0 on a benchmark that asks for exact replacement.
- **LPIPS / DINO-similarity** require a frozen feature extractor,
  which (a) makes the metric a moving target as backbones change and
  (b) gives credit for "semantically close" rather than "correct."
- **LLM-as-judge** introduces a second model's biases into the
  evaluator and is impossible to reproduce without API access to a
  specific model snapshot. PaintBench's CIE76 path is pure numpy,
  CPU-only, no network — bit-for-bit reproducible across machines.

The cost of this choice is that PaintBench is hostile to models that
make *plausible* edits to *the wrong region*: there's no partial credit
for "good edit, wrong place." That's by design — the benchmark is built
to be sensitive to exactly this failure mode.
