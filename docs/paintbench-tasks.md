# PaintBench tasks

PaintBench has **4 task categories × 5 tasks = 20 tasks**, evaluated
across **8 visual conditions × 12 problems = 96 problems each**, for a
total of **1,920** scored problems. A `preservation/` diagnostic split
adds 96 problems (excluded from scoring) that probe a model's
input-fidelity floor.

The four categories correspond to broad capability axes; the five tasks
in each category probe different facets of that axis.

| Category | Tasks |
|---|---|
| **Geometric transformation** | translation, rotation, reflection, scaling, shearing |
| **Structural manipulation** | construction, removal, copying, border, cropping |
| **Color change** | recolor, flood_fill, blending, gradient, point_operations |
| **Symbolic reasoning** | comparison, ordering, pattern, counting, legend |

Task generators live in `src/tasks/<name>.py` and are registered in the
`TASKS` list at the top of `src/generate_benchmark.py`. See
[`extending.md`](extending.md) for the recipe to add a new task.

## Visual conditions

Each task is generated across 8 visual conditions that vary one axis
of the scene at a time from the baseline. This is the primary
robustness-probe dimension of PaintBench — a model that does well on
baseline but collapses on `striped` is hiding a brittle background
assumption.

| Condition | Canvas | Palette | Background | Object count |
|---|---|---|---|---|
| `baseline` | 1024×1024 | standard | solid | default (3–60 depending on task) |
| `horizontal` | 1024×576 | standard | solid | default |
| `vertical` | 576×1024 | standard | solid | default |
| `nonstandard` | 1024×1024 | nonstandard | solid | default |
| `striped` | 1024×1024 | standard | striped | default |
| `n_med` | 1024×1024 | standard | solid | medium |
| `n_high` | 1024×1024 | standard | solid | high |
| `n_xhigh` | 1024×1024 | standard | solid | extra-high |

Per-task object counts at each `n_level` are tuned to keep the task
solvable and visually legible at scale. See `_n_levels_for(task_name)`
in `src/generate_benchmark.py` for the exact mapping (most tasks use
`[3, 10, 25, 60]`; `pattern`, `comparison`, `ordering`, `counting`
override).

## Modes

Multi-mode tasks parameterise a key structural axis of the task,
splitting the 12 per-(task, visual condition) problems evenly across
the modes. For example, `translation` has two modes (`amount` and
`align`), so each (visual condition) yields 6 problems per mode.

## Tasks by category

Every task exports `NAME`, `PARAMETERS` (parameter grid), and
`generate(seed, bg_spec, W, H, obj_colors, **kwargs) -> Problem`. The
table per category gives the modes (or `—` if single-mode) and a one-
or two-line description of what each mode does.

### Geometric transformation

| Task | Modes | Description |
|---|---|---|
| `translation` | `amount`, `align` | `amount`: move a shape by a specified (dx, dy) in pixels. `align`: move a shape so one of its control points (corner / midpoint / center) lands on a target control point. |
| `rotation` | `local`, `external` | `local`: rotate a shape about its own center. `external`: rotate it about a named external pivot (a canvas control point or another shape's centroid). |
| `reflection` | `local`, `external` | `local`: reflect a shape across a chord defined inside its bounding box. `external`: reflect across a canvas axis or a bbox edge of another shape. |
| `scaling` | `amount`, `match` | `amount`: scale a shape by a factor about an anchor. `match`: scale it so its bbox side / corner matches a reference shape's. |
| `shearing` | — | Shear a shape relative to a fixed edge by a specified slope. |

### Structural manipulation

| Task | Modes | Description |
|---|---|---|
| `construction` | `line`, `circle`, `polygon` | Draw a new filled shape with specified geometry and color. The three modes cover the three shape primitives PaintBench supports. |
| `removal` | `attribute`, `location` | `attribute`: remove all shapes matching an attribute (shape type or color). `location`: remove all shapes inside/outside a spatial region. |
| `copying` | — | Copy a shape to a new location (anchor-aligned). |
| `border` | — | Add a colored outline of specified thickness around a shape. |
| `cropping` | `straight`, `tilted` | `straight`: crop to an axis-aligned rectangle and scale to fill the canvas. `tilted`: same but for a rotated rectangle. |

### Color change

| Task | Modes | Description |
|---|---|---|
| `recolor` | `color_code`, `dropper` | `color_code`: recolor a target shape to a specified hex value. `dropper`: recolor it to match an existing shape's color (no hex specified). |
| `flood_fill` | `background`, `foreground` | Replace all pixels of one color region (the background or a foreground shape) with a new color. |
| `blending` | — | Alpha-blend a color over a polygonal region. |
| `gradient` | `background`, `foreground` | Apply a linear color gradient inside the background or a foreground shape, defined by a parallelogram region. |
| `point_operations` | `invert`, `grayscale`, `brightness` | Apply a per-pixel operation to selected shapes: invert RGB, convert to grayscale, or shift brightness by a signed delta. |

### Symbolic reasoning

| Task | Modes | Description |
|---|---|---|
| `comparison` | — | Remove the shape at a specified ordinal rank by attribute (e.g. the third-largest, the second-most-red). |
| `ordering` | — | Rearrange same-type shapes by size along a specified axis (smallest-to-largest, left-to-right). |
| `pattern` | `grid`, `circular` | Complete missing cells in a 2D repeating pattern arranged on a grid (`grid` mode) or around a circle (`circular`). |
| `counting` | — | Adjust a row of tally marks (or numeric annotation) to match a count of target shapes elsewhere in the scene. |
| `legend` | — | Generate a color legend mapping each visible shape's color to a label. |

## The preservation diagnostic

The `preservation/` folder is a 96-problem diagnostic split, NOT scored
in any aggregate metric. It's built from the `removal/` task: for each
of the 96 (visual condition × seed) `removal` problems, preservation
takes the same input image but sets `answer = input` (no edit needed).

Reading off the per-problem stats from this split tells you the
**input-fidelity floor** for a model — how well it can copy an input
forward when the correct behaviour is "do nothing." A model that
preserves the input perfectly here but fails everywhere else has a
different failure profile than one that mangles even the baseline.

Use it like:

```bash
# Compare a model's preservation_accuracy on preservation/ vs main tasks
make eval
# then inspect eval_outputs/problem_stats.jsonl rows with task == "preservation"
```

## Per-problem on-disk layout

For each problem, three files land under
`benchmarks/PaintBench/<task>/`:

```
NNN_input.png        # the input image the model sees
NNN_answer.png       # the pixel-exact target
NNN.json             # metadata: instruction, seed, mode, visual_condition, task-specific params
```

`NNN` is a zero-padded 3-digit index within the task folder. See
[`metric.md`](metric.md) for what eval does with these triplets and
[`run_cycle.md`](run_cycle.md) for the commands that produce them.

## Adding a task

See [`extending.md`](extending.md) for the full recipe. The short
version:

1. New file `src/tasks/<name>.py` exporting `NAME`, `PARAMETERS`, and
   `generate(...) -> Problem`.
2. Append `("tasks.<name>", "<name>")` to `TASKS` in
   `src/generate_benchmark.py`, plus the task → category mapping in
   `TASK_CATEGORIES`.
3. `tests/test_task_registry.py` picks it up automatically on the next
   run; commit the new byte-hashes printed by
   `tests/test_determinism.py` alongside the task code.
