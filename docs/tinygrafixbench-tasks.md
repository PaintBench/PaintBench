# TinyGrafixBench tasks

TinyGrafixBench has **5 chart types × 4 tasks = 20 task variants**. With
30 seeds per variant, the benchmark is 600 problems total. Every problem
is a deterministic `(input_figure, instruction, answer_figure)` triplet
generated from a seed.

The four tasks per chart cover one operation from each of these
categories:

| Category | What it tests |
|---|---|
| **Construction** | Add a missing element back to the chart |
| **Transformation** | Reorder, rescale, or remap existing elements |
| **Removal** | Drop one or more elements |
| **Recoloring** | Change one or more colors, propagating through any legend |

Generators live in `src/tinygrafixbench/<chart>.py` and export a `TASKS`
list plus a `generate_task(seed, task) -> (input_fig, answer_fig, instruction)`
function.

---

## Network (`network.py`)

A small undirected graph laid out on a circle, with a color-coded
node-label key beside it.

| Task | Description | Example instruction |
|---|---|---|
| `add_node` | Restore one node (and its edges) that was deleted from the graph but kept in the key | *Add the node "M9" so that all nodes are evenly spaced on a circle. Connect it to "Q4", "R7".* |
| `swap_nodes` | Swap the positions of two nodes; their edges follow | *Swap the positions of nodes "M9" and "Q4".* |
| `remove_node` | Delete a node and its incident edges; the key stays intact | *Remove node "M9" and its incident edges. Leave the key unchanged.* |
| `recolor_node` | Change one node's color in both the graph and the key | *Recolor node "M9" to #2c7be5. Update the color in both the graph and the key.* |

## Bar chart (`bar_chart.py`)

A vertical bar chart with 4–8 named bars whose heights are pairwise
distinct.

| Task | Description | Example instruction |
|---|---|---|
| `add_bar` | Restore one bar that was deleted from the input (the label stays visible) | *Add the bar for "Q4" with value 87.5 and color #c0392b.* |
| `sort_bars` | Sort bars ascending or descending by height; labels follow | *Sort the bars in descending order, moving the corresponding labels.* |
| `remove_bar` | Blank one bar and its label; every other bar stays in place | *Remove the bar and label for "Q4". Keep everything else in the same place.* |
| `recolor_bar` | Recolor one bar to a new hex value | *Recolor the bar for "Q4" to #2c7be5.* |

## Scatter plot (`scatter_plot.py`)

Two classes of points on a square axis (same x/y scale). Class A has a
line of best fit; class B does not.

| Task | Description | Example instruction |
|---|---|---|
| `draw_best_fit_line` | Draw the OLS line of best fit for class B (matching its color and line thickness) | *Draw the line of best fit for the class of points without a line. Use the same color as those points and the same thickness as the existing line.* |
| `swap_axes` | Swap x and y coordinates of every point and the existing fit line | *Swap the x and y coordinates of every point and the line of best fit.* |
| `remove_outlier` | Remove the class-A point furthest vertically from the fit line; leave the line unchanged | *In the class of points with the line of best fit, remove the point that is vertically furthest from the line. Keep the line in place.* |
| `recolor_class` | Recolor one of the two classes (and its fit line if present) to a new hex | *Recolor the line of best fit and its corresponding points to #2c7be5.* |

## Heatmap (`heatmap.py`)

A rectangular value grid with a continuous two-endpoint colormap and a
key. Some cells are empty (NaN) by construction.

| Task | Description | Example instruction |
|---|---|---|
| `add_cell` | Fill one empty cell with a specified value (rendered via the colormap) | *Fill the empty cell at row 3, column 5 (1-based indexing from top left) with the color corresponding to the value 42.7.* |
| `shift_heatmap` | Translate the whole heatmap 1–2 cells in a cardinal direction; clip edge, expose empty | *Shift the heatmap 2 cells up. Cells that fall off the edge should be discarded, and cells exposed on the opposite side should become empty.* |
| `mask_cells` | Empty (NaN) every cell strictly above or below a threshold | *Remove every cell with a value greater than 60.5.* |
| `change_colormap` | Replace both colormap endpoint colors with new hex values | *Edit the heatmap and key to use a gradient with a low-value color of #2c7be5 and a high-value color of #f39c12.* |

## Line chart (`line_chart.py`)

A single noisy time-series line with axis labels. Single-line by design
— no legend, no multi-series.

| Task | Description | Example instruction |
|---|---|---|
| `draw_segments` | Bridge 2–3 interior gaps in the line with straight segments in the same color and width | *Connect the gaps with straight segments in the same width and color as existing segments.* |
| `normalize_series` | Rescale the y-series so its min/max land on specified targets, axis limits unchanged | *Scale and shift the series vertically so its lowest point corresponds to "Rate" = 12.5 and its highest point corresponds to "Rate" = 89.0. Keep the axes unchanged.* |
| `filter_series` | Keep only the parts of the line where y satisfies a threshold; clip exactly at the crossing | *Only show the parts of the series where "Rate" is at most 47.3.* |
| `shade_interval` | Shade the area under the line between two x values with a given color | *In the plot, shade the area under the series between "Year" = 2025 and "Year" = 2032 with the color #2c7be5.* |

---

## Adding a chart type

To add a new chart family, create `src/tinygrafixbench/<chart>.py` that:

1. Exports `TASKS: list[str]` — exactly 4 task names, one per category
   (Construction / Transformation / Removal / Recoloring).
2. Implements `build_state(seed) -> dict` that constructs the base scene
   deterministically from a seed.
3. Implements `task_<name>(base, rng) -> (input_state, answer_state, instruction)`
   for each task in `TASKS`. The `rng` is per-task; mutate `base` only
   via `_copy_state`.
4. Exports `generate_task(seed, task) -> (input_fig, answer_fig, instruction)`
   that dispatches by task name.
5. Implements a `render_state(state)` helper that returns a `matplotlib.figure.Figure`
   at fixed `FIG_W_PX × FIG_H_PX` at `FIG_DPI` so output PNG dimensions
   are exact.

All randomness must use `numpy.random.default_rng` seeded from the
provided seed. Determinism is enforced by `tests/test_determinism.py`.
