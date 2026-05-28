import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .utils import (
    FONT_SIZES,
    apply_theme,
    make_rng,
    random_bg_and_text,
    random_color,
    random_magnitude,
    random_title,
    rgb_to_hex,
    round_sig,
)


TASKS = ["add_cell", "shift_heatmap", "mask_cells", "change_colormap"]

FIG_W_PX, FIG_H_PX, FIG_DPI = 1024, 768, 160


def build_state(seed):
    """Build the base state for a seed. Deterministic."""
    rng = make_rng(seed)
    rows = int(rng.integers(4, 10))
    cols = int(rng.integers(4, 10))

    scale = random_magnitude(rng)
    vmin = float(rng.uniform(-50.0, 50.0)) * scale
    vmax = vmin + float(rng.uniform(20.0, 100.0)) * scale
    data = rng.uniform(vmin, vmax, size=(rows, cols))

    # Kept below 0.25 so task_mask_cells still leaves a reasonably populated
    # grid after masking another 30–70% of the remaining cells.
    mask_frac = float(rng.uniform(0.1, 0.25))
    mask = rng.random((rows, cols)) < mask_frac
    data = np.where(mask, np.nan, data)
    # Ensure at least one empty tile for add_cell to target.
    if not np.isnan(data).any():
        i = int(rng.integers(0, rows))
        j = int(rng.integers(0, cols))
        data[i, j] = np.nan

    bg, text = random_bg_and_text(rng)
    # bg is the color shown for empty cells, so the two gradient endpoints
    # must be distinguishable from it.
    c1 = random_color(rng, avoid=[bg])
    c2 = random_color(rng, avoid=[c1, bg])

    title = random_title(rng)

    return {
        "rows": rows,
        "cols": cols,
        "data": data,
        "vmin": vmin,
        "vmax": vmax,
        "c1": c1,
        "c2": c2,
        "bg": bg,
        "text": text,
        "title": title,
    }


def _copy_state(s):
    return {
        "rows": s["rows"],
        "cols": s["cols"],
        "data": s["data"].copy(),
        "vmin": s["vmin"],
        "vmax": s["vmax"],
        "c1": s["c1"],
        "c2": s["c2"],
        "bg": s["bg"],
        "text": s["text"],
        "title": s["title"],
    }


def render_state(state):
    fig, ax = plt.subplots(
        figsize=(FIG_W_PX / FIG_DPI, FIG_H_PX / FIG_DPI),
        dpi=FIG_DPI,
        layout="constrained",
    )

    cmap = LinearSegmentedColormap.from_list("tg_heat", [state["c1"], state["c2"]])
    cmap.set_bad(state["bg"])

    im = ax.imshow(state["data"], cmap=cmap,
                   vmin=state["vmin"], vmax=state["vmax"],
                   aspect="auto")
    ax.set_title(state["title"])
    ax.set_xticks([])
    ax.set_yticks([])

    tick_vals = list(np.linspace(state["vmin"], state["vmax"], 5))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([f"{v:.3g}" for v in tick_vals])
    cbar.ax.tick_params(colors=state["text"], labelsize=FONT_SIZES["tick"])
    cbar.outline.set_edgecolor(state["text"])

    apply_theme(fig, ax, state["bg"], state["text"])
    return fig


def task_add_cell(base, rng):
    """Fill one of the empty (NaN) cells with a specified value."""
    nan_positions = [(i, j)
                     for i in range(base["rows"])
                     for j in range(base["cols"])
                     if np.isnan(base["data"][i, j])]
    idx = int(rng.integers(0, len(nan_positions)))
    i, j = nan_positions[idx]
    value = round_sig(rng.uniform(base["vmin"], base["vmax"]))
    input_state = _copy_state(base)
    answer_state = _copy_state(base)
    answer_state["data"][i, j] = value
    instruction = (
        f"Fill the empty cell at row {i + 1}, column {j + 1} (1-based indexing "
        f"from top left) with the color corresponding to the value {value:.3g}."
    )
    return input_state, answer_state, instruction


def task_mask_cells(base, rng):
    """Mask (set to empty) all cells strictly above or below a threshold."""
    above = bool(rng.random() < 0.5)
    flat = base["data"][~np.isnan(base["data"])]
    q = float(rng.uniform(0.3, 0.7))
    threshold = round_sig(np.quantile(flat, q))

    input_state = _copy_state(base)
    answer_state = _copy_state(base)
    data = answer_state["data"]
    if above:
        cond = (data > threshold)
    else:
        cond = (data < threshold)
    data[cond] = np.nan
    answer_state["data"] = data

    comparator = "greater than" if above else "less than"
    instruction = f"Remove every cell with a value {comparator} {threshold:.3g}."
    return input_state, answer_state, instruction


def task_change_colormap(base, rng):
    """Replace both colormap endpoint colors with new hex values."""
    new_c1 = random_color(rng, avoid=[base["c1"], base["c2"], base["bg"]])
    new_c2 = random_color(rng, avoid=[base["c1"], base["c2"], base["bg"], new_c1])

    input_state = _copy_state(base)
    answer_state = _copy_state(base)
    answer_state["c1"] = new_c1
    answer_state["c2"] = new_c2
    instruction = (
        f"Edit the heatmap and key to use a gradient with a low-value color of "
        f"{rgb_to_hex(new_c1)} and a high-value color of {rgb_to_hex(new_c2)}."
    )
    return input_state, answer_state, instruction


def task_shift_heatmap(base, rng):
    """Translate the whole heatmap by a small integer number of cells in
    one of the four cardinal directions. Cells that fall off the edge are
    clipped; cells on the opposite edge become empty."""
    amount = int(rng.integers(1, 3))  # 1 or 2
    direction = rng.choice(["up", "down", "left", "right"])
    dr = {"up": -amount, "down": amount, "left": 0, "right": 0}[direction]
    dc = {"up": 0, "down": 0, "left": -amount, "right": amount}[direction]

    rows, cols = base["rows"], base["cols"]
    new_data = np.full_like(base["data"], np.nan)
    for i in range(rows):
        for j in range(cols):
            ni, nj = i + dr, j + dc
            if 0 <= ni < rows and 0 <= nj < cols:
                new_data[ni, nj] = base["data"][i, j]

    input_state = _copy_state(base)
    answer_state = _copy_state(base)
    answer_state["data"] = new_data
    plural = "s" if amount > 1 else ""
    instruction = (
        f"Shift the heatmap {amount} cell{plural} {direction}. Cells that fall "
        f"off the edge should be discarded, and cells exposed on the opposite "
        f"side should become empty."
    )
    return input_state, answer_state, instruction


TASK_FNS = {
    "add_cell":        task_add_cell,
    "shift_heatmap":   task_shift_heatmap,
    "mask_cells":      task_mask_cells,
    "change_colormap": task_change_colormap,
}


def generate_task(seed, task):
    """Return (input_fig, answer_fig, instruction) for the given task."""
    if task not in TASK_FNS:
        raise ValueError(f"unknown task {task!r}; choose from {TASKS}")
    base = build_state(seed)
    task_seed = int(seed) * 17 + TASKS.index(task)
    rng = make_rng(task_seed)
    input_state, answer_state, instruction = TASK_FNS[task](base, rng)
    return render_state(input_state), render_state(answer_state), instruction
