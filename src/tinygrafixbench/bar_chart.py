import matplotlib.pyplot as plt

from .utils import (
    apply_theme,
    make_rng,
    random_axis_label,
    random_bg_and_text,
    random_color,
    random_magnitude,
    random_palette,
    random_string,
    random_title,
    rgb_to_hex,
    round_sig,
)


TASKS = ["add_bar", "sort_bars", "remove_bar", "recolor_bar"]

# Pixel dimensions are fixed per chart type (multiples of 64, <= 1024) so
# savefig output is exact. Layout inside the figure is handled by matplotlib's
# constrained_layout engine — no manual margins.
FIG_W_PX, FIG_H_PX, FIG_DPI = 1024, 768, 160
BAR_WIDTH = 0.8


def _is_sorted(values):
    asc = all(values[i] <= values[i + 1] for i in range(len(values) - 1))
    desc = all(values[i] >= values[i + 1] for i in range(len(values) - 1))
    return asc or desc


def build_state(seed):
    """Build the base state for a seed. Deterministic."""
    rng = make_rng(seed)
    n = int(rng.integers(4, 8))
    names = [random_string(rng) for _ in range(n)]

    bg, text = random_bg_and_text(rng)
    # Bars must contrast with the background, so seed the palette with bg
    # in its avoid list.
    colors = random_palette(rng, n, avoid=[bg])

    scale = random_magnitude(rng)
    y_high = float(rng.uniform(30.0, 150.0)) * scale

    # Resample until every pair of bars differs by at least 5% of y_high so
    # sort_bars always produces a visually distinct answer.
    min_gap = 0.05 * y_high
    for _ in range(200):
        values = rng.uniform(0.0, y_high, size=n).tolist()
        s = sorted(values)
        if all(s[i + 1] - s[i] >= min_gap for i in range(n - 1)):
            break
    # Avoid a monotone sequence, which would make sort_bars a no-op.
    if _is_sorted(values):
        values[0], values[1] = values[1], values[0]

    vmax = max(values)
    ylim = (0.0, vmax * 1.12)

    bars = [{"value": values[i], "color": colors[i]} for i in range(n)]
    title = random_title(rng)
    x_label = random_axis_label(rng)
    y_label = random_axis_label(rng)

    return {
        "n": n,
        "names": names,  # parallel to bars; "" means slot is blanked
        "bars": bars,    # list of {value, color} or None, length == n
        "ylim": ylim,
        "bg": bg,
        "text": text,
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
    }


def _copy_state(s):
    return {
        "n": s["n"],
        "names": list(s["names"]),
        "bars": [dict(b) if b is not None else None for b in s["bars"]],
        "ylim": s["ylim"],
        "bg": s["bg"],
        "text": s["text"],
        "title": s["title"],
        "x_label": s["x_label"],
        "y_label": s["y_label"],
    }


def render_state(state):
    fig, ax = plt.subplots(
        figsize=(FIG_W_PX / FIG_DPI, FIG_H_PX / FIG_DPI),
        dpi=FIG_DPI,
        layout="constrained",
    )
    n = state["n"]

    for i, bar in enumerate(state["bars"]):
        if bar is not None:
            ax.bar(i, bar["value"], width=BAR_WIDTH, color=bar["color"])

    ax.set_xticks(list(range(n)))
    ax.set_xticklabels(list(state["names"]))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(*state["ylim"])
    ax.set_title(state["title"])
    ax.set_xlabel(state["x_label"])
    ax.set_ylabel(state["y_label"])
    apply_theme(fig, ax, state["bg"], state["text"])
    ax.tick_params(axis="x", length=0)
    return fig


def task_add_bar(base, rng):
    """Delete one bar from the input (label stays visible); answer restores it."""
    k = int(rng.integers(0, base["n"]))
    input_state = _copy_state(base)
    input_state["bars"][k] = None
    # names[k] stays visible so the target label is in the input
    answer_state = _copy_state(base)
    value = round_sig(base["bars"][k]["value"])
    color = base["bars"][k]["color"]
    answer_state["bars"][k]["value"] = value
    instruction = (
        f'Add the bar for "{base["names"][k]}" with value {value:.3g} '
        f'and color {rgb_to_hex(color)}.'
    )
    return input_state, answer_state, instruction


def task_sort_bars(base, rng):
    """Sort bars ascending or descending by height; labels+colors follow."""
    ascending = bool(rng.random() < 0.5)
    order = sorted(range(base["n"]), key=lambda i: base["bars"][i]["value"],
                   reverse=not ascending)
    input_state = _copy_state(base)
    answer_state = _copy_state(base)
    answer_state["bars"] = [dict(base["bars"][i]) for i in order]
    answer_state["names"] = [base["names"][i] for i in order]
    direction = "ascending" if ascending else "descending"
    instruction = (
        f"Sort the bars in {direction} order, moving the corresponding labels."
    )
    return input_state, answer_state, instruction


def task_remove_bar(base, rng):
    """Blank the bar AND its label at a slot; keep every other bar in place."""
    k = int(rng.integers(0, base["n"]))
    input_state = _copy_state(base)
    answer_state = _copy_state(base)
    answer_state["bars"][k] = None
    answer_state["names"][k] = ""
    instruction = (
        f'Remove the bar and label for "{base["names"][k]}". Keep everything '
        f'else in the same place.'
    )
    return input_state, answer_state, instruction


def task_recolor_bar(base, rng):
    """Recolor one bar to a new hex."""
    k = int(rng.integers(0, base["n"]))
    new_color = random_color(
        rng, avoid=[b["color"] for b in base["bars"]] + [base["bg"]],
    )
    input_state = _copy_state(base)
    answer_state = _copy_state(base)
    answer_state["bars"][k]["color"] = new_color
    instruction = (
        f'Recolor the bar for "{base["names"][k]}" to {rgb_to_hex(new_color)}.'
    )
    return input_state, answer_state, instruction


TASK_FNS = {
    "add_bar":     task_add_bar,
    "sort_bars":   task_sort_bars,
    "remove_bar":  task_remove_bar,
    "recolor_bar": task_recolor_bar,
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
