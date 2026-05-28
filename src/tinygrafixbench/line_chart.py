import matplotlib.pyplot as plt
import numpy as np

from .utils import (
    apply_theme,
    make_rng,
    random_axis_label,
    random_bg_and_text,
    random_color,
    random_magnitude,
    random_title,
    rgb_to_hex,
    round_sig,
)


TASKS = ["draw_segments", "normalize_series", "filter_series", "shade_interval"]

FIG_W_PX, FIG_H_PX, FIG_DPI = 1024, 768, 160
LINE_WIDTH = 2.0


def build_state(seed):
    """Build the base state for a seed. Deterministic."""
    rng = make_rng(seed)
    n = int(rng.integers(20, 45))

    # Per-problem magnitude: scales x range, y-step size, and y start together.
    scale = random_magnitude(rng)
    x_low = float(rng.uniform(-50.0, 50.0)) * scale
    x_high = x_low + float(rng.uniform(40.0, 100.0)) * scale
    xs = np.linspace(x_low, x_high, n)

    step_scale = float(rng.uniform(0.5, 4.0)) * scale
    steps = rng.normal(0.0, step_scale, size=n)
    start = float(rng.uniform(-30.0, 30.0)) * scale
    ys = np.cumsum(steps) + start

    y_min, y_max = float(ys.min()), float(ys.max())
    y_pad = 0.15 * (y_max - y_min)
    ylim = (y_min - y_pad, y_max + y_pad)

    bg, text = random_bg_and_text(rng)
    color = random_color(rng, avoid=[bg])
    title = random_title(rng)
    x_label = random_axis_label(rng)
    y_label = random_axis_label(rng)

    return {
        "xs": xs,
        "ys": ys,
        "n": n,
        "pieces": [np.arange(n)],  # list of index arrays; each piece = one polyline
        "color": color,
        "xlim": (x_low, x_high),
        "ylim": ylim,
        "shade": None,  # None or {"x_low", "x_high", "color"}
        "bg": bg,
        "text": text,
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
    }


def _copy_state(s):
    return {
        "xs": np.array(s["xs"]),
        "ys": np.array(s["ys"]),
        "n": s["n"],
        "pieces": [np.array(p, dtype=int) for p in s["pieces"]],
        "color": s["color"],
        "xlim": s["xlim"],
        "ylim": s["ylim"],
        "shade": dict(s["shade"]) if s["shade"] is not None else None,
        "bg": s["bg"],
        "text": s["text"],
        "title": s["title"],
        "x_label": s["x_label"],
        "y_label": s["y_label"],
    }


def _contiguous_runs(indices):
    """Split a sorted 1-D array of indices into contiguous runs."""
    if len(indices) == 0:
        return []
    runs = []
    start = int(indices[0])
    prev = start
    for idx in indices[1:]:
        i = int(idx)
        if i == prev + 1:
            prev = i
        else:
            runs.append(np.arange(start, prev + 1))
            start = i
            prev = i
    runs.append(np.arange(start, prev + 1))
    return runs


def render_state(state):
    fig, ax = plt.subplots(
        figsize=(FIG_W_PX / FIG_DPI, FIG_H_PX / FIG_DPI),
        dpi=FIG_DPI,
        layout="constrained",
    )

    if state["shade"] is not None:
        sh = state["shade"]
        mask = (state["xs"] >= sh["x_low"]) & (state["xs"] <= sh["x_high"])
        idx = np.where(mask)[0]
        if len(idx) > 0:
            xs_s = state["xs"][idx]
            ys_s = state["ys"][idx]
            ax.fill_between(xs_s, ys_s, state["ylim"][0],
                            color=sh["color"], zorder=1)

    for piece in state["pieces"]:
        if len(piece) == 0:
            continue
        xs = state["xs"][piece]
        ys = state["ys"][piece]
        ax.plot(xs, ys, color=state["color"], linewidth=LINE_WIDTH, zorder=2)

    ax.set_xlim(*state["xlim"])
    ax.set_ylim(*state["ylim"])
    ax.set_title(state["title"])
    ax.set_xlabel(state["x_label"])
    ax.set_ylabel(state["y_label"])
    apply_theme(fig, ax, state["bg"], state["text"])
    return fig


def task_draw_segments(base, rng):
    """Input: line has several interior gaps of variable widths; answer: whole.
    Every kept run has at least 2 points so it renders as a segment (never a
    lone dot). Beginning and end segments are always present."""
    n = base["n"]
    # Choose number of gaps; keep small relative to n.
    n_gaps = int(rng.integers(2, 4))  # 2 or 3
    # Carve gaps of length 2-5 in interior, never touching endpoints, and
    # leaving at least 2 visible points between adjacent gaps so every run
    # in the input renders as a segment.
    missing = set()
    attempts = 0
    gaps_placed = 0
    while gaps_placed < n_gaps and attempts < 200:
        attempts += 1
        gap_len = int(rng.integers(2, 6))
        # Interior placement: start at >=2, end at <= n - 3.
        start = int(rng.integers(2, max(3, n - gap_len - 2)))
        candidate = set(range(start, start + gap_len))
        # Must not touch or overlap existing missing indices (need >=2 visible
        # points between gaps), and must preserve endpoints.
        if 0 in candidate or n - 1 in candidate:
            continue
        buffer = set(range(min(candidate) - 2, max(candidate) + 3))
        if buffer & missing:
            continue
        missing |= candidate
        gaps_placed += 1

    kept = np.array(sorted(set(range(n)) - missing), dtype=int)
    input_state = _copy_state(base)
    input_state["pieces"] = _contiguous_runs(kept)

    # Answer: a single polyline through the kept points only. Dropping the
    # missing indices (instead of interpolating them) means each bridged gap
    # becomes one straight segment with no intermediate vertices, so there
    # are no miter-join artifacts that would look like bends.
    answer_state = _copy_state(base)
    answer_state["xs"] = base["xs"][kept]
    answer_state["ys"] = base["ys"][kept]
    answer_state["n"] = len(kept)
    answer_state["pieces"] = [np.arange(len(kept))]
    instruction = (
        "Connect the gaps with straight segments in the same width and color "
        "as existing segments."
    )
    return input_state, answer_state, instruction


def task_normalize_series(base, rng):
    """Rescale ys (shift+scale) so min/max land on specified targets, with
    the axis limits unchanged."""
    y_lo, y_hi = base["ylim"]
    span = y_hi - y_lo
    # Pick a target [lo_t, hi_t] strictly inside the current ylim. The
    # disjoint sampling windows guarantee hi_t - lo_t >= 0.20 * span, so
    # the normalized line always occupies a visible fraction of the axis.
    lo_t = round_sig(rng.uniform(y_lo + 0.10 * span, y_lo + 0.40 * span))
    hi_t = round_sig(rng.uniform(y_lo + 0.60 * span, y_lo + 0.90 * span))

    ys = base["ys"]
    cur_min = float(np.min(ys))
    cur_max = float(np.max(ys))
    cur_range = max(cur_max - cur_min, 1e-9)
    new_ys = (ys - cur_min) / cur_range * (hi_t - lo_t) + lo_t

    input_state = _copy_state(base)
    answer_state = _copy_state(base)
    answer_state["ys"] = new_ys
    instruction = (
        f'Scale and shift the series vertically so its lowest point '
        f'corresponds to "{base["y_label"]}" = {lo_t:.3g} and its highest '
        f'point corresponds to "{base["y_label"]}" = {hi_t:.3g}. Keep the '
        f'axes unchanged.'
    )
    return input_state, answer_state, instruction


def task_filter_series(base, rng):
    """Keep only parts of the line whose y-value satisfies a threshold. The
    line is clipped at the exact crossing, so the answer's polyline ends
    flush with the threshold rather than at the last in-range data point."""
    # keep_at_most=True means keep y <= threshold (filter out what's above);
    # keep_at_most=False means keep y >= threshold (filter out what's below).
    keep_at_most = bool(rng.random() < 0.5)
    label = base["y_label"]
    ys = np.asarray(base["ys"], dtype=float)
    xs = np.asarray(base["xs"], dtype=float)
    y_min, y_max = float(ys.min()), float(ys.max())
    q = float(rng.uniform(0.35, 0.65))
    threshold = round_sig(y_min + q * (y_max - y_min))

    def passes(y):
        return y <= threshold if keep_at_most else y >= threshold

    new_xs = []
    new_ys = []
    pieces = []
    current = []

    def push_point(x, y):
        current.append(len(new_xs))
        new_xs.append(float(x))
        new_ys.append(float(y))

    def flush():
        nonlocal current
        if current:
            pieces.append(np.array(current, dtype=int))
        current = []

    n = len(xs)
    if n > 0 and passes(ys[0]):
        push_point(xs[0], ys[0])
    for i in range(1, n):
        y_prev, y_cur = float(ys[i - 1]), float(ys[i])
        x_prev, x_cur = float(xs[i - 1]), float(xs[i])
        prev_ok = passes(y_prev)
        cur_ok = passes(y_cur)
        if prev_ok and cur_ok:
            push_point(x_cur, y_cur)
        elif prev_ok and not cur_ok:
            t = (threshold - y_prev) / (y_cur - y_prev)
            push_point(x_prev + t * (x_cur - x_prev), threshold)
            flush()
        elif not prev_ok and cur_ok:
            t = (threshold - y_prev) / (y_cur - y_prev)
            push_point(x_prev + t * (x_cur - x_prev), threshold)
            push_point(x_cur, y_cur)
        # else: both out of range — nothing to emit
    flush()

    input_state = _copy_state(base)
    answer_state = _copy_state(base)
    answer_state["xs"] = np.array(new_xs, dtype=float)
    answer_state["ys"] = np.array(new_ys, dtype=float)
    answer_state["n"] = len(new_xs)
    answer_state["pieces"] = pieces

    comparator = "at most" if keep_at_most else "at least"
    instruction = (
        f'Only show the parts of the series where "{label}" is {comparator} '
        f'{threshold:.3g}.'
    )
    return input_state, answer_state, instruction


def task_shade_interval(base, rng):
    """Shade the area under the line between two x values with a given color."""
    x_lo, x_hi = base["xlim"]
    span = x_hi - x_lo
    # Disjoint sampling windows guarantee b - a >= 0.20 * span so the
    # shaded region is always a visible fraction of the axis.
    a = round_sig(rng.uniform(x_lo + 0.10 * span, x_lo + 0.40 * span))
    b = round_sig(rng.uniform(x_lo + 0.60 * span, x_lo + 0.90 * span))
    shade_color = random_color(rng, avoid=[base["color"], base["bg"]])

    input_state = _copy_state(base)
    answer_state = _copy_state(base)
    answer_state["shade"] = {"x_low": a, "x_high": b, "color": shade_color}
    instruction = (
        f'In the plot, shade the area under the series between '
        f'"{base["x_label"]}" = {a:.3g} and "{base["x_label"]}" = {b:.3g} '
        f'with the color {rgb_to_hex(shade_color)}.'
    )
    return input_state, answer_state, instruction


TASK_FNS = {
    "draw_segments":    task_draw_segments,
    "normalize_series": task_normalize_series,
    "filter_series":    task_filter_series,
    "shade_interval":   task_shade_interval,
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
