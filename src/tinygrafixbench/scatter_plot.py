import matplotlib.pyplot as plt
import numpy as np

from .utils import (
    apply_theme,
    make_rng,
    random_bg_and_text,
    random_color,
    random_magnitude,
    random_title,
    rgb_to_hex,
)


TASKS = ["draw_best_fit_line", "swap_axes", "remove_outlier", "recolor_class"]

FIG_W_PX, FIG_H_PX, FIG_DPI = 1024, 768, 160
POINT_SIZE = 60
FIT_LINEWIDTH = 2


def _too_close(p, existing, min_dist):
    x, y = p
    for ex, ey in existing:
        if (x - ex) ** 2 + (y - ey) ** 2 < min_dist ** 2:
            return True
    return False


def build_state(seed):
    """Build the base state for a seed. Deterministic.

    Two classes of points on a square (same x/y numerical scale):
    - class A has a line of best fit (same color as its points)
    - class B does not
    Points are rejection-sampled to enforce a minimum pairwise distance
    across both classes so none visually overlap.
    """
    rng = make_rng(seed)

    # Shared numerical range for both axes. `scale` varies the overall order
    # of magnitude across problems; everything derived (span, noise, fits)
    # tracks it automatically.
    scale = random_magnitude(rng)
    lo = float(rng.uniform(-50.0, 50.0)) * scale
    hi = lo + float(rng.uniform(60.0, 120.0)) * scale
    span = hi - lo
    margin = 0.08 * span
    min_dist = 0.05 * span  # pairwise minimum point separation (all classes)

    # Class A — has a line of best fit. Slope is bounded so the generating
    # line, plus noise and the outlier push, stays inside the square axis.
    n_a = int(rng.integers(9, 14))
    slope = float(rng.uniform(-0.5, 0.5))
    center = 0.5 * (lo + hi)
    intercept = center - slope * center + float(rng.uniform(-0.08, 0.08) * span)
    noise = 0.05 * span
    class_a = []
    for _ in range(n_a):
        for _attempt in range(400):
            x = float(rng.uniform(lo + margin, hi - margin))
            y = slope * x + intercept + float(rng.normal(0.0, noise))
            if not _too_close((x, y), class_a, min_dist):
                break
        class_a.append((x, y))
    # Force one clear outlier so remove_outlier is unambiguous. The push
    # direction is aligned with the point's natural residual so the push
    # never cancels noise and leaves another point as the true max-residual.
    # Clip to the axis so the outlier is never rendered off-canvas.
    outlier_idx = int(rng.integers(0, n_a))
    x_o, y_o = class_a[outlier_idx]
    r_natural = y_o - (slope * x_o + intercept)
    direction = 1.0 if r_natural >= 0.0 else -1.0
    y_o_pushed = y_o + direction * 3.5 * noise
    y_o_pushed = float(np.clip(y_o_pushed, lo + margin, hi - margin))
    class_a[outlier_idx] = (x_o, y_o_pushed)
    xs_a = np.array([p[0] for p in class_a])
    ys_a = np.array([p[1] for p in class_a])
    m_a, b_a = np.polyfit(xs_a, ys_a, 1)

    # Class B — no line of best fit, uniformly scattered, min-dist against
    # the full set of existing points.
    n_b = int(rng.integers(9, 14))
    class_b = []
    for _ in range(n_b):
        for _attempt in range(400):
            x = float(rng.uniform(lo + margin, hi - margin))
            y = float(rng.uniform(lo + margin, hi - margin))
            if not _too_close((x, y), class_a + class_b, min_dist):
                break
        class_b.append((x, y))

    bg, text = random_bg_and_text(rng)
    color_a = random_color(rng, avoid=[bg])
    color_b = random_color(rng, avoid=[color_a, bg])

    title = random_title(rng)

    return {
        "lim": (lo, hi),
        "class_a": class_a,
        "class_b": class_b,
        "class_a_color": color_a,
        "class_b_color": color_b,
        "class_a_has_fit": True,
        "class_b_has_fit": False,
        "class_a_fit": (float(m_a), float(b_a)),
        "class_b_fit": None,
        "bg": bg,
        "text": text,
        "title": title,
        "swapped": False,
    }


def _copy_state(s):
    return {
        "lim": s["lim"],
        "class_a": list(s["class_a"]),
        "class_b": list(s["class_b"]),
        "class_a_color": s["class_a_color"],
        "class_b_color": s["class_b_color"],
        "class_a_has_fit": s["class_a_has_fit"],
        "class_b_has_fit": s["class_b_has_fit"],
        "class_a_fit": s["class_a_fit"],
        "class_b_fit": s["class_b_fit"],
        "bg": s["bg"],
        "text": s["text"],
        "title": s["title"],
        "swapped": s["swapped"],
    }


def _xy(points, swapped):
    if swapped:
        return [p[1] for p in points], [p[0] for p in points]
    return [p[0] for p in points], [p[1] for p in points]


def _fit_endpoints(m, b, lo, hi):
    return [(lo, m * lo + b), (hi, m * hi + b)]


def render_state(state):
    lo, hi = state["lim"]
    swapped = state["swapped"]

    fig, ax = plt.subplots(
        figsize=(FIG_W_PX / FIG_DPI, FIG_H_PX / FIG_DPI),
        dpi=FIG_DPI,
        layout="constrained",
    )

    # Fit-class (A) rendered first so its points and line sit underneath the
    # no-fit class (B) — explicit zorder makes the ordering unambiguous.
    xa, ya = _xy(state["class_a"], swapped)
    ax.scatter(xa, ya, c=[state["class_a_color"]], s=POINT_SIZE, zorder=2)
    if state["class_a_has_fit"]:
        m, b = state["class_a_fit"]
        pts = _fit_endpoints(m, b, lo, hi)
        lx, ly = _xy(pts, swapped)
        ax.plot(lx, ly, color=state["class_a_color"],
                linewidth=FIT_LINEWIDTH, zorder=3)

    xb, yb = _xy(state["class_b"], swapped)
    ax.scatter(xb, yb, c=[state["class_b_color"]], s=POINT_SIZE, zorder=4)
    if state["class_b_has_fit"]:
        m, b = state["class_b_fit"]
        pts = _fit_endpoints(m, b, lo, hi)
        lx, ly = _xy(pts, swapped)
        ax.plot(lx, ly, color=state["class_b_color"],
                linewidth=FIT_LINEWIDTH, zorder=5)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(state["title"])
    apply_theme(fig, ax, state["bg"], state["text"])
    return fig


def task_draw_best_fit_line(base, rng):
    """Draw the line of best fit for class B (the class without a line)."""
    input_state = _copy_state(base)
    answer_state = _copy_state(base)
    xs = np.array([p[0] for p in base["class_b"]])
    ys = np.array([p[1] for p in base["class_b"]])
    m, b = np.polyfit(xs, ys, 1)
    answer_state["class_b_has_fit"] = True
    answer_state["class_b_fit"] = (float(m), float(b))
    instruction = (
        "Draw the line of best fit for the class of points without a line. "
        "Use the same color as those points and the same thickness as the "
        "existing line. Overlay the line on top of all existing elements."
    )
    return input_state, answer_state, instruction


def task_swap_axes(base, rng):
    """Swap x and y axes. Both axes share the same numerical scale,
    so tick labels do not change — only point placement."""
    input_state = _copy_state(base)
    answer_state = _copy_state(base)
    answer_state["swapped"] = not base["swapped"]
    instruction = (
        "Swap the x and y coordinates of every point and the line of best "
        "fit. Points in the class without the line of best fit should be "
        "overlaid on top."
    )
    return input_state, answer_state, instruction


def task_remove_outlier(base, rng):
    """Remove the point in class A that is furthest vertically from the line
    of best fit, leaving the line unchanged."""
    input_state = _copy_state(base)
    answer_state = _copy_state(base)
    m, b = base["class_a_fit"]
    residuals = [abs(y - (m * x + b)) for x, y in base["class_a"]]
    outlier_idx = int(np.argmax(residuals))
    answer_state["class_a"] = [
        p for i, p in enumerate(base["class_a"]) if i != outlier_idx
    ]
    # class_a_fit stays the same by design.
    instruction = (
        "In the class of points with the line of best fit, remove the point "
        "that is vertically furthest from the line. Keep the line in place."
    )
    return input_state, answer_state, instruction


def task_recolor_class(base, rng):
    """50/50 recolor class A or class B; recolor its line too if present."""
    which = "a" if rng.random() < 0.5 else "b"
    key_color = f"class_{which}_color"
    key_has_fit = f"class_{which}_has_fit"
    orig = base[key_color]
    other_key = "class_b_color" if which == "a" else "class_a_color"
    new_color = random_color(rng, avoid=[orig, base[other_key], base["bg"]])
    input_state = _copy_state(base)
    answer_state = _copy_state(base)
    answer_state[key_color] = new_color
    hex_new = rgb_to_hex(new_color)
    if base[key_has_fit]:
        instruction = (
            f"Recolor the line of best fit and its corresponding points to "
            f"{hex_new}."
        )
    else:
        instruction = (
            f"Recolor the points that are not represented by the line of best "
            f"fit to {hex_new}."
        )
    return input_state, answer_state, instruction


TASK_FNS = {
    "draw_best_fit_line": task_draw_best_fit_line,
    "swap_axes":          task_swap_axes,
    "remove_outlier":     task_remove_outlier,
    "recolor_class":      task_recolor_class,
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
