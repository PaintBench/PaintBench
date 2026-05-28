import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .utils import (
    FONT_SIZES,
    make_rng,
    random_bg_and_text,
    random_color,
    random_palette,
    random_string,
    random_title,
    rgb_to_hex,
    tighten_layout,
)


TASKS = ["add_node", "swap_nodes", "remove_node", "recolor_node"]

FIG_W_PX, FIG_H_PX, FIG_DPI = 1024, 768, 160
NODE_SIZE = 400


def _quoted_and(items):
    """Join items with double quotes, commas, and Oxford-comma "and" before the last."""
    quoted = [f'"{x}"' for x in items]
    if len(quoted) == 0:
        return ""
    if len(quoted) == 1:
        return quoted[0]
    if len(quoted) == 2:
        return f"{quoted[0]} and {quoted[1]}"
    return ", ".join(quoted[:-1]) + f", and {quoted[-1]}"


def build_state(seed):
    """Build the base state for a seed. Deterministic."""
    rng = make_rng(seed)
    n = int(rng.integers(5, 10))

    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    xs = np.cos(angles).tolist()
    ys = np.sin(angles).tolist()

    bg, text = random_bg_and_text(rng)
    # Nodes must stand out against the background.
    colors = random_palette(rng, n, avoid=[bg])
    labels = [random_string(rng) for _ in range(n)]

    p_edge = float(rng.uniform(0.25, 0.55))
    edges = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if rng.random() < p_edge
    ]

    title = random_title(rng)

    return {
        "n": n,
        "positions": list(zip(xs, ys)),
        "colors": colors,
        "labels": labels,
        "edges": edges,
        "bg": bg,
        "text": text,
        "title": title,
        "visible_nodes": list(range(n)),
        "key_nodes": list(range(n)),
    }


def _copy_state(s):
    return {
        "n": s["n"],
        "positions": list(s["positions"]),
        "colors": list(s["colors"]),
        "labels": list(s["labels"]),
        "edges": list(s["edges"]),
        "bg": s["bg"],
        "text": s["text"],
        "title": s["title"],
        "visible_nodes": list(s["visible_nodes"]),
        "key_nodes": list(s["key_nodes"]),
    }


def render_state(state):
    positions = state["positions"]
    colors = state["colors"]
    labels = state["labels"]
    bg = state["bg"]
    text = state["text"]
    visible = set(state["visible_nodes"])

    fig, ax = plt.subplots(
        figsize=(FIG_W_PX / FIG_DPI, FIG_H_PX / FIG_DPI),
        dpi=FIG_DPI,
        layout="constrained",
    )

    for i, j in state["edges"]:
        if i in visible and j in visible:
            xi, yi = positions[i]
            xj, yj = positions[j]
            ax.plot([xi, xj], [yi, yj], color=text, zorder=1)

    vis_list = [i for i in range(state["n"]) if i in visible]
    if vis_list:
        xs = [positions[i][0] for i in vis_list]
        ys = [positions[i][1] for i in vis_list]
        cs = [colors[i] for i in vis_list]
        ax.scatter(xs, ys, c=cs, s=NODE_SIZE, zorder=2, edgecolors=text)

    ax.set_title(state["title"], color=text, fontsize=FONT_SIZES["title"])
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")

    if state["key_nodes"]:
        handles = [Patch(color=colors[i], label=labels[i])
                   for i in state["key_nodes"]]
        legend = fig.legend(
            handles=handles,
            loc="center right",
            fontsize=FONT_SIZES["legend"],
        )
        legend.get_frame().set_facecolor(bg)
        legend.get_frame().set_edgecolor(text)
        for t in legend.get_texts():
            t.set_color(text)

    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    tighten_layout(fig)
    return fig


def _neighbors(state, k):
    out = set()
    for i, j in state["edges"]:
        if i == k:
            out.add(j)
        elif j == k:
            out.add(i)
    return sorted(out)


def task_add_node(base, rng):
    """Input: one node is missing from the graph (but still in the key).
    Answer: full graph."""
    k = int(rng.integers(0, base["n"]))
    input_state = _copy_state(base)
    input_state["visible_nodes"] = [i for i in input_state["visible_nodes"] if i != k]
    input_state["edges"] = [(a, b) for (a, b) in input_state["edges"]
                            if a != k and b != k]
    # key_nodes stays full

    answer_state = _copy_state(base)

    neighbors = _neighbors(base, k)
    neighbor_labels = [base["labels"][i] for i in neighbors]
    label = base["labels"][k]
    if neighbor_labels:
        instruction = (
            f'Add the node "{label}" so that all nodes are evenly spaced on a '
            f'circle. Connect it to {_quoted_and(neighbor_labels)}.'
        )
    else:
        instruction = (
            f'Add the node "{label}" so that all nodes are evenly spaced on a '
            f'circle. It should have no edges.'
        )
    return input_state, answer_state, instruction


def task_swap_nodes(base, rng):
    """Swap the positions of two nodes; their edges follow."""
    idx = rng.choice(base["n"], size=2, replace=False)
    i, j = int(idx[0]), int(idx[1])
    input_state = _copy_state(base)
    answer_state = _copy_state(base)
    answer_state["positions"][i], answer_state["positions"][j] = (
        base["positions"][j], base["positions"][i],
    )
    instruction = (
        f'Swap the positions of nodes "{base["labels"][i]}" and '
        f'"{base["labels"][j]}".'
    )
    return input_state, answer_state, instruction


def task_remove_node(base, rng):
    """Remove a node and its incident edges from the graph; keep the key."""
    k = int(rng.integers(0, base["n"]))
    input_state = _copy_state(base)
    answer_state = _copy_state(base)
    answer_state["visible_nodes"] = [i for i in answer_state["visible_nodes"] if i != k]
    answer_state["edges"] = [(a, b) for (a, b) in answer_state["edges"]
                             if a != k and b != k]
    # key_nodes unchanged
    instruction = (
        f'Remove node "{base["labels"][k]}" and its incident edges. Leave the '
        f'key unchanged.'
    )
    return input_state, answer_state, instruction


def task_recolor_node(base, rng):
    """Recolor a node in both the graph and the key."""
    k = int(rng.integers(0, base["n"]))
    new_color = random_color(rng, avoid=list(base["colors"]) + [base["bg"]])
    input_state = _copy_state(base)
    answer_state = _copy_state(base)
    answer_state["colors"][k] = new_color
    instruction = (
        f'Recolor node "{base["labels"][k]}" to {rgb_to_hex(new_color)}. '
        f'Update the color in both the graph and the key.'
    )
    return input_state, answer_state, instruction


TASK_FNS = {
    "add_node":     task_add_node,
    "swap_nodes":   task_swap_nodes,
    "remove_node":  task_remove_node,
    "recolor_node": task_recolor_node,
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
