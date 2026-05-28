"""Task 13 — Comparison.

Several shapes of the same type and color are placed on the canvas with
varying sizes.  The model must remove a shape identified by its rank along
some metric, counted from either end ("smallest"/"largest", "2nd widest",
"3rd tallest", ...).  Rank 1 from either end is phrased as the superlative
alone, without an ordinal.

For area metric: shapes scale uniformly; rotation is either shared (all
shapes same angle) or independent (each randomized), chosen 50/50.
For width/height metrics: one dimension is fixed, the other varies; only
1D-scalable shapes are used; no rotation.

Parameters
----------
n_min, n_max : number of shapes to compare (must be >= 2)
"""
from __future__ import annotations
import random

from .base import (
    Problem, fill_params,
    _make_scene, _shape_label, _rgb_to_hex,
)
from core.background import make_background
from core.canvas import _bbox_overlap
from core.shapes import ShapeInstance, SHAPES, ALL_SHAPE_NAMES
from core.colors import name_of

NAME = "comparison"

_N_MIN_OPTIONS = [2, 3, 4, 5]
_N_MAX_OPTIONS = [4, 6, 7, 8]
_BG_RANGE      = (0, 5)   # background distractor shape count
_CMARGIN       = 0.05

PARAMETERS = {
    "n_min": _N_MIN_OPTIONS,
    "n_max": _N_MAX_OPTIONS,
}

_METRICS = ["area", "width", "height"]

_ADJ = {
    ("area",   "smallest"): "smallest",
    ("area",   "largest"):  "largest",
    ("width",  "smallest"): "narrowest",
    ("width",  "largest"):  "widest",
    ("height", "smallest"): "shortest",
    ("height", "largest"):  "tallest",
}


def _ordinal(n: int) -> str:
    """Return the English ordinal suffix form: 1 → '1st', 2 → '2nd', ..."""
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _metric_val(shape: ShapeInstance, metric: str) -> float:
    if metric == "area":
        return shape.w * shape.h  # actual area — rotation-independent
    ax1, ay1, ax2, ay2 = shape.axis_aligned_bbox()
    if metric == "width":
        return ax2 - ax1
    else:
        return ay2 - ay1


def generate(seed: int, bg_spec, W: int, H: int,
             obj_colors: list, **kwargs) -> Problem:
    rng = random.Random(seed)
    p   = fill_params(kwargs, PARAMETERS, rng)
    n_min, n_max = p["n_min"], p["n_max"]
    n_min, n_max = min(n_min, n_max), max(n_min, n_max)

    if n_max < 2:
        raise ValueError(f"n_max={n_max} is too small; comparison requires at least 2 shapes")

    sname     = rng.choice(ALL_SHAPE_NAMES)
    shape_def = SHAPES[sname]
    ar        = shape_def.ASPECT_RATIO

    # Width/height metrics require 1D-scalable shapes; area works for all.
    metrics = _METRICS if shape_def.SCALABLE_1D else ["area"]
    metric  = rng.choice(metrics)

    # Area metric: 50% all shapes share one rotation, 50% each rotated independently.
    # Width/height metrics: no rotation (bbox must reflect the intended axis).
    if metric == "area" and shape_def.ROTATABLE:
        same_rot = rng.random() < 0.5
        base_rot = rng.uniform(0, 360)
    else:
        same_rot = True
        base_rot = 0.0

    bg_set = {tuple(c) for c in bg_spec.colors}
    avail  = [c for c in obj_colors if tuple(c) not in bg_set] or list(obj_colors)
    color  = tuple(rng.choice(avail))
    n_comp = rng.randint(n_min, n_max)

    # Stratify the varying dimension into n_comp buckets so every rank is distinct.
    short    = min(W, H)
    size_lo  = max(40.0, short * 0.08)
    size_hi  = short * 0.24
    bucket_w = (size_hi - size_lo) / n_comp
    sizes    = [rng.uniform(size_lo + i * bucket_w, size_lo + (i + 1) * bucket_w)
                for i in range(n_comp)]
    rng.shuffle(sizes)

    # For width/height metrics, fix the perpendicular dimension to a single value.
    fixed_dim = rng.uniform(size_lo, size_hi) if metric != "area" else None

    comp_shapes: list[ShapeInstance] = []
    comp_bboxes: list[tuple] = []
    for size in sizes:
        if metric == "width":
            w, h = size, fixed_dim
        elif metric == "height":
            w, h = fixed_dim, size
        else:  # area — uniform scaling
            w, h = (size, size / ar) if ar >= 1.0 else (size * ar, size)
        rot    = base_rot if same_rot else rng.uniform(0, 360)
        placed = False
        for _ in range(100):
            x    = rng.uniform(W * _CMARGIN, W * (1 - _CMARGIN) - w)
            y    = rng.uniform(H * _CMARGIN, H * (1 - _CMARGIN) - h)
            inst = ShapeInstance(sname, x, y, w, h, rot, color)
            bbox = inst.axis_aligned_bbox()
            if not any(_bbox_overlap(bbox, b, 6.0) for b in comp_bboxes):
                comp_shapes.append(inst)
                comp_bboxes.append(bbox)
                placed = True
                break
        if not placed:
            break

    if len(comp_shapes) < n_comp:
        raise RuntimeError(
            f"Could not place {n_comp} {sname} shapes on a {W}x{H} canvas without overlap "
            f"(placed {len(comp_shapes)})"
        )

    bg_colors = [c for c in obj_colors if tuple(c) != color]
    bg_scene  = _make_scene(bg_spec, W, H, bg_colors or obj_colors, rng, *_BG_RANGE)
    bg_shapes = [
        s for s in bg_scene.shapes
        if s.shape_name != sname
        and not any(_bbox_overlap(s.axis_aligned_bbox(), cb, 6.0) for cb in comp_bboxes)
    ]

    metric_cache  = {id(s): _metric_val(s, metric) for s in comp_shapes}
    sorted_shapes = sorted(comp_shapes, key=lambda s: metric_cache[id(s)])

    # ── Attribute RNG — seed-only, so direction and description style are stable
    #    across n variants for the same seed ──────────────────────────────────
    attr_rng  = random.Random(seed ^ 0xC0401)
    direction = attr_rng.choice(["smallest", "largest"])
    _c        = attr_rng.randint(0, 2)

    col    = name_of(color)
    slabel = _shape_label(sname)
    rank      = rng.randint(1, len(sorted_shapes))  # 1 = superlative end
    idx       = rank - 1 if direction == "smallest" else len(sorted_shapes) - rank
    target    = sorted_shapes[idx]
    adj       = _ADJ[(metric, direction)]
    qualifier = adj if rank == 1 else f"{_ordinal(rank)} {adj}"

    shape_ref   = (f"{col} shape" if _c == 0 else
                   slabel         if _c == 1 else
                   f"{col} {slabel}")
    instruction = f"Remove the {qualifier} {shape_ref}."

    bg = make_background(W, H, bg_spec)

    input_img = bg.copy()
    for s in bg_shapes:
        s.draw(input_img)
    for s in comp_shapes:
        s.draw(input_img)

    answer_img = bg.copy()
    for s in bg_shapes:
        s.draw(answer_img)
    for s in comp_shapes:
        if s is not target:
            s.draw(answer_img)

    return Problem(
        input_image=input_img,
        instruction=instruction,
        answer_image=answer_img,
        metadata={
            "params": p,
            "bg_colors": [_rgb_to_hex(c) for c in bg_spec.colors],
            "shape": sname,
            "color": _rgb_to_hex(color),
            "n_shapes": len(comp_shapes),
            "metric": metric,
            "rotation_mode": "shared" if same_rot else "independent",
            "direction": direction,
            "rank": rank,
            "adj": adj,
            "qualifier": qualifier,
            "comparison_shape_sizes": sorted(metric_cache[id(s)] for s in comp_shapes),
        },
    )
