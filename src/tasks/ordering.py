"""Task 14 — Ordering.

Several copies of the same shape type and color are placed with a named
control point aligned along a horizontal or vertical line.  Background
shapes fill the rest of the canvas.

The model rearranges the shapes in increasing or decreasing order of size,
keeping each control point location in place.

Metric is coupled to axis direction (width↔horizontal, height↔vertical,
area↔either) so the varying dimension aligns naturally with the line.
Shape sizes are stratified into n buckets (3:1 ratio) so each rank is
visually distinct.

For area metric on rotatable shapes, all shapes share a single random
rotation.  The bbox center is used as the control point so it stays
on the line after rotation.  Width/height metrics: no rotation.

Parameters
----------
n_min, n_max : number of shapes to order (must be >= 2)
"""
from __future__ import annotations
import math
import random

from PIL import ImageDraw

from .base import (
    Problem, fill_params,
    _make_scene, _shape_plural, _rgb_to_hex,
)
from core.background import make_background
from core.canvas import _bbox_overlap
from core.shapes import ShapeInstance, SHAPES, ALL_SHAPE_NAMES
from core.colors import name_of

NAME = "ordering"

_N_MIN_OPTIONS = [2, 3, 4, 5]
_N_MAX_OPTIONS = [4, 6, 7, 8]
_BG_RANGE      = (0, 5)
_MIN_SHAPE_PX  = 32.0
_GAP           = 0.78   # shape_size / slot_size ratio (controls spacing between shapes)

PARAMETERS = {
    "n_min": _N_MIN_OPTIONS,
    "n_max": _N_MAX_OPTIONS,
}


def _place_at_control_point(sname, rx, ry, ax, ay, w, h, color, rot=0.0):
    return ShapeInstance(sname, ax - rx * w, ay - ry * h, w, h, rot, color)


def _in_band(bbox, axis, band_lo, band_hi, pad=0.0):
    """Return True if bbox overlaps the line band (measured perpendicular to axis)."""
    bx1, by1, bx2, by2 = bbox
    if axis == "horizontal":
        return not (by2 + pad < band_lo or by1 - pad > band_hi)
    else:
        return not (bx2 + pad < band_lo or bx1 - pad > band_hi)


def generate(seed: int, bg_spec, W: int, H: int,
             obj_colors: list, **kwargs) -> Problem:
    rng = random.Random(seed)
    p   = fill_params(kwargs, PARAMETERS, rng)
    n_min, n_max = p["n_min"], p["n_max"]
    n_min, n_max = min(n_min, n_max), max(n_min, n_max)

    if n_max < 2:
        raise ValueError(f"n_max={n_max} is too small; ordering requires at least 2 shapes")

    sname     = rng.choice(ALL_SHAPE_NAMES)
    shape_def = SHAPES[sname]
    ar        = shape_def.ASPECT_RATIO

    axis = rng.choice(["horizontal", "vertical"])

    # Couple metric to axis: width varies along a horizontal line,
    # height varies along a vertical line; area works with either.
    metrics = ["area"] + (["width" if axis == "horizontal" else "height"]
                          if shape_def.SCALABLE_1D else [])
    metric  = rng.choice(metrics)

    # Area metric on rotatable shapes: all shapes share one random rotation.
    # The bbox center is used as the control point so it stays on the line.
    # Width/height metrics: no rotation (bbox axis must reflect the intended dimension).
    base_rot = rng.uniform(0, 360) if (metric == "area" and shape_def.ROTATABLE) else 0.0

    # Trig for rotation-adjusted layout.
    cos_a = abs(math.cos(math.radians(base_rot)))
    sin_a = abs(math.sin(math.radians(base_rot)))

    # Exclude bg colors from shape color (same as comparison).
    bg_set = {tuple(c) for c in bg_spec.colors}
    avail  = [c for c in obj_colors if tuple(c) not in bg_set] or list(obj_colors)
    color  = tuple(rng.choice(avail))

    # Pick a box outline color: from palette, not used by shape or background.
    box_avail = [c for c in obj_colors if tuple(c) != color and tuple(c) not in bg_set]
    if not box_avail:
        box_avail = [c for c in obj_colors if tuple(c) != color] or list(obj_colors)
    box_color = tuple(rng.choice(box_avail))

    n_shapes = rng.randint(n_min, n_max)
    order    = rng.choice(["increasing", "decreasing"])

    control_point_name = "bounding box center"
    rx, ry = 0.5, 0.5

    cm = 0.05 * min(W, H)

    # Compute size_cap: max value of the varying dimension, adjusted so the
    # largest *rotated* shape fills exactly one slot (prevents off-canvas placement).
    if axis == "horizontal":
        max_slot = _GAP * (W - 2 * cm) / (n_shapes - 1 + _GAP)
        if metric == "width":
            size_cap = max_slot
        else:  # area — rotated width = size * rot_fac_w must equal max_slot
            rot_fac  = (cos_a + sin_a / ar) if ar >= 1.0 else (ar * cos_a + sin_a)
            size_cap = max_slot / rot_fac
    else:  # vertical
        max_slot = _GAP * (H - 2 * cm) / (n_shapes - 1 + _GAP)
        if metric == "height":
            size_cap = max_slot
        else:  # area — rotated height = size * rot_fac_h must equal max_slot
            rot_fac  = (sin_a + cos_a / ar) if ar >= 1.0 else (ar * sin_a + cos_a)
            size_cap = max_slot / rot_fac

    if size_cap < _MIN_SHAPE_PX:
        raise RuntimeError(
            f"{n_shapes} shapes cannot fit on a {W}x{H} canvas "
            f"(size_cap={size_cap:.1f}px < {_MIN_SHAPE_PX}px minimum)"
        )

    # Stratify size range into n_shapes buckets (3:1 ratio, same as comparison).
    size_lo = max(_MIN_SHAPE_PX, size_cap / 3.0)
    size_hi = size_cap

    # For width/height metrics, fix the perpendicular dimension (same as comparison).
    fixed_dim = rng.uniform(size_lo, size_hi) if metric != "area" else None

    def wh(size):
        if metric == "width":
            return size, fixed_dim
        elif metric == "height":
            return fixed_dim, size
        elif ar >= 1.0:
            return size, size / ar
        else:
            return size * ar, size

    bucket = (size_hi - size_lo) / n_shapes
    sizes  = [rng.uniform(size_lo + i * bucket, size_lo + (i + 1) * bucket)
              for i in range(n_shapes)]

    # Shuffle until input is not already in the target order.
    target_order = sorted(sizes, reverse=(order == "decreasing"))
    for _ in range(100):
        rng.shuffle(sizes)
        if sizes != target_order:
            break
    else:
        raise RuntimeError("Could not produce a disordered arrangement (degenerate size list)")

    # Evenly-spaced control point positions along the line axis.
    # Use the rotated bounding box of the largest shape for margins so no
    # shape can fall outside the canvas regardless of how sizes are ordered.
    max_w_u, max_h_u = wh(size_cap)
    max_w = max_w_u * cos_a + max_h_u * sin_a   # rotated bbox width
    max_h = max_w_u * sin_a + max_h_u * cos_a   # rotated bbox height

    if axis == "horizontal":
        control_point_lo = cm + rx * max_w
        control_point_hi = W - cm - (1 - rx) * max_w
        spacing          = (control_point_hi - control_point_lo) / max(1, n_shapes - 1)
        ln_lo            = cm + ry * max_h
        ln_hi            = H - cm - (1 - ry) * max_h
        line_pos         = rng.uniform(ln_lo, max(ln_lo + 1, ln_hi))
        control_point_positions = [(control_point_lo + i * spacing, line_pos) for i in range(n_shapes)]
    else:
        control_point_lo = cm + ry * max_h
        control_point_hi = H - cm - (1 - ry) * max_h
        spacing          = (control_point_hi - control_point_lo) / max(1, n_shapes - 1)
        ln_lo            = cm + rx * max_w
        ln_hi            = W - cm - (1 - rx) * max_w
        line_pos         = rng.uniform(ln_lo, max(ln_lo + 1, ln_hi))
        control_point_positions = [(line_pos, control_point_lo + i * spacing) for i in range(n_shapes)]

    rotations = [base_rot] * n_shapes

    input_shapes  = [_place_at_control_point(sname, rx, ry, ax, ay, *wh(s), color, rot)
                     for s, (ax, ay), rot in zip(sizes, control_point_positions, rotations)]
    sorted_sizes  = sorted(sizes, reverse=(order == "decreasing"))
    answer_shapes = [_place_at_control_point(sname, rx, ry, ax, ay, *wh(s), color, rot)
                     for s, (ax, ay), rot in zip(sorted_sizes, control_point_positions, rotations)]

    # ── Attribute RNG — seed-only, so description style is stable across n variants
    attr_rng = random.Random(seed ^ 0x04DEB)
    _c       = attr_rng.randint(0, 2)

    col        = name_of(color)
    splural    = _shape_plural(sname)
    descriptor = (f"{col} shapes" if _c == 0 else
                  splural          if _c == 1 else
                  f"{col} {splural}")

    dir_desc   = "left-to-right" if axis == "horizontal" else "top-to-bottom"
    order_desc = "size" if metric == "area" else metric

    instruction = (
        f"Rearrange the {descriptor} {dir_desc} in {order} order of {order_desc}"
        f", keeping each shape in the same position inside its box."
    )

    # Background: exclude shapes of the same type, shapes overlapping any
    # ordering position, and shapes in the line band (perpendicular strip).
    all_bboxes = [s.axis_aligned_bbox() for s in input_shapes + answer_shapes]
    if axis == "horizontal":
        band_lo = line_pos - ry * max_h
        band_hi = line_pos + (1 - ry) * max_h
    else:
        band_lo = line_pos - rx * max_w
        band_hi = line_pos + (1 - rx) * max_w

    bg_colors = [c for c in obj_colors if tuple(c) != color]
    bg_scene  = _make_scene(bg_spec, W, H, bg_colors or obj_colors, rng, *_BG_RANGE)
    bg_shapes = [
        s for s in bg_scene.shapes
        if s.shape_name != sname
        and not any(_bbox_overlap(s.axis_aligned_bbox(), bb, 4.0) for bb in all_bboxes)
        and not _in_band(s.axis_aligned_bbox(), axis, band_lo, band_hi, pad=4.0)
    ]

    bg = make_background(W, H, bg_spec)

    # Draw outline boxes at each control point position.
    # Box size is uniform: large enough to contain the largest rotated shape
    # plus a small padding.
    box_pad  = 4.0
    box_half_w = max_w / 2 + box_pad
    box_half_h = max_h / 2 + box_pad
    box_lw     = max(2, int(round(min(W, H) * 0.005)))

    def _draw_boxes(img):
        draw = ImageDraw.Draw(img)
        for (cx, cy) in control_point_positions:
            x0, y0 = cx - box_half_w, cy - box_half_h
            x1, y1 = cx + box_half_w, cy + box_half_h
            draw.rectangle([x0, y0, x1, y1], outline=box_color, width=box_lw)

    input_img = bg.copy()
    for s in bg_shapes:
        s.draw(input_img)
    _draw_boxes(input_img)
    for s in input_shapes:
        s.draw(input_img)

    answer_img = bg.copy()
    for s in bg_shapes:
        s.draw(answer_img)
    _draw_boxes(answer_img)
    for s in answer_shapes:
        s.draw(answer_img)

    return Problem(
        input_image=input_img,
        instruction=instruction,
        answer_image=answer_img,
        metadata={
            "params":        p,
            "bg_colors":     [_rgb_to_hex(c) for c in bg_spec.colors],
            "shape":         sname,
            "color":         _rgb_to_hex(color),
            "box_color":     _rgb_to_hex(box_color),
            "n_shapes":      n_shapes,
            "axis":          axis,
            "metric":        metric,
            "rotation":      base_rot,
            "order":         order,
            "control_point": control_point_name,
            "input_sizes":   sizes,
            "sorted_sizes":  sorted_sizes,
        },
    )
