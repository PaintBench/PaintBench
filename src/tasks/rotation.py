"""Task 03 — Rotation.

Parameters
----------
n    : density bucket
mode : "local"  — rotate the shape around one of its own control points
       "external"  — rotate the shape around an external pivot: 50 % a named
                  control point on another scene shape or the canvas,
                  50 % a random canvas coordinate

Both modes:
  • 50 % overlay; 50 % underlay.

Angle sampling (1/3 each):
  • [90, 180, 270]
  • Multiples of 15° (excluding 90, 180, 270)
  • 1° increments (non-multiples of 15)

Direction: 50 % clockwise, 50 % counterclockwise.
"""
from __future__ import annotations
import random
import copy
import math

from .base import (
    Problem, fill_params, N_MIN_OPTIONS, N_MAX_OPTIONS,
    _make_scene, _unique_desc, _fpos, _order_clause, _CLIP_CLAUSE,
    _sample_control_point, _sample_shape_control_point,
    _scene_shapes_meta, _bg_colors_meta, _rgb_to_hex,
)
from core.colors import name_of

NAME = "rotation"
PARAMETERS = {
    "n_min": N_MIN_OPTIONS,
    "n_max": N_MAX_OPTIONS,
    "mode":  ["local", "external"],
}

_ANGLES_90    = [90, 180, 270]
_ANGLES_15    = [a for a in range(1, 360) if a % 15 == 0 and a not in (90, 180, 270)]
_ANGLES_1     = [a for a in range(1, 360) if a % 15 != 0]


def _sample_angle(rng: random.Random) -> int:
    bucket = rng.randint(0, 2)
    if bucket == 0:
        return rng.choice(_ANGLES_90)
    elif bucket == 1:
        return rng.choice(_ANGLES_15)
    else:
        return rng.choice(_ANGLES_1)


def _rotate_shape_around(shape, cx_pivot: float, cy_pivot: float,
                          angle_deg: float):
    """Return a copy of *shape* rotated *angle_deg* CCW around (cx_pivot, cy_pivot)."""
    moved = copy.copy(shape)
    a  = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    old_cx, old_cy = shape.cx, shape.cy
    dx, dy = old_cx - cx_pivot, old_cy - cy_pivot
    new_cx = cx_pivot + dx * ca - dy * sa
    new_cy = cy_pivot + dx * sa + dy * ca
    moved.x = new_cx - shape.w / 2
    moved.y = new_cy - shape.h / 2
    moved.rotation = (shape.rotation + angle_deg) % 360
    return moved


def generate(seed: int, bg_spec, W: int, H: int,
             obj_colors: list, **kwargs) -> Problem:
    rng = random.Random(seed)
    p   = fill_params(kwargs, PARAMETERS, rng)
    n_min, n_max, mode = p["n_min"], p["n_max"], p["mode"]

    scene     = _make_scene(bg_spec, W, H, obj_colors, rng, n_min, n_max)
    shapes    = scene.shapes
    input_img = scene.render()

    if not shapes:
        return Problem(input_img, "No shapes.", input_img.copy(), {"params": p})

    # ── Attribute RNG — seed-only, so rotation angle and direction are stable
    #    across n variants for the same seed ──────────────────────────────────
    attr_rng = random.Random(seed ^ 0x4061E)
    angle    = _sample_angle(attr_rng)
    cw       = attr_rng.random() < 0.5

    target  = rng.choice(shapes)
    desc    = _unique_desc(target, shapes, rng)
    overlay = rng.random() < 0.5
    dir_str = "clockwise" if cw else "counterclockwise"
    # Internal rotation is CCW; negate angle for CW
    angle_ccw = -angle if cw else angle

    if mode == "local":
        control_point_name, control_point_px = _sample_shape_control_point(target, rng)
        pivot_x, pivot_y = control_point_px
        rotated    = _rotate_shape_around(target, pivot_x, pivot_y, angle_ccw)
        pivot_desc = f"its {control_point_name}"
    else:  # point — 50% named reference, 50% raw coordinate
        others = [s for s in shapes if s is not target]
        if rng.random() < 0.5:  # named reference (canvas always available as fallback)
            src, control_point_name, (pivot_x, pivot_y) = _sample_control_point(others, W, H, rng)
            rotated = _rotate_shape_around(target, pivot_x, pivot_y, angle_ccw)
            if src is None:
                pivot_desc = f"the {control_point_name} of the image"
            else:
                pivot_desc = (f"the {control_point_name} of the "
                              f"{name_of(src.fill)} {src.shape_name}")
        else:  # raw coordinate
            pivot_x = rng.uniform(0.05 * W, 0.95 * W)
            pivot_y = rng.uniform(0.05 * H, 0.95 * H)
            rotated = _rotate_shape_around(target, pivot_x, pivot_y, angle_ccw)
            pivot_desc = _fpos(pivot_x, pivot_y, W, H)

    # Render answer
    answer_img = scene.render_background()
    others = [s for s in shapes if s is not target]
    if overlay:
        for s in others:
            s.draw(answer_img)
        rotated.draw(answer_img)
    else:
        rotated.draw(answer_img)
        for s in others:
            s.draw(answer_img)

    instruction = (f"Rotate {desc} by {angle}° {dir_str} about {pivot_desc}."
                   f"{_order_clause(overlay)}{_CLIP_CLAUSE}")

    meta = {
        "params": p,
        "bg_colors": _bg_colors_meta(bg_spec),
        "scene_shapes": _scene_shapes_meta(shapes),
        "target_shape": target.shape_name,
        "target_color": _rgb_to_hex(target.fill),
        "angle": angle,
        "clockwise": cw,
        "overlay": overlay,
    }
    if mode == "local":
        meta["pivot_control_point"] = control_point_name
    else:
        meta["pivot_x_frac"] = pivot_x / W
        meta["pivot_y_frac"] = pivot_y / H
    return Problem(
        input_image=input_img,
        instruction=instruction,
        answer_image=answer_img,
        metadata=meta,
    )
