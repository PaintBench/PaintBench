"""Task 01 — Translation.

Parameters
----------
n    : density bucket
mode : "amount"          — translate a shape by a relative displacement
                          (1/4 horizontal only, 1/4 vertical only, 1/2 both)
       "align"           — translate a shape so that one of its named
                          control points aligns with a destination control
                          point: 1/4 vertical (match x, move horizontally),
                          1/4 horizontal (match y, move vertically),
                          1/2 exact (both axes)

Both modes:
  • 50 % place translated shape on top of existing shapes (overlay);
    50 % behind them (underlay).
"""
from __future__ import annotations
import random
import copy
from PIL import Image

from .base import (
    Problem, fill_params, N_MIN_OPTIONS, N_MAX_OPTIONS,
    _make_scene, _unique_desc, _sample_control_point, _sample_shape_control_point,
    _order_clause, _CLIP_CLAUSE,
    _scene_shapes_meta, _bg_colors_meta, _rgb_to_hex,
)
from core.colors import name_of
from core.shapes import ShapeInstance

NAME = "translation"
PARAMETERS = {
    "n_min": N_MIN_OPTIONS,
    "n_max": N_MAX_OPTIONS,
    "mode":  ["amount", "align"],
}


def _render_with_target(scene, target, new_target: "ShapeInstance",
                        overlay: bool, W: int, H: int) -> Image.Image:
    img    = scene.render_background()
    others = [s for s in scene.shapes if s is not target]
    if overlay:
        for s in others:
            s.draw(img)
        new_target.draw(img)
    else:
        new_target.draw(img)
        for s in others:
            s.draw(img)
    return img


def generate(seed: int, bg_spec, W: int, H: int,
             obj_colors: list, **kwargs) -> Problem:
    rng = random.Random(seed)
    p   = fill_params(kwargs, PARAMETERS, rng)
    n_min, n_max, mode = p["n_min"], p["n_max"], p["mode"]

    # ── Attribute RNG — seed-only, so axis choice is stable across n variants
    #    and modes for the same seed.
    #    amount "horizontal" ↔ align "vertical" (both move only horizontally)
    #    amount "vertical"   ↔ align "horizontal" (both move only vertically)
    #    amount "both"       ↔ align "exact"
    attr_rng    = random.Random(seed ^ 0x7A115)
    axis_choice = attr_rng.choices(["horizontal", "vertical", "both"], weights=[1, 1, 2])[0]

    scene     = _make_scene(bg_spec, W, H, obj_colors, rng, n_min, n_max)
    shapes    = scene.shapes
    input_img = scene.render()

    if not shapes:
        return Problem(input_img, "No shapes.", input_img.copy(), {"params": p})

    target = rng.choice(shapes)
    desc   = _unique_desc(target, shapes, rng)
    overlay = rng.random() < 0.5
    moved   = copy.copy(target)

    if mode == "amount":
        both_axes = axis_choice == "both"
        ax1, ay1, ax2, ay2 = target.axis_aligned_bbox()
        shape_area = max((ax2 - ax1) * (ay2 - ay1), 1.0)

        dx_frac = dy_frac = 0.0
        for _ in range(20):
            if axis_choice == "horizontal":
                dx_frac = rng.choice([-1, 1]) * rng.uniform(0.10, 0.30)
                dy_frac = 0.0
            elif axis_choice == "vertical":
                dx_frac = 0.0
                dy_frac = rng.choice([-1, 1]) * rng.uniform(0.10, 0.30)
            else:
                dx_frac = rng.choice([-1, 1]) * rng.uniform(0.10, 0.30)
                dy_frac = rng.choice([-1, 1]) * rng.uniform(0.10, 0.30)

            dx, dy = dx_frac * W, dy_frac * H
            nx1, ny1, nx2, ny2 = ax1 + dx, ay1 + dy, ax2 + dx, ay2 + dy
            overlap_w = max(0.0, min(nx2, W) - max(nx1, 0))
            overlap_h = max(0.0, min(ny2, H) - max(ny1, 0))
            if overlap_w * overlap_h >= 0.25 * shape_area:
                break

        moved.x += dx
        moved.y += dy

        parts = []
        if dx_frac:
            parts.append(f"{'right' if dx_frac > 0 else 'left'} by "
                         f"{abs(dx_frac)*100:.2f}% of the image width")
        if dy_frac:
            parts.append(f"{'down' if dy_frac > 0 else 'up'} by "
                         f"{abs(dy_frac)*100:.2f}% of the image height")
        move_desc = " and ".join(parts)

    else:  # align
        # Derived from shared axis_choice: horizontal→vertical, vertical→horizontal, both→exact
        align_type = {"horizontal": "vertical", "vertical": "horizontal",
                      "both": "exact"}[axis_choice]

        # Pick source and destination control points, resampling if the
        # resulting displacement is zero (can happen when axis filtering
        # zeroes out the only non-zero component).
        for _ in range(20):
            s_control_point_name, s_control_point_px = _sample_shape_control_point(target, rng)
            _dst_src, dst_control_point_name, dst_control_point_px = _sample_control_point(
                shapes, W, H, rng, exclude=(target, s_control_point_name)
            )
            full_dx = dst_control_point_px[0] - s_control_point_px[0]
            full_dy = dst_control_point_px[1] - s_control_point_px[1]
            dx = full_dx if align_type in ("exact", "vertical")  else 0.0
            dy = full_dy if align_type in ("exact", "horizontal") else 0.0
            if abs(dx) > 1.0 or abs(dy) > 1.0:
                break
        moved.x += dx
        moved.y += dy

        if _dst_src is None:
            dst_label = f"the {dst_control_point_name} of the image"
        elif _dst_src is target:
            dst_label = f"its {dst_control_point_name}"
        else:
            dst_label = (f"the {dst_control_point_name} of the "
                         f"{name_of(_dst_src.fill)} {_dst_src.shape_name}")

        if align_type == "exact":
            move_desc = f"so that its {s_control_point_name} aligns with {dst_label}"
        elif align_type == "vertical":
            move_desc = (f"horizontally so that its {s_control_point_name} aligns vertically "
                         f"with {dst_label}")
        else:
            move_desc = (f"vertically so that its {s_control_point_name} aligns horizontally "
                         f"with {dst_label}")

    answer_img  = _render_with_target(scene, target, moved, overlay, W, H)
    instruction = (f"Translate {desc} {move_desc}."
                   f"{_order_clause(overlay)}{_CLIP_CLAUSE}")

    meta = {
        "params": p,
        "bg_colors": _bg_colors_meta(bg_spec),
        "scene_shapes": _scene_shapes_meta(shapes),
        "target_shape": target.shape_name,
        "target_color": _rgb_to_hex(target.fill),
        "overlay": overlay,
    }
    if mode == "amount":
        meta.update({"both_axes": both_axes, "dx_px": dx, "dy_px": dy})
    else:
        meta.update({"src_control_point": s_control_point_name,
                     "dst_control_point": dst_control_point_name,
                     "align_type": align_type})
    return Problem(
        input_image=input_img,
        instruction=instruction,
        answer_image=answer_img,
        metadata=meta,
    )
