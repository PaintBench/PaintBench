"""Task 04 — Scaling.

Parameters
----------
n    : density bucket
mode : "amount" — scale by an explicit factor (50 % 1D, 50 % uniform 2D);
                  anchor = center or fixed edge/corner (50/50)
       "match"  — scale to match a bounding box dimension of another shape:
                  25 % width only  (match reference bbox width)
                  25 % height only (match reference bbox height)
                  25 % uniform scale so height matches reference bbox height
                  25 % uniform scale so width  matches reference bbox width

Scale factor for "amount": log-uniformly sampled from [40%, 250%],
rounded to nearest 1%.
"""
from __future__ import annotations
import math
import random
import copy

from .base import (
    Problem, fill_params, N_MIN_OPTIONS, N_MAX_OPTIONS,
    _make_scene, _unique_desc, _order_clause, _CLIP_CLAUSE,
    _scene_shapes_meta, _bg_colors_meta, _rgb_to_hex,
)

NAME = "scaling"
PARAMETERS = {
    "n_min": N_MIN_OPTIONS,
    "n_max": N_MAX_OPTIONS,
    "mode":  ["amount", "match"],
}

_LOG_MIN = math.log(0.40)
_LOG_MAX = math.log(2.50)


def _sample_factor(rng: random.Random) -> float:
    """Log-uniform in [0.40, 2.50], rounded to nearest 1%."""
    raw = math.exp(rng.uniform(_LOG_MIN, _LOG_MAX))
    return round(raw * 100) / 100


def _apply_1d_anchor(scaled, target, axis: str, rng: random.Random) -> tuple:
    """Apply anchor for a 1D scale using axis-aligned bbox edges.
    Returns (anchor_desc, anchor_side).

    Strategy: place scaled at the same center as target, read its actual bbox,
    then shift x (or y) so the chosen bbox edge stays fixed.  Exact for all shapes.
    """
    use_center = rng.random() < 0.5
    anchor_side = None
    ax1, ay1, ax2, ay2 = target.axis_aligned_bbox()

    if axis == "width":
        scaled.y = target.y  # keeps cy unchanged
        if use_center:
            scaled.x    = target.cx - scaled.w / 2
            anchor_desc = "bounding box center"
        else:
            anchor_side = rng.choice(["left", "right"])
            scaled.x = target.cx - scaled.w / 2          # preliminary: same cx
            s_ax1, _, s_ax2, _ = scaled.axis_aligned_bbox()
            if anchor_side == "left":
                scaled.x   += ax1 - s_ax1
                anchor_desc = "left bounding box edge"
            else:
                scaled.x   += ax2 - s_ax2
                anchor_desc = "right bounding box edge"
    else:  # height
        scaled.x = target.x  # keeps cx unchanged
        if use_center:
            scaled.y    = target.cy - scaled.h / 2
            anchor_desc = "bounding box center"
        else:
            anchor_side = rng.choice(["top", "bottom"])
            scaled.y = target.cy - scaled.h / 2          # preliminary: same cy
            _, s_ay1, _, s_ay2 = scaled.axis_aligned_bbox()
            if anchor_side == "top":
                scaled.y   += ay1 - s_ay1
                anchor_desc = "top bounding box edge"
            else:
                scaled.y   += ay2 - s_ay2
                anchor_desc = "bottom bounding box edge"

    return anchor_desc, anchor_side


def _apply_2d_anchor(scaled, target, rng: random.Random) -> tuple:
    """Apply anchor for a uniform 2D scale using axis-aligned bbox corners.
    Returns (anchor_desc, anchor_corner).

    Strategy: place scaled at the same center as target, read its actual bbox,
    then shift x and y so the chosen bbox corner stays fixed.  Exact for all shapes.
    """
    use_center = rng.random() < 0.5
    anchor_corner = None
    ax1, ay1, ax2, ay2 = target.axis_aligned_bbox()

    # Start with same center for both center and corner cases.
    scaled.x = target.cx - scaled.w / 2
    scaled.y = target.cy - scaled.h / 2

    if use_center:
        anchor_desc = "bounding box center"
    else:
        anchor_corner = rng.choice(["top-left", "top-right",
                                    "bottom-left", "bottom-right"])
        s_ax1, s_ay1, s_ax2, s_ay2 = scaled.axis_aligned_bbox()
        if anchor_corner == "top-left":
            scaled.x += ax1 - s_ax1;  scaled.y += ay1 - s_ay1
        elif anchor_corner == "top-right":
            scaled.x += ax2 - s_ax2;  scaled.y += ay1 - s_ay1
        elif anchor_corner == "bottom-left":
            scaled.x += ax1 - s_ax1;  scaled.y += ay2 - s_ay2
        else:  # bottom-right
            scaled.x += ax2 - s_ax2;  scaled.y += ay2 - s_ay2
        anchor_desc = f"{anchor_corner} bounding box corner"

    return anchor_desc, anchor_corner


def generate(seed: int, bg_spec, W: int, H: int,
             obj_colors: list, **kwargs) -> Problem:
    rng = random.Random(seed)
    p   = fill_params(kwargs, PARAMETERS, rng)
    n_min, n_max, mode = p["n_min"], p["n_max"], p["mode"]

    # ── Attribute RNG — seed-only, so scale attributes are stable across n variants
    attr_rng   = random.Random(seed ^ 0x5CA1E)
    factor     = _sample_factor(attr_rng)
    use_1d     = attr_rng.random() < 0.5
    axis       = attr_rng.choice(["width", "height"])
    match_type = attr_rng.choice(["width_only", "height_only",
                                   "uniform_by_height", "uniform_by_width"])

    scene     = _make_scene(bg_spec, W, H, obj_colors, rng, n_min, n_max)
    shapes    = scene.shapes
    input_img = scene.render()

    if not shapes:
        return Problem(input_img, "No shapes.", input_img.copy(), {"params": p})

    target  = rng.choice(shapes)
    desc    = _unique_desc(target, shapes, rng)
    overlay = rng.random() < 0.5
    scaled  = copy.copy(target)

    meta = {
        "params": p,
        "bg_colors": _bg_colors_meta(bg_spec),
        "scene_shapes": _scene_shapes_meta(shapes),
        "target_shape": target.shape_name,
        "target_color": _rgb_to_hex(target.fill),
        "overlay": overlay,
    }

    if mode == "amount":
        pct              = round(factor * 100)
        effective_factor = factor

        if use_1d:
            direction  = "horizontally" if axis == "width" else "vertically"
            if axis == "width":
                scaled.w = target.w * factor
            else:
                scaled.h = target.h * factor
            anchor_desc, anchor_side = _apply_1d_anchor(scaled, target, axis, rng)
            scale_desc = (f"Scale {desc} {direction} so its bounding box {axis} is {pct}% "
                          f"of its current {axis}, keeping its {anchor_desc} fixed.")
            meta.update({"factor": factor, "axis": axis, "direction": direction, "anchor": anchor_desc})
            if anchor_side:
                meta["anchor_side"] = anchor_side
        else:
            scaled.w = target.w * factor
            scaled.h = target.h * factor
            anchor_desc, anchor_corner = _apply_2d_anchor(scaled, target, rng)
            scale_desc = (f"Scale {desc} uniformly so its bounding box is {pct}% of its "
                          f"current size, keeping its {anchor_desc} fixed.")
            meta.update({"factor": factor, "anchor": anchor_desc})
            if anchor_corner:
                meta["anchor_corner"] = anchor_corner

    else:  # match
        others = [s for s in shapes if s is not target]
        if not others:
            # Fall back to amount mode if no other shape available
            return generate(seed, bg_spec, W, H, obj_colors,
                            **{**kwargs, "mode": "amount"})

        ref          = rng.choice(others)
        ref_ax1, ref_ay1, ref_ax2, ref_ay2 = ref.axis_aligned_bbox()
        ref_w        = ref_ax2 - ref_ax1
        ref_h        = ref_ay2 - ref_ay1
        ref_desc     = _unique_desc(ref, shapes, rng)

        if match_type == "width_only":
            t_ax1, t_ay1, t_ax2, t_ay2 = target.axis_aligned_bbox()
            target_bbox_w = t_ax2 - t_ax1
            effective_factor = ref_w / target_bbox_w if target_bbox_w > 0 else 1.0
            scaled.w = ref_w
            anchor_desc, anchor_side = _apply_1d_anchor(scaled, target, "width", rng)
            scale_desc = (f"Scale {desc} horizontally so its bounding box width matches "
                          f"the bounding box width of {ref_desc}, keeping its {anchor_desc} fixed.")
            meta.update({"match_type": match_type, "ref_shape": ref_desc,
                         "anchor": anchor_desc})
            if anchor_side:
                meta["anchor_side"] = anchor_side

        elif match_type == "height_only":
            t_ax1, t_ay1, t_ax2, t_ay2 = target.axis_aligned_bbox()
            target_bbox_h = t_ay2 - t_ay1
            effective_factor = ref_h / target_bbox_h if target_bbox_h > 0 else 1.0
            scaled.h = ref_h
            anchor_desc, anchor_side = _apply_1d_anchor(scaled, target, "height", rng)
            scale_desc = (f"Scale {desc} vertically so its bounding box height matches "
                          f"the bounding box height of {ref_desc}, keeping its {anchor_desc} fixed.")
            meta.update({"match_type": match_type, "ref_shape": ref_desc,
                         "anchor": anchor_desc})
            if anchor_side:
                meta["anchor_side"] = anchor_side

        elif match_type == "uniform_by_height":
            t_ax1, t_ay1, t_ax2, t_ay2 = target.axis_aligned_bbox()
            target_bbox_h = t_ay2 - t_ay1
            factor   = ref_h / target_bbox_h if target_bbox_h > 0 else 1.0
            effective_factor = factor
            scaled.w = target.w * factor
            scaled.h = target.h * factor
            anchor_desc, anchor_corner = _apply_2d_anchor(scaled, target, rng)
            scale_desc = (f"Scale {desc} uniformly so its bounding box height matches "
                          f"the bounding box height of {ref_desc}, keeping its {anchor_desc} fixed.")
            meta.update({"match_type": match_type, "ref_shape": ref_desc,
                         "factor": factor, "anchor": anchor_desc})
            if anchor_corner:
                meta["anchor_corner"] = anchor_corner

        else:  # uniform_by_width
            t_ax1, t_ay1, t_ax2, t_ay2 = target.axis_aligned_bbox()
            target_bbox_w = t_ax2 - t_ax1
            factor   = ref_w / target_bbox_w if target_bbox_w > 0 else 1.0
            effective_factor = factor
            scaled.w = target.w * factor
            scaled.h = target.h * factor
            anchor_desc, anchor_corner = _apply_2d_anchor(scaled, target, rng)
            scale_desc = (f"Scale {desc} uniformly so its bounding box width matches "
                          f"the bounding box width of {ref_desc}, keeping its {anchor_desc} fixed.")
            meta.update({"match_type": match_type, "ref_shape": ref_desc,
                         "factor": factor, "anchor": anchor_desc})
            if anchor_corner:
                meta["anchor_corner"] = anchor_corner

    # Render answer
    answer_img = scene.render_background()
    others_render = [s for s in shapes if s is not target]
    if overlay:
        for s in others_render:
            s.draw(answer_img)
        scaled.draw(answer_img)
    else:
        scaled.draw(answer_img)
        for s in others_render:
            s.draw(answer_img)

    if effective_factor >= 1.0:
        suffix = f"{_order_clause(overlay)}{_CLIP_CLAUSE}"
    else:
        suffix = ""
    instruction = f"{scale_desc}{suffix}"
    return Problem(
        input_image=input_img,
        instruction=instruction,
        answer_image=answer_img,
        metadata=meta,
    )
