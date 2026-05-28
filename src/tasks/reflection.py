"""Task 02 — Reflection.

Parameters
----------
n    : density bucket
mode : "local"     — reflect across one of 8 axes of the shape's own
                     bounding box: 4 edges, 2 center lines, 2 diagonals
       "external"  — reflect across a line defined by two external points:
                     50 % two named control points from scene shapes/canvas,
                     50 % two random canvas coordinates

All modes: 50 % overlay; 50 % underlay.

Reflection is implemented at the pixel level using PIL's AFFINE transform,
so all shapes (including asymmetric and non-square ones) are reflected
correctly regardless of their geometry or aspect ratio.
"""
from __future__ import annotations
import copy
import math
import random

from PIL import Image

from .base import (
    Problem, fill_params, N_MIN_OPTIONS, N_MAX_OPTIONS,
    _make_scene, _unique_desc, _fpos, _order_clause, _CLIP_CLAUSE,
    _sample_control_point,
    _scene_shapes_meta, _bg_colors_meta, _rgb_to_hex,
)
from core.colors import name_of

NAME = "reflection"
PARAMETERS = {
    "n_min": N_MIN_OPTIONS,
    "n_max": N_MAX_OPTIONS,
    "mode":  ["local", "external"],
}


# ---------------------------------------------------------------------------
# Pixel-level reflection helpers
# ---------------------------------------------------------------------------

def _affine_reflect_params(ax: float, ay: float,
                            bx: float, by: float) -> tuple:
    """Return PIL AFFINE data for reflection across the line (ax,ay)→(bx,by).

    Reflection is its own inverse, so the same matrix maps output→input
    and input→output.
    """
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    cos2 = (dx * dx - dy * dy) / L2
    sin2 = 2 * dx * dy / L2
    c = ax * (1 - cos2) - ay * sin2
    f = ay * (1 + cos2) - ax * sin2
    return (cos2, sin2, c,
            sin2, -cos2, f)


def _reflected_layer(target, W: int, H: int,
                     ax: float, ay: float,
                     bx: float, by: float) -> tuple:
    """Render target reflected across the line (ax,ay)→(bx,by).

    Returns (rgb_img, mask_img):
      rgb_img  — RGB image of the reflected shape on a black background
      mask_img — L-mode mask: 255 at shape pixels, 0 at background
    """
    params = _affine_reflect_params(ax, ay, bx, by)

    # Render shape with its real colour on a black background
    shape_img = Image.new("RGB", (W, H), (0, 0, 0))
    target.draw(shape_img)

    # Render white silhouette for the mask (independent of fill colour)
    white = copy.copy(target)
    white.fill = (255, 255, 255)
    mask_rgb = Image.new("RGB", (W, H), (0, 0, 0))
    white.draw(mask_rgb)
    mask_img = mask_rgb.split()[0]   # L-mode: 255 at shape, 0 at background

    # Apply pixel-level reflection to both
    rgb  = shape_img.transform((W, H), Image.AFFINE, params, Image.BILINEAR)
    mask = mask_img.transform((W, H), Image.AFFINE, params, Image.BILINEAR)
    return rgb, mask


# ---------------------------------------------------------------------------
# Instruction helpers
# ---------------------------------------------------------------------------

def _control_point_label(src, control_point_name: str) -> str:
    """Human-readable label for a control point."""
    if src is None:
        return f"the {control_point_name} of the image"
    return f"the {control_point_name} of the {name_of(src.fill)} {src.shape_name}"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate(seed: int, bg_spec, W: int, H: int,
             obj_colors: list, **kwargs) -> Problem:
    rng = random.Random(seed)
    p   = fill_params(kwargs, PARAMETERS, rng)
    n_min, n_max, mode = p["n_min"], p["n_max"], p["mode"]

    # ── Attribute RNG — seed-only, so reflection axis is stable across n variants
    attr_rng  = random.Random(seed ^ 0xEF1EC)
    bbox_axis = attr_rng.choice([
        "top edge", "bottom edge", "left edge", "right edge",
        "horizontal center line", "vertical center line",
        "top-left to bottom-right diagonal", "top-right to bottom-left diagonal",
    ])

    scene     = _make_scene(bg_spec, W, H, obj_colors, rng, n_min, n_max)
    shapes    = scene.shapes
    input_img = scene.render()

    if not shapes:
        return Problem(input_img, "No shapes.", input_img.copy(), {"params": p})

    target  = rng.choice(shapes)
    desc    = _unique_desc(target, shapes, rng)
    overlay = rng.random() < 0.5

    if mode == "local":
        ax1, ay1, ax2, ay2 = target.axis_aligned_bbox()
        cx, cy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
        if bbox_axis == "top edge":
            ax, ay, bx, by = ax1, ay1, ax2, ay1
        elif bbox_axis == "bottom edge":
            ax, ay, bx, by = ax1, ay2, ax2, ay2
        elif bbox_axis == "left edge":
            ax, ay, bx, by = ax1, ay1, ax1, ay2
        elif bbox_axis == "right edge":
            ax, ay, bx, by = ax2, ay1, ax2, ay2
        elif bbox_axis == "horizontal center line":
            ax, ay, bx, by = ax1, cy, ax2, cy
        elif bbox_axis == "vertical center line":
            ax, ay, bx, by = cx, ay1, cx, ay2
        elif bbox_axis == "top-left to bottom-right diagonal":
            ax, ay, bx, by = ax1, ay1, ax2, ay2
        else:  # top-right to bottom-left diagonal
            ax, ay, bx, by = ax2, ay1, ax1, ay2
        axis_desc = f"the {bbox_axis} of its bounding box"

    else:  # external — 50% named control points, 50% raw coordinates
        bbox_axis = None
        if rng.random() < 0.5:  # named control points
            src1, control_point_1_name, (p1x, p1y) = _sample_control_point(shapes, W, H, rng)
            for _ in range(20):
                src2, control_point_2_name, (p2x, p2y) = _sample_control_point(
                    shapes, W, H, rng, exclude=(src1, control_point_1_name)
                )
                if math.hypot(p2x - p1x, p2y - p1y) >= 0.05 * min(W, H):
                    break
            ax, ay, bx, by = p1x, p1y, p2x, p2y
            axis_desc = (f"the line that passes through {_control_point_label(src1, control_point_1_name)} "
                         f"and {_control_point_label(src2, control_point_2_name)}")
        else:  # raw coordinates
            for _ in range(100):
                p1x = rng.uniform(0.1 * W, 0.9 * W)
                p1y = rng.uniform(0.1 * H, 0.9 * H)
                p2x = rng.uniform(0.1 * W, 0.9 * W)
                p2y = rng.uniform(0.1 * H, 0.9 * H)
                if math.hypot(p2x - p1x, p2y - p1y) >= 0.15 * min(W, H):
                    break
            ax, ay, bx, by = p1x, p1y, p2x, p2y
            axis_desc = (
                f"the line that passes through {_fpos(ax, ay, W, H)} and "
                f"{_fpos(bx, by, W, H)}"
            )

    # Guard: degenerate line (shouldn't happen, but be safe)
    if math.hypot(bx - ax, by - ay) < 1e-6:
        bx += 1.0

    # Pixel-level reflection
    refl_rgb, refl_mask = _reflected_layer(target, W, H, ax, ay, bx, by)

    answer_img = scene.render_background()
    others = [s for s in shapes if s is not target]
    if overlay:
        for s in others:
            s.draw(answer_img)
        answer_img.paste(refl_rgb, mask=refl_mask)
    else:
        answer_img.paste(refl_rgb, mask=refl_mask)
        for s in others:
            s.draw(answer_img)

    instruction = (f"Reflect {desc} across {axis_desc}."
                   f"{_order_clause(overlay)}{_CLIP_CLAUSE}")

    meta = {
        "params": p,
        "bg_colors": _bg_colors_meta(bg_spec),
        "scene_shapes": _scene_shapes_meta(shapes),
        "target_shape": target.shape_name,
        "target_color": _rgb_to_hex(target.fill),
        "overlay": overlay,
    }
    if mode == "local":
        meta["bbox_axis"] = bbox_axis
    else:
        meta["line_p1_frac"] = [ax / W, ay / H]
        meta["line_p2_frac"] = [bx / W, by / H]
    return Problem(
        input_image=input_img,
        instruction=instruction,
        answer_image=answer_img,
        metadata=meta,
    )
