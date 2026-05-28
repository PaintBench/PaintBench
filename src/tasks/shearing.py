"""Task 05 — Shearing.

Only one mode.  Shearing is applied relative to the shape's own bounding box.

Fixed-line conventions (50 % edge, 50 % center-line):
  Edge options:
    "top"      — horizontal shear; top edge stays, bottom shifts.
    "bottom"   — horizontal shear; bottom edge stays, top shifts.
    "left"     — vertical shear; left edge stays, right shifts.
    "right"    — vertical shear; right edge stays, left shifts.
  Center-line options:
    "center_h" — horizontal shear; center horizontal midline stays,
                 top and bottom edges displace in opposite directions.
    "center_v" — vertical shear; center vertical midline stays,
                 left and right edges displace in opposite directions.

The shear amount is expressed as a fraction of the bounding-box dimension
perpendicular to the shear direction (range 0.2–0.8).

The original shape is NOT preserved.  50 % overlay; 50 % underlay.
"""
from __future__ import annotations
import random

from .base import (
    Problem, fill_params, N_MIN_OPTIONS, N_MAX_OPTIONS,
    _make_scene, _unique_desc, _order_clause, _CLIP_CLAUSE,
    _scene_shapes_meta, _bg_colors_meta, _rgb_to_hex,
)
from core.shapes import _draw_polygon_aa, _draw_ring_aa

NAME = "shearing"
PARAMETERS = {
    "n_min": N_MIN_OPTIONS,
    "n_max": N_MAX_OPTIONS,
}

_EDGE_SIDES = ["top", "bottom", "left", "right"]


def _shear_pts(pts: list, fixed_side: str, k: float,
               x0: float, y0: float, x1: float, y1: float) -> list:
    """Apply shear to polygon points relative to bounding box (x0,y0)-(x1,y1).

    Positive k → right/down shift of the free edge (or bottom/right for center modes).
    """
    out = []
    for px, py in pts:
        if fixed_side == "top":
            frac = (py - y0) / max(1, y1 - y0)
            out.append((px + k * frac * (x1 - x0), py))
        elif fixed_side == "bottom":
            frac = (y1 - py) / max(1, y1 - y0)
            out.append((px + k * frac * (x1 - x0), py))
        elif fixed_side == "left":
            frac = (px - x0) / max(1, x1 - x0)
            out.append((px, py + k * frac * (y1 - y0)))
        elif fixed_side == "right":
            frac = (x1 - px) / max(1, x1 - x0)
            out.append((px, py + k * frac * (y1 - y0)))
        elif fixed_side == "center_h":
            # Horizontal center line fixed; frac in [-0.5, +0.5]
            y_center = (y0 + y1) / 2
            frac = (py - y_center) / max(1, y1 - y0)
            out.append((px + k * frac * (x1 - x0), py))
        else:  # center_v
            # Vertical center line fixed; frac in [-0.5, +0.5]
            x_center = (x0 + x1) / 2
            frac = (px - x_center) / max(1, x1 - x0)
            out.append((px, py + k * frac * (y1 - y0)))
    return out


class _ShearedInstance:
    """Lightweight wrapper storing a sheared polygon for drawing.

    If `inner_pts` is provided, the shape is rendered as an annulus (outer
    minus inner) — used for the ring, whose hole must survive the shear.
    """
    def __init__(self, pts, fill, inner_pts=None):
        self._pts = pts
        self._inner_pts = inner_pts
        self.fill = fill

    def draw(self, img):
        if self._inner_pts is None:
            _draw_polygon_aa(img, self._pts, self.fill)
        else:
            _draw_ring_aa(img, self._pts, self._inner_pts, self.fill)


def generate(seed: int, bg_spec, W: int, H: int,
             obj_colors: list, **kwargs) -> Problem:
    rng = random.Random(seed)
    p   = fill_params(kwargs, PARAMETERS, rng)
    n_min, n_max = p["n_min"], p["n_max"]

    scene     = _make_scene(bg_spec, W, H, obj_colors, rng, n_min, n_max)
    shapes    = scene.shapes
    input_img = scene.render()

    if not shapes:
        return Problem(input_img, "No shapes.", input_img.copy(), {"params": p})

    # ── Attribute RNG — seed-only, so shear attributes are stable across n variants
    attr_rng   = random.Random(seed ^ 0x5EAB)
    k          = round(attr_rng.uniform(0.20, 0.80) * 100) / 100 * attr_rng.choice([-1, 1])
    use_center = attr_rng.random() < 0.5
    fixed_side = attr_rng.choice(["center_h", "center_v"] if use_center else _EDGE_SIDES)

    target  = rng.choice(shapes)
    desc    = _unique_desc(target, shapes, rng)
    overlay = rng.random() < 0.5

    ax1, ay1, ax2, ay2 = target.axis_aligned_bbox()
    orig_pts    = target._canvas_pts()
    sheared_pts = _shear_pts(orig_pts, fixed_side, k, ax1, ay1, ax2, ay2)

    # Ring: shear the inner boundary too so the hole survives.
    if target.shape_name == "ring":
        orig_inner    = target.shape.inner_pts(orig_pts, target.cx, target.cy)
        sheared_inner = _shear_pts(orig_inner, fixed_side, k, ax1, ay1, ax2, ay2)
        sheared       = _ShearedInstance(sheared_pts, target.fill, sheared_inner)
    else:
        sheared       = _ShearedInstance(sheared_pts, target.fill)

    answer_img = scene.render_background()
    others = [s for s in shapes if s is not target]
    if overlay:
        for s in others:
            s.draw(answer_img)
        sheared.draw(answer_img)
    else:
        sheared.draw(answer_img)
        for s in others:
            s.draw(answer_img)

    # Build instruction
    pct = round(abs(k) * 100)

    if fixed_side == "center_h":
        half_pct = round(abs(k) * 100) // 2
        top_dir  = "left"  if k > 0 else "right"
        bot_dir  = "right" if k > 0 else "left"
        instruction_body = (
            f"Shear {desc} horizontally so its top bounding box edge shifts {top_dir} "
            f"and its bottom bounding box edge shifts {bot_dir}, each by {half_pct}% "
            f"of its bounding box width, keeping the horizontal center line fixed."
        )
    elif fixed_side == "center_v":
        half_pct  = round(abs(k) * 100) // 2
        left_dir  = "up"   if k > 0 else "down"
        right_dir = "down" if k > 0 else "up"
        instruction_body = (
            f"Shear {desc} vertically so its left bounding box edge shifts {left_dir} "
            f"and its right bounding box edge shifts {right_dir}, each by {half_pct}% "
            f"of its bounding box height, keeping the vertical center line fixed."
        )
    else:
        if fixed_side == "top":
            free_edge, dim, direction = "bottom", "width",  "right" if k > 0 else "left"
        elif fixed_side == "bottom":
            free_edge, dim, direction = "top",    "width",  "right" if k > 0 else "left"
        elif fixed_side == "left":
            free_edge, dim, direction = "right",  "height", "down"  if k > 0 else "up"
        else:  # right
            free_edge, dim, direction = "left",   "height", "down"  if k > 0 else "up"
        instruction_body = (
            f"Shear {desc} so its {free_edge} bounding box edge shifts {direction} "
            f"by {pct}% of its bounding box {dim}, keeping the {fixed_side} "
            f"bounding box edge fixed."
        )

    instruction = instruction_body + _order_clause(overlay) + _CLIP_CLAUSE

    return Problem(
        input_image=input_img,
        instruction=instruction,
        answer_image=answer_img,
        metadata={
            "params": p,
            "bg_colors": _bg_colors_meta(bg_spec),
            "scene_shapes": _scene_shapes_meta(shapes),
            "target_shape": target.shape_name,
            "target_color": _rgb_to_hex(target.fill),
            "fixed_side": fixed_side,
            "k": k,
            "overlay": overlay,
        },
    )
