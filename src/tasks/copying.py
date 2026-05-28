"""Task 20 — Copy Region (shapes only).

Two square rectangles of identical size are outlined on the canvas.
The source rectangle contains shapes scaled from a full-canvas scene;
the destination is empty.  The model must copy the shapes from the source
rectangle into the destination rectangle at the same relative positions,
leaving the background unchanged.

Shapes are generated on the full canvas (same as all other tasks) and then
uniformly scaled into the source rectangle, so shape count capacity matches
other tasks.

Layout is horizontal (left/right) or vertical (upper/lower), chosen 50/50.
The fixed axis is pinned at the 25%/75% canvas positions; the free axis is
either shared (both rects at the same random position) or independently
randomized, each with 50% probability.

Parameters
----------
n_min / n_max : total number of shapes in the source rectangle
"""
from __future__ import annotations
import random
from PIL import Image, ImageDraw

from .base import (
    Problem, fill_params, N_MIN_OPTIONS, N_MAX_OPTIONS,
    _make_scene, _pick_unused_color, _shape_occupancy_mask,
    _scene_shapes_meta, _bg_colors_meta, _rgb_to_hex,
)
from core.shapes import ShapeInstance

NAME = "copying"

PARAMETERS = {
    "n_min": N_MIN_OPTIONS,
    "n_max": N_MAX_OPTIONS,
}

_RECT_FRAC = 0.44


def _draw_rect_outline(img: Image.Image,
                       x0: int, y0: int, size: int,
                       color: tuple) -> None:
    lw = max(3, img.width // 100)
    # Expand by lw so the outline wraps fully outside the shape generation area
    ImageDraw.Draw(img).rectangle(
        [x0 - lw, y0 - lw, x0 + size + lw - 1, y0 + size + lw - 1],
        outline=color, width=lw
    )


def generate(seed: int, bg_spec, W: int, H: int,
             obj_colors: list, **kwargs) -> Problem:
    rng = random.Random(seed)
    p   = fill_params(kwargs, PARAMETERS, rng)
    n_min, n_max = p["n_min"], p["n_max"]

    # Square rects — fixed size avoids shape distortion and keeps scale consistent
    rect_size = int(_RECT_FRAC * min(W, H))
    mg        = int(0.02 * min(W, H))

    # Rectangle placement: fixed axis pinned at 25%/75%, free axis randomized.
    layout  = rng.choice(["horizontal", "vertical"])
    aligned = rng.random() < 0.5

    def clamp(v: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, v))

    if layout == "horizontal":
        x1 = clamp(int(W * 0.25 - rect_size / 2), mg, W - rect_size - mg)
        x2 = clamp(int(W * 0.75 - rect_size / 2), mg, W - rect_size - mg)
        y_lo, y_hi = mg, H - rect_size - mg
        y_anchor = rng.randint(y_lo, y_hi)
        y1 = y_anchor if aligned else rng.randint(y_lo, y_hi)
        y2 = y_anchor if aligned else rng.randint(y_lo, y_hi)
        pos1, pos2 = (x1, y1), (x2, y2)
        name1, name2 = "left", "right"
    else:  # vertical
        y1 = clamp(int(H * 0.25 - rect_size / 2), mg, H - rect_size - mg)
        y2 = clamp(int(H * 0.75 - rect_size / 2), mg, H - rect_size - mg)
        x_lo, x_hi = mg, W - rect_size - mg
        x_anchor = rng.randint(x_lo, x_hi)
        x1 = x_anchor if aligned else rng.randint(x_lo, x_hi)
        x2 = x_anchor if aligned else rng.randint(x_lo, x_hi)
        pos1, pos2 = (x1, y1), (x2, y2)
        name1, name2 = "upper", "lower"

    if rng.random() < 0.5:
        (src_x, src_y), (dst_x, dst_y) = pos1, pos2
        src_name, dst_name = name1, name2
    else:
        (src_x, src_y), (dst_x, dst_y) = pos2, pos1
        src_name, dst_name = name2, name1

    # ── Attribute RNG — seed-only, so these are stable across n variants ────────
    attr_rng      = random.Random(seed ^ 0xC0F1A)
    outline_color = _pick_unused_color(list(bg_spec.colors), attr_rng,
                                       palette=list(obj_colors))

    # Reserve the outline color so no scene shape shares it.
    scene_colors = [c for c in obj_colors if tuple(c) != tuple(outline_color)]

    # Background
    bg_img = _make_scene(bg_spec, W, H, obj_colors, rng, 0, 0).render_background()

    # Generate shapes on a square canvas matching the rect proportions,
    # then uniformly scale into source rect — no distortion since both are square
    S = min(W, H)
    shape_scene = _make_scene(bg_spec, S, S, scene_colors, rng, n_min, n_max)
    scale = rect_size / S

    src_shapes = [
        ShapeInstance(s.shape_name,
                      src_x + s.x * scale,
                      src_y + s.y * scale,
                      s.w * scale,
                      s.h * scale,
                      s.rotation,
                      s.fill)
        for s in shape_scene.shapes
    ]

    # Base canvas: background + shapes in source rect
    canvas = bg_img.copy()
    for s in src_shapes:
        s.draw(canvas)

    # Input image: both rects outlined
    input_img = canvas.copy()
    _draw_rect_outline(input_img, src_x, src_y, rect_size, outline_color)
    _draw_rect_outline(input_img, dst_x, dst_y, rect_size, outline_color)

    # Answer image: paste shape pixels (not background) into destination
    shape_mask    = _shape_occupancy_mask(src_shapes, W, H)
    src_pixels    = canvas.crop((src_x, src_y,
                                 src_x + rect_size, src_y + rect_size))
    src_mask_crop = shape_mask.crop((src_x, src_y,
                                     src_x + rect_size, src_y + rect_size))

    answer_img = canvas.copy()
    answer_img.paste(src_pixels, (dst_x, dst_y), mask=src_mask_crop)
    _draw_rect_outline(answer_img, src_x, src_y, rect_size, outline_color)
    _draw_rect_outline(answer_img, dst_x, dst_y, rect_size, outline_color)

    instruction = (
        f"Copy the shapes (ignoring background) in the {src_name} outlined "
        f"square into the {dst_name} outlined square, "
        f"maintaining the exact shape positions."
    )

    return Problem(
        input_image=input_img,
        instruction=instruction,
        answer_image=answer_img,
        metadata={
            "params":        p,
            "bg_colors":     _bg_colors_meta(bg_spec),
            "scene_shapes":  _scene_shapes_meta(src_shapes),
            "src_rect":      [src_x / W, src_y / H,
                              (src_x + rect_size) / W,
                              (src_y + rect_size) / H],
            "dst_rect":      [dst_x / W, dst_y / H,
                              (dst_x + rect_size) / W,
                              (dst_y + rect_size) / H],
            "layout":        layout,
            "aligned":       aligned,
            "outline_color": _rgb_to_hex(outline_color),
            "scale":         round(scale, 4),
        },
    )
