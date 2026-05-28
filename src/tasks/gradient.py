"""Task 11 — Gradient (Parallelogram Region).

A parallelogram region is drawn with a specified outline color.  A linear
gradient is applied inside the region, blending from one colour to another.

Orientation:
  "horizontal" — parallelogram wider than tall;
                 side_to_side gradient flows top→bottom (across the short axis).
  "vertical"   — parallelogram taller than wide;
                 side_to_side gradient flows left→right (across the short axis).

Direction within orientation (randomised 50/50):
  "side_to_side" — gradient between the two parallel non-slanted edges.
  "corner"       — gradient diagonally from one corner to the opposite corner;
                   either top-left→bottom-right or top-right→bottom-left.

Parameters
----------
n    : density bucket
mode : "background" — apply gradient only to background pixels in the region
       "foreground" — apply gradient only to shape pixels in the region
       # "both"     — apply gradient to all pixels in the region (removed)
"""
from __future__ import annotations
import random
import numpy as np
from PIL import Image

from .base import (
    Problem, fill_params, N_MIN_OPTIONS, N_MAX_OPTIONS,
    _make_scene, _random_parallelogram, _para_mask, _draw_poly_outline,
    _desc_color, _pick_unused_color,
    _shape_occupancy_mask, _region_coverage,
    _scene_shapes_meta, _bg_colors_meta, _rgb_to_hex,
)
from core.colors import name_of

NAME = "gradient"
PARAMETERS = {
    "n_min": N_MIN_OPTIONS,
    "n_max": N_MAX_OPTIONS,
    "mode":  ["background", "foreground"],  # "both" removed: doesn't require fg/bg distinction
}


def _gradient_image(W: int, H: int,
                    c1: tuple, c2: tuple,
                    orientation: str, direction: str,
                    corner_dir: str,
                    para: list) -> Image.Image:
    """Create a full-canvas gradient image normalised to the parallelogram bbox.

    The gradient runs fully from c1 to c2 across the para's bounding box, so
    c1 appears at one edge/corner and c2 at the opposite edge/corner.
    Corner gradients use true linear projection so iso-lines are perpendicular
    to the c1→c2 diagonal for any aspect ratio.
    """
    xs = [pt[0] for pt in para]
    ys = [pt[1] for pt in para]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bw = max(1.0, x_max - x_min)
    bh = max(1.0, y_max - y_min)

    px = np.arange(W, dtype=np.float32)[None, :]  # (1, W)
    py = np.arange(H, dtype=np.float32)[:, None]  # (H, 1)

    if direction == "side_to_side":
        if orientation == "horizontal":
            t = np.clip((py - y_min) / bh, 0.0, 1.0)  # varies top→bottom
        else:
            t = np.clip((px - x_min) / bw, 0.0, 1.0)  # varies left→right
    else:  # corner — true linear projection onto c1→c2 diagonal
        corners = {
            "tl": (x_min, y_min), "tr": (x_max, y_min),
            "bl": (x_min, y_max), "br": (x_max, y_max),
        }
        a, b   = corner_dir.split("_")
        sx, sy = corners[a]
        ex, ey = corners[b]
        dx, dy = ex - sx, ey - sy
        L2     = max(1.0, dx * dx + dy * dy)
        t = np.clip(((px - sx) * dx + (py - sy) * dy) / L2, 0.0, 1.0)

    c1_arr = np.array(c1, dtype=np.float32)
    c2_arr = np.array(c2, dtype=np.float32)
    t3     = np.broadcast_to(t[..., None], (H, W, 3))
    result = (c1_arr * (1.0 - t3) + c2_arr * t3 + 0.5).astype(np.uint8)
    return Image.fromarray(result, "RGB")


def generate(seed: int, bg_spec, W: int, H: int,
             obj_colors: list, **kwargs) -> Problem:
    rng = random.Random(seed)
    p   = fill_params(kwargs, PARAMETERS, rng)
    n_min, n_max, mode = p["n_min"], p["n_max"], p["mode"]

    # Pick gradient colors before building the scene so the palette is never
    # exhausted when choosing c1/c2. Shapes are then drawn from the remainder,
    # guaranteeing outline, c1, and c2 are mutually distinct.
    pal           = list(obj_colors)
    used_bg       = list(bg_spec.colors)
    outline_color = _pick_unused_color(used_bg,                       rng, palette=pal)
    c1            = _pick_unused_color(used_bg + [outline_color],     rng, palette=pal)
    c2            = _pick_unused_color(used_bg + [outline_color, c1], rng, palette=pal)

    # ── Attribute RNG — seed-only, so gradient direction attributes are stable
    #    across n variants for the same seed ──────────────────────────────────
    attr_rng    = random.Random(seed ^ 0x64D1E)
    orientation = attr_rng.choice(["horizontal", "vertical"])
    direction   = attr_rng.choice(["side_to_side", "corner"])
    corner_dir  = attr_rng.choice(["tl_br", "tr_bl", "br_tl", "bl_tr"])

    shape_colors = [c for c in pal if c not in {outline_color, c1, c2}]
    scene    = _make_scene(bg_spec, W, H, shape_colors, rng, n_min, n_max)
    bg_img   = scene.render_background()
    full_img = scene.render()

    # Compute shape occupancy mask once; used for coverage check and foreground mode.
    shape_mask = _shape_occupancy_mask(scene.shapes, W, H)

    # Use a separate rng so the parallelogram is identical across modes for the same seed.
    para_rng = random.Random(seed ^ 0xF007BA)
    for _ in range(50):
        para = _random_parallelogram(para_rng, W, H, orientation)
        fg_frac, bg_frac = _region_coverage(_para_mask(W, H, para), shape_mask)
        if fg_frac >= 0.05 and bg_frac >= 0.10:
            break
    else:
        raise RuntimeError("gradient: could not find valid parallelogram region")

    input_img = full_img.copy()
    _draw_poly_outline(input_img, para, outline_color)

    para_mask_img = _para_mask(W, H, para)
    grad_img      = _gradient_image(W, H, c1, c2, orientation, direction, corner_dir, para)

    if mode == "background":
        # Render bg → flood para with gradient → draw shapes on top.
        # Shapes drawn last naturally cover their own AA edge pixels.
        answer_img = bg_img.copy()
        answer_img.paste(grad_img, mask=para_mask_img)
        for s in scene.shapes:
            s.draw(answer_img)

    elif mode == "foreground":
        # Composite grad over bg using shape occupancy as alpha → paste inside para.
        # AA edges blend gradient with bg, eliminating fringing.
        inside_para = bg_img.copy()
        inside_para.paste(grad_img, mask=shape_mask)
        answer_img  = full_img.copy()
        answer_img.paste(inside_para, mask=para_mask_img)

    # else:  # both — removed: doesn't require fg/bg distinction
    #     answer_img = full_img.copy()
    #     answer_img.paste(grad_img, mask=para_mask_img)

    # Redraw outline on answer
    _draw_poly_outline(answer_img, para, outline_color)

    c1_d         = _desc_color(c1, is_new=True)
    c2_d         = _desc_color(c2, is_new=True)
    outline_name = name_of(outline_color)

    if direction == "side_to_side":
        if orientation == "horizontal":
            start_anchor, end_anchor = "top edge", "bottom edge"
        else:
            start_anchor, end_anchor = "left edge", "right edge"
    else:
        start_anchor, end_anchor = {
            "tl_br": ("top-left corner",     "bottom-right corner"),
            "tr_bl": ("top-right corner",    "bottom-left corner"),
            "br_tl": ("bottom-right corner", "top-left corner"),
            "bl_tr": ("bottom-left corner",  "top-right corner"),
        }[corner_dir]

    # if mode == "both":  # removed
    #     recolor_keep = "Recolor all pixels inside the region; keep the outline as is."
    if mode == "background":
        recolor_keep = "Recolor only background pixels; keep non-background pixels and the outline as is."
    else:  # foreground
        recolor_keep = "Recolor only non-background pixels; keep background pixels and the outline as is."

    instruction = (
        f"Apply a linear RGB gradient from {c1_d} at the {start_anchor} "
        f"to {c2_d} at the {end_anchor} of the interior of the "
        f"{outline_name} outlined region. {recolor_keep}"
    )

    return Problem(
        input_image=input_img,
        instruction=instruction,
        answer_image=answer_img,
        metadata={
            "params": p,
            "bg_colors": _bg_colors_meta(bg_spec),
            "scene_shapes": _scene_shapes_meta(scene.shapes),
            "orientation": orientation,
            "direction": direction,
            "corner_dir": corner_dir,
            "outline_color": _rgb_to_hex(outline_color),
            "color1": _rgb_to_hex(c1),
            "color2": _rgb_to_hex(c2),
        },
    )
