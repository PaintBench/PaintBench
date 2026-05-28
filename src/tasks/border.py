"""Task 19 — Shape Border (Proximity Ring).

Pixels outside a target shape within a given Euclidean distance of its edge
are colored a new color, without recoloring the shape itself.

Parameters
----------
n : density bucket

Distance D: sampled uniformly from [2%, 4%] of image width, expressed as a
percentage of image width in the instruction.

Implementation: exact Euclidean morphological dilation via numpy — correct
for all shape types including non-convex shapes.
"""
from __future__ import annotations
import math
import random
import numpy as np
from PIL import Image

from .base import (
    Problem, fill_params, N_MIN_OPTIONS, N_MAX_OPTIONS,
    _make_scene, _unique_desc, _shape_occupancy_mask,
    _desc_color, _pick_unused_color,
    _scene_shapes_meta, _bg_colors_meta, _rgb_to_hex,
)

NAME = "border"
PARAMETERS = {
    "n_min": N_MIN_OPTIONS,
    "n_max": N_MAX_OPTIONS,
}

_DIST_MIN_FRAC = 0.02
_DIST_MAX_FRAC = 0.04


# ---------------------------------------------------------------------------
# Dilation helper
# ---------------------------------------------------------------------------

def _dilation_mask(shape_mask_img: Image.Image, radius_px: float) -> Image.Image:
    """Return an L-mode mask with 255 for every pixel within Euclidean distance
    radius_px of any shape pixel. Exact for all shape types.
    """
    arr = np.array(shape_mask_img) > 128
    if not arr.any():
        return Image.new("L", shape_mask_img.size, 0)

    H, W   = arr.shape
    r      = int(math.ceil(radius_px))
    result = arr.copy()

    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > radius_px * radius_px:
                continue
            # Mark every pixel displaced (dy, dx) away from a shape pixel
            src_y0, src_y1 = max(0, -dy), min(H, H - dy)
            src_x0, src_x1 = max(0, -dx), min(W, W - dx)
            dst_y0, dst_y1 = max(0,  dy), min(H, H + dy)
            dst_x0, dst_x1 = max(0,  dx), min(W, W + dx)
            result[dst_y0:dst_y1, dst_x0:dst_x1] |= arr[src_y0:src_y1, src_x0:src_x1]

    return Image.fromarray(result.astype(np.uint8) * 255)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate(seed: int, bg_spec, W: int, H: int,
             obj_colors: list, **kwargs) -> Problem:
    rng = random.Random(seed)
    p   = fill_params(kwargs, PARAMETERS, rng)
    n_min, n_max = p["n_min"], p["n_max"]

    # ── Attribute RNG — seed-only, so these are stable across n variants ────────
    attr_rng  = random.Random(seed ^ 0xB04D3)
    d_frac    = attr_rng.uniform(_DIST_MIN_FRAC, _DIST_MAX_FRAC)
    new_color = _pick_unused_color(list(bg_spec.colors), attr_rng, palette=list(obj_colors))

    # Reserve the border color so no scene shape shares it.
    scene_colors = [c for c in obj_colors if tuple(c) != tuple(new_color)]
    scene  = _make_scene(bg_spec, W, H, scene_colors, rng, n_min, n_max)
    shapes = scene.shapes

    if not shapes:
        img = scene.render()
        return Problem(img, "No shapes.", img.copy(), {"params": p})

    target = rng.choice(shapes)
    desc   = _unique_desc(target, shapes, rng)
    d_px   = d_frac * W
    d_pct  = d_frac * 100
    c_desc    = _desc_color(new_color, is_new=True)

    scene_img  = scene.render()
    shape_mask = _shape_occupancy_mask([target], W, H)
    dilated    = _dilation_mask(shape_mask, d_px)

    # Ring covers pixels within the dilation but outside the target itself.
    # It is pasted on top of the scene so it overlays neighboring shapes.
    dil_arr    = np.array(dilated) > 128
    target_arr = np.array(shape_mask) > 128
    ring_arr   = (dil_arr & ~target_arr).astype(np.uint8) * 255
    colored_mask = Image.fromarray(ring_arr)

    instruction = (
        f"Color all pixels within a Euclidean distance of at most {d_pct:.1f}% image "
        f"width from any pixel in {desc} to {c_desc}, without recoloring the shape itself."
    )

    color_layer = Image.new("RGB", (W, H), new_color)
    answer_img  = scene_img.copy()
    answer_img.paste(color_layer, mask=colored_mask)

    return Problem(
        input_image=scene_img,
        instruction=instruction,
        answer_image=answer_img,
        metadata={
            "params":        p,
            "bg_colors":     _bg_colors_meta(bg_spec),
            "scene_shapes":  _scene_shapes_meta(shapes),
            "target_shape":  target.shape_name,
            "target_color":  _rgb_to_hex(target.fill),
            "distance_frac": round(d_frac, 4),
            "new_color":     _rgb_to_hex(new_color),
        },
    )
