"""Task 18 — Pixel Formula (Polygonal Region).

Same setup as flood fill: a convex polygon is outlined on the input image.
Instead of recoloring to a flat target color, a pixel-level formula is applied
to every pixel inside the polygon.

Parameters
----------
n    : density bucket
mode : "invert"     — (R, G, B) → (255-R, 255-G, 255-B)
       "grayscale"  — convert to luminance grayscale:
                      L = 0.299R + 0.587G + 0.114B; all channels set to L
       "brightness" — add a signed offset k to each channel, clamped to [0, 255]:
                      (R, G, B) → (clamp(R+k, 0, 255), clamp(G+k, 0, 255), clamp(B+k, 0, 255))
                      k is chosen uniformly from [-80, -30] ∪ [+30, +80]

The polygon outline is preserved in the answer image.
"""
from __future__ import annotations
import random
from PIL import Image, ImageChops

from .base import (
    Problem, fill_params, N_MIN_OPTIONS, N_MAX_OPTIONS,
    _make_scene, _shape_polygon, _make_poly_mask, _draw_poly_outline,
    _pick_unused_color, _KEEP_OUTLINE_CLAUSE,
    _scene_shapes_meta, _bg_colors_meta, _rgb_to_hex,
)
from core.colors import name_of

NAME = "point_operations"
PARAMETERS = {
    "n_min": N_MIN_OPTIONS,
    "n_max": N_MAX_OPTIONS,
    "mode":  ["invert", "grayscale", "brightness"],
}

_BRIGHTNESS_RANGE = (30, 80)


def _apply_formula(img: Image.Image, mode: str, brightness_offset: int = 0) -> Image.Image:
    """Apply formula to the entire image; caller masks to the polygon region."""
    if mode == "invert":
        return ImageChops.invert(img)
    elif mode == "grayscale":
        return img.convert("L").convert("RGB")
    else:  # brightness
        r, g, b = img.split()
        def _shift(ch):
            lut = [max(0, min(255, v + brightness_offset)) for v in range(256)]
            return ch.point(lut)
        return Image.merge("RGB", (_shift(r), _shift(g), _shift(b)))


def generate(seed: int, bg_spec, W: int, H: int,
             obj_colors: list, **kwargs) -> Problem:
    rng = random.Random(seed)
    p   = fill_params(kwargs, PARAMETERS, rng)
    n_min, n_max, mode = p["n_min"], p["n_max"], p["mode"]

    # ── Attribute RNG — seed-only, so these are stable across n variants ────────
    attr_rng      = random.Random(seed ^ 0xA771B)
    outline_color = _pick_unused_color(list(bg_spec.colors), attr_rng,
                                       palette=list(obj_colors))
    brightness_offset = 0
    if mode == "brightness":
        lo, hi = _BRIGHTNESS_RANGE
        magnitude = attr_rng.randint(lo, hi)
        brightness_offset = magnitude if attr_rng.random() < 0.5 else -magnitude

    # Reserve the outline color so no scene shape shares it.
    scene_colors = [c for c in obj_colors if tuple(c) != tuple(outline_color)]
    scene    = _make_scene(bg_spec, W, H, scene_colors, rng, n_min, n_max)
    full_img = scene.render()

    # Use a separate rng so the polygon is stable across modes for the same seed
    poly_rng = random.Random(seed ^ 0xF007BA)
    poly     = _shape_polygon(poly_rng, scene.shapes, W, H)

    outline_name = name_of(outline_color)

    input_img = full_img.copy()
    _draw_poly_outline(input_img, poly, outline_color)

    poly_mask = _make_poly_mask(W, H, poly)

    transformed = _apply_formula(full_img, mode, brightness_offset)
    answer_img  = full_img.copy()
    answer_img.paste(transformed, mask=poly_mask)
    _draw_poly_outline(answer_img, poly, outline_color)

    if mode == "grayscale":
        instruction = (
            f"Convert all pixels inside the {outline_name} outlined polygon "
            f"to grayscale (luminance = 0.299R + 0.587G + 0.114B).{_KEEP_OUTLINE_CLAUSE}"
        )
    elif mode == "invert":
        instruction = (
            f"Invert the colors of all pixels inside the {outline_name} outlined "
            f"polygon.{_KEEP_OUTLINE_CLAUSE}"
        )
    else:  # brightness
        magnitude = abs(brightness_offset)
        if brightness_offset >= 0:
            op_desc = f"Add {magnitude} to each RGB channel of"
        else:
            op_desc = f"Subtract {magnitude} from each RGB channel of"
        instruction = (
            f"{op_desc} all pixels inside the {outline_name} outlined polygon, "
            f"clamping each channel to [0, 255].{_KEEP_OUTLINE_CLAUSE}"
        )

    meta = {
        "params": p,
        "bg_colors": _bg_colors_meta(bg_spec),
        "scene_shapes": _scene_shapes_meta(scene.shapes),
        "outline_color": _rgb_to_hex(outline_color),
    }
    if mode == "brightness":
        meta["brightness_offset"] = brightness_offset

    return Problem(
        input_image=input_img,
        instruction=instruction,
        answer_image=answer_img,
        metadata=meta,
    )
