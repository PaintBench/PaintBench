"""Task 10 — Flood Fill (Polygonal Region).

A convex polygon anchored around a scene shape is drawn on the input image
with a specified outline color.  The model must recolor the pixels inside
the polygon according to the target set, and preserve the polygon outline
in the answer.

Parameters
----------
n    : density bucket
mode : "background" — recolor only background pixels inside the polygon
       "foreground" — recolor only shape pixels inside the polygon
       # "both"     — recolor every pixel inside the polygon (removed)
"""
from __future__ import annotations
import random
from PIL import Image, ImageChops

from .base import (
    Problem, fill_params, N_MIN_OPTIONS, N_MAX_OPTIONS,
    _make_scene, _shape_polygon, _make_poly_mask, _draw_poly_outline,
    _desc_color, _pick_unused_color, _KEEP_OUTLINE_CLAUSE,
    _shape_occupancy_mask, _scene_shapes_meta, _bg_colors_meta, _rgb_to_hex,
)
from core.colors import name_of as _name_of

NAME = "flood_fill"
PARAMETERS = {
    "n_min": N_MIN_OPTIONS,
    "n_max": N_MAX_OPTIONS,
    "mode":  ["background", "foreground"],  # "both" removed: doesn't require fg/bg distinction
}


def generate(seed: int, bg_spec, W: int, H: int,
             obj_colors: list, **kwargs) -> Problem:
    rng = random.Random(seed)
    p   = fill_params(kwargs, PARAMETERS, rng)
    n_min, n_max, mode = p["n_min"], p["n_max"], p["mode"]

    # ── Attribute RNG — seed-only, so these are stable across n variants ────────
    attr_rng      = random.Random(seed ^ 0xF100D)
    pal           = list(obj_colors)
    bg_colors     = list(bg_spec.colors)
    outline_color = _pick_unused_color(bg_colors,                   attr_rng, palette=pal)
    new_color     = _pick_unused_color(bg_colors + [outline_color], attr_rng, palette=pal)

    # Reserve both special colors so no scene shape shares them.
    reserved     = {tuple(outline_color), tuple(new_color)}
    scene_colors = [c for c in obj_colors if tuple(c) not in reserved]
    scene    = _make_scene(bg_spec, W, H, scene_colors, rng, n_min, n_max)
    full_img = scene.render()

    # Separate rng so polygon is identical across modes for the same seed.
    poly_rng = random.Random(seed ^ 0xF007BA)
    poly = _shape_polygon(poly_rng, scene.shapes, W, H)

    input_img = full_img.copy()
    _draw_poly_outline(input_img, poly, outline_color)

    poly_mask  = _make_poly_mask(W, H, poly)
    new_layer  = Image.new("RGB", (W, H), new_color)

    if mode == "background":
        # Render bg → flood polygon with new_color → draw shapes on top.
        # Shapes drawn last naturally cover their own AA edge pixels.
        answer_img = scene.render_background()
        answer_img.paste(new_layer, mask=poly_mask)
        for s in scene.shapes:
            s.draw(answer_img)
        target_desc = "all background pixels"

    elif mode == "foreground":
        # Intersect poly_mask with shape occupancy to recolor only shape pixels.
        shape_mask = _shape_occupancy_mask(scene.shapes, W, H)
        combined   = ImageChops.multiply(poly_mask, shape_mask)
        answer_img = full_img.copy()
        answer_img.paste(new_layer, mask=combined)
        target_desc = "all non-background pixels"

    # else:  # both — removed: doesn't require fg/bg distinction
    #     answer_img = full_img.copy()
    #     answer_img.paste(new_layer, mask=poly_mask)
    #     target_desc = "all pixels"

    _draw_poly_outline(answer_img, poly, outline_color)

    outline_name = _name_of(outline_color)
    c_desc       = _desc_color(new_color, is_new=True)
    instruction  = (
        f"Recolor {target_desc} inside the {outline_name} outlined polygon "
        f"to {c_desc}.{_KEEP_OUTLINE_CLAUSE}"
    )

    return Problem(
        input_image=input_img,
        instruction=instruction,
        answer_image=answer_img,
        metadata={
            "params": p,
            "bg_colors": _bg_colors_meta(bg_spec),
            "scene_shapes": _scene_shapes_meta(scene.shapes),
            "outline_color": _rgb_to_hex(outline_color),
            "new_color": _rgb_to_hex(new_color),
        },
    )
