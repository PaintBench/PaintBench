"""Task 12 — Alpha Blending (Polygonal Region).

A random convex polygon is marked on the input with a specified outline
color.  The model must blend a specified solid colour at a specified
transparency over all pixels inside the region, and preserve the outline.

Parameters
----------
n    : density bucket
Transparency is a random integer 1–99 (percentage opacity of the overlay).
"""
from __future__ import annotations
import random
from PIL import Image

from .base import (
    Problem, fill_params, N_MIN_OPTIONS, N_MAX_OPTIONS,
    _make_scene, _shape_polygon, _make_poly_mask, _draw_poly_outline,
    _desc_color, _pick_unused_color, _KEEP_OUTLINE_CLAUSE,
    _scene_shapes_meta, _bg_colors_meta, _rgb_to_hex,
)
from core.colors import name_of as _name_of

NAME = "blending"
PARAMETERS = {
    "n_min": N_MIN_OPTIONS,
    "n_max": N_MAX_OPTIONS,
}


def generate(seed: int, bg_spec, W: int, H: int,
             obj_colors: list, **kwargs) -> Problem:
    rng = random.Random(seed)
    p   = fill_params(kwargs, PARAMETERS, rng)
    n_min, n_max = p["n_min"], p["n_max"]

    # ── Attribute RNG — seed-only, so these are stable across n variants ────────
    attr_rng      = random.Random(seed ^ 0xB1E4D)
    pal           = list(obj_colors)
    bg_colors     = list(bg_spec.colors)
    outline_color = _pick_unused_color(bg_colors,                   attr_rng, palette=pal)
    blend_rgb     = _pick_unused_color(bg_colors + [outline_color], attr_rng, palette=pal)
    opacity       = attr_rng.randint(10, 90)

    # Reserve both special colors so no scene shape shares them.
    reserved     = {tuple(outline_color), tuple(blend_rgb)}
    scene_colors = [c for c in obj_colors if tuple(c) not in reserved]
    scene    = _make_scene(bg_spec, W, H, scene_colors, rng, n_min, n_max)
    full_img = scene.render()

    poly_rng = random.Random(seed ^ 0xF007BA)
    poly = _shape_polygon(poly_rng, scene.shapes, W, H)

    input_img = full_img.copy()
    _draw_poly_outline(input_img, poly, outline_color)

    poly_mask = _make_poly_mask(W, H, poly)
    alpha     = round(opacity / 100 * 255)

    alpha_mask = poly_mask.point(lambda v: int(v * alpha / 255))
    blended    = full_img.copy()
    blended.paste(Image.new("RGB", (W, H), blend_rgb), mask=alpha_mask)

    # Redraw outline on answer
    _draw_poly_outline(blended, poly, outline_color)

    outline_name = _name_of(outline_color)
    c_desc       = _desc_color(blend_rgb, is_new=True)
    instruction  = (
        f"Blend the color {c_desc} at {opacity}% opacity over all pixels inside the "
        f"{outline_name} outlined polygon.{_KEEP_OUTLINE_CLAUSE}"
    )

    return Problem(
        input_image=input_img,
        instruction=instruction,
        answer_image=blended,
        metadata={
            "params": p,
            "bg_colors": _bg_colors_meta(bg_spec),
            "scene_shapes": _scene_shapes_meta(scene.shapes),
            "opacity": opacity,
            "outline_color": _rgb_to_hex(outline_color),
            "blend_color": _rgb_to_hex(blend_rgb),
        },
    )
