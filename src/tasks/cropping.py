r"""Task 10 — Cropping (Outlined Region).

An outlined rectangle — matching the canvas aspect ratio — is drawn on the
input image to mark the crop region.  The model must crop to that region and
scale the result to fill the canvas using nearest-neighbor interpolation.
Because the crop region shares the canvas aspect ratio the scale is always
uniform (no distortion).

Parameters
----------
n    : density bucket
mode : "straight" — axis-aligned rectangle; crop and scale up to fill canvas
       "tilted"   — rectangle rotated by a random integer angle in [-44, 45] \ {0};
                    deskew then scale up to fill the canvas
"""
from __future__ import annotations
import math
import random
from PIL import Image, ImageDraw

from .base import (
    Problem, fill_params, N_MIN_OPTIONS, N_MAX_OPTIONS,
    _make_scene, _pick_unused_color,
    _scene_shapes_meta, _bg_colors_meta, _rgb_to_hex,
)

NAME = "cropping"
PARAMETERS = {
    "n_min": N_MIN_OPTIONS,
    "n_max": N_MAX_OPTIONS,
    "mode":  ["straight", "tilted"],
}

_SIZE_MIN_FRAC = 0.25
_SIZE_MAX_FRAC = 0.75
_MARGIN        = 0.04

# Integer angles in [-44, 45] excluding 0
_VALID_ANGLES = list(range(-44, 0)) + list(range(1, 46))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _rect_corners(cx: float, cy: float, half_w: float, half_h: float,
                  angle_deg: float) -> list[tuple]:
    """Corners of the outlined rectangle in the original (un-rotated) image.

    Returns TL, TR, BR, BL of the axis-aligned crop box after PIL rotates
    the image CCW by angle_deg around (cx, cy).
    """
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    return [
        (cx - half_w * cos_a + half_h * sin_a, cy - half_w * sin_a - half_h * cos_a),  # TL
        (cx + half_w * cos_a + half_h * sin_a, cy + half_w * sin_a - half_h * cos_a),  # TR
        (cx + half_w * cos_a - half_h * sin_a, cy + half_w * sin_a + half_h * cos_a),  # BR
        (cx - half_w * cos_a - half_h * sin_a, cy - half_w * sin_a + half_h * cos_a),  # BL
    ]


def _draw_region_outline(img: Image.Image, corners: list, color: tuple) -> None:
    lw  = max(3, img.width // 100)
    pts = [(int(round(x)), int(round(y))) for x, y in corners]
    ImageDraw.Draw(img).polygon(pts, outline=color, width=lw)


def _box_contains_shape(shapes, x0: int, y0: int, w: int, h: int,
                         angle: float = 0.0,
                         cx: float = 0.0, cy: float = 0.0) -> bool:
    """Return True if any shape's center falls inside the crop box.

    For tilted mode the shape center is first transformed into the
    deskewed (rotated) coordinate space before checking.
    """
    cos_a, sin_a = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    x1, y1 = x0, y0
    x2, y2 = x0 + w, y0 + h
    for s in shapes:
        scx, scy = s.cx, s.cy
        rx = cx + (scx - cx) * cos_a + (scy - cy) * sin_a
        ry = cy - (scx - cx) * sin_a + (scy - cy) * cos_a
        if x1 <= rx <= x2 and y1 <= ry <= y2:
            return True
    return False


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate(seed: int, bg_spec, W: int, H: int,
             obj_colors: list, **kwargs) -> Problem:
    rng = random.Random(seed)
    p   = fill_params(kwargs, PARAMETERS, rng)
    n_min, n_max, mode = p["n_min"], p["n_max"], p["mode"]

    # ── Attribute RNG — seed-only, so these are stable across n variants ────────
    attr_rng      = random.Random(seed ^ 0xCB075)
    outline_color = _pick_unused_color(list(bg_spec.colors), attr_rng,
                                       palette=list(obj_colors))
    tilt_angle    = attr_rng.choice(_VALID_ANGLES)
    top_corner_45 = attr_rng.choice(["top-left", "top-right"])

    # Reserve the outline color so no scene shape shares it.
    scene_colors = [c for c in obj_colors if tuple(c) != tuple(outline_color)]
    scene     = _make_scene(bg_spec, W, H, scene_colors, rng, n_min, n_max)
    shapes    = scene.shapes
    scene_img = scene.render()

    mg = int(_MARGIN * min(W, H))

    if mode == "straight":
        # Crop region: rectangle matching canvas aspect ratio.
        f      = rng.uniform(_SIZE_MIN_FRAC, _SIZE_MAX_FRAC)
        half_w = f * W / 2
        half_h = f * H / 2
        crop_w = int(round(2 * half_w))
        crop_h = int(round(2 * half_h))
        for _ in range(100):
            cx = rng.uniform(half_w + mg, W - half_w - mg)
            cy = rng.uniform(half_h + mg, H - half_h - mg)
            x0 = int(round(cx - half_w))
            y0 = int(round(cy - half_h))
            if not shapes or _box_contains_shape(shapes, x0, y0, crop_w, crop_h):
                break

        crop       = scene_img.crop((x0, y0, x0 + crop_w, y0 + crop_h))
        answer_img = crop.resize((W, H), Image.NEAREST)

        input_img = scene_img.copy()
        _draw_region_outline(input_img,
                             _rect_corners(cx, cy, half_w, half_h, 0),
                             outline_color)

        instruction = ("Crop to the interior of the outlined region. "
                       "Scale to fill the canvas using nearest-neighbor interpolation.")
        meta_extra = {
            "cx_frac":   cx / W,
            "cy_frac":   cy / H,
            "x0_frac":   x0 / W,
            "y0_frac":   y0 / H,
            "crop_size_frac": f,
        }

    else:  # tilted
        angle   = tilt_angle
        a_rad   = math.radians(angle)
        abs_cos = abs(math.cos(a_rad))
        abs_sin = abs(math.sin(a_rad))

        # Maximum f such that the rotated rectangle fits within canvas with margin.
        f_max = min(
            (W - 2 * mg) / (W * abs_cos + H * abs_sin),
            (H - 2 * mg) / (W * abs_sin + H * abs_cos),
        )
        f      = rng.uniform(_SIZE_MIN_FRAC, min(_SIZE_MAX_FRAC, f_max))
        half_w = f * W / 2
        half_h = f * H / 2
        crop_w = int(round(2 * half_w))
        crop_h = int(round(2 * half_h))

        # Axis-aligned bounding box half-extents of the rotated rectangle.
        bbox_half_w = half_w * abs_cos + half_h * abs_sin
        bbox_half_h = half_w * abs_sin + half_h * abs_cos

        for _ in range(100):
            cx = rng.uniform(bbox_half_w + mg, W - bbox_half_w - mg)
            cy = rng.uniform(bbox_half_h + mg, H - bbox_half_h - mg)
            x0 = int(round(cx - half_w))
            y0 = int(round(cy - half_h))
            if not shapes or _box_contains_shape(shapes, x0, y0, crop_w, crop_h, angle, cx, cy):
                break

        rotated    = scene_img.rotate(angle, resample=Image.NEAREST, center=(cx, cy))
        crop       = rotated.crop((x0, y0, x0 + crop_w, y0 + crop_h))

        # Determine which corner of the original tilted rectangle is highest
        # (smallest y in PIL coordinates). For angle > 0 the TL corner of the
        # axis-aligned crop maps to the topmost point in the original image
        # (→ "top-left"); for angle < 0 the TR corner does (→ "top-right").
        # At exactly 45° on a square canvas the two top corners are equidistant
        # → random choice for visual variety; on a rectangular canvas TL is
        # always highest.
        if angle > 0 and angle != 45:
            top_corner = "top-left"
        elif angle < 0:
            top_corner = "top-right"
        else:  # angle == 45
            if W == H:
                top_corner = top_corner_45
                if top_corner == "top-right":
                    crop = crop.transpose(Image.ROTATE_270)
            else:
                top_corner = "top-left"

        answer_img = crop.resize((W, H), Image.NEAREST)

        input_img = scene_img.copy()
        _draw_region_outline(input_img,
                             _rect_corners(cx, cy, half_w, half_h, angle),
                             outline_color)

        instruction = (
            f"Crop to the interior of the outlined region, deskewing so that "
            f"the highest corner of the region interior corresponds to the {top_corner} corner "
            f"of the cropped image. Scale to fill the canvas using nearest-neighbor "
            f"interpolation."
        )
        meta_extra = {
            "cx_frac":    cx / W,
            "cy_frac":    cy / H,
            "x0_frac":    x0 / W,
            "y0_frac":    y0 / H,
            "crop_size_frac":  f,
            "angle":      angle,
            "top_corner": top_corner,
        }

    return Problem(
        input_image=input_img,
        instruction=instruction,
        answer_image=answer_img,
        metadata={
            "params":        p,
            "bg_colors":     _bg_colors_meta(bg_spec),
            "scene_shapes":  _scene_shapes_meta(shapes),
            "outline_color": _rgb_to_hex(outline_color),
            **meta_extra,
        },
    )
