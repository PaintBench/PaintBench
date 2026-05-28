"""Task 21 — Visual Key (Color Mapping).

A color-mapping key is drawn along one edge of the canvas directly on the
background (no strip fill or dividing line — same style as counting tallies).
Each entry shows a source color swatch, an arrow, and either:
  • a destination color swatch  — recolor all shapes of that color
  • a bold red X mark           — remove all shapes of that color

Every color present on the canvas appears exactly once in the key.
Shapes are placed to never overlap the key area.
The model must apply all mappings to produce the answer image.

Parameters
----------
n_min / n_max : total number of shapes on the canvas
"""
from __future__ import annotations
import copy
import math
import random
from PIL import Image, ImageDraw

from .base import (
    Problem, fill_params, N_MIN_OPTIONS, N_MAX_OPTIONS,
    _scene_shapes_meta, _bg_colors_meta, _rgb_to_hex,
)
from core.background import make_background
from core.shapes import ShapeInstance, SHAPES, ALL_SHAPE_NAMES

NAME = "legend"
PARAMETERS = {
    "n_min": N_MIN_OPTIONS,
    "n_max": N_MAX_OPTIONS,
}

_STRIP_FRAC   = 0.14
_STRIP_MIN_PX = 48
_CMARGIN      = 0.05
_X_COLOR       = (220, 60, 60)
_SWATCH_BORDER = (0, 0, 0)
_SWATCH_BG     = (255, 255, 255)


# ---------------------------------------------------------------------------
# Shape placement (same pattern as counting task)
# ---------------------------------------------------------------------------

def _overlaps(a: ShapeInstance, b: ShapeInstance, pad: float = 6.0) -> bool:
    ax1, ay1, ax2, ay2 = a.axis_aligned_bbox()
    bx1, by1, bx2, by2 = b.axis_aligned_bbox()
    return not (ax2 + pad <= bx1 or bx2 + pad <= ax1 or
                ay2 + pad <= by1 or by2 + pad <= ay1)


_NEAR_SQUARE = {"rectangle", "ring", "diamond"}
_AR_RANGE    = (0.4, 2.5)

def _sample_ar(rng, sname: str) -> float:
    """Sample aspect ratio the same way generate_scene does for SCALABLE_1D shapes."""
    shape_def = SHAPES[sname]
    if not shape_def.SCALABLE_1D:
        return shape_def.ASPECT_RATIO
    for _ in range(100):
        ar = math.exp(rng.uniform(math.log(_AR_RANGE[0]), math.log(_AR_RANGE[1])))
        if sname not in _NEAR_SQUARE or not (0.8 <= ar <= 1.25):
            return ar
    return shape_def.ASPECT_RATIO


def _try_place(rng, sname, color, W, H, existing, blockers, size_lo, size_hi):
    ar = _sample_ar(rng, sname)
    for _ in range(100):
        size  = rng.uniform(size_lo, size_hi)
        w, h  = (size, size / ar) if ar >= 1.0 else (size * ar, size)
        x     = rng.uniform(W * _CMARGIN, W * (1 - _CMARGIN) - w)
        y     = rng.uniform(H * _CMARGIN, H * (1 - _CMARGIN) - h)
        angle = rng.uniform(0, 360) if SHAPES[sname].ROTATABLE else 0.0
        inst  = ShapeInstance(sname, x, y, w, h, angle, color)
        if not any(_overlaps(inst, s) for s in existing + blockers):
            return inst
    return None


def _strip_blocker(side: str, strip_size: int, W: int, H: int) -> ShapeInstance:
    """Return a dummy ShapeInstance covering the key strip, used as a placement blocker."""
    sname = ALL_SHAPE_NAMES[0]
    if side == "top":
        return ShapeInstance(sname, 0, 0, W, strip_size, 0.0, (0, 0, 0))
    elif side == "bottom":
        return ShapeInstance(sname, 0, H - strip_size, W, strip_size, 0.0, (0, 0, 0))
    elif side == "left":
        return ShapeInstance(sname, 0, 0, strip_size, H, 0.0, (0, 0, 0))
    else:  # right
        return ShapeInstance(sname, W - strip_size, 0, strip_size, H, 0.0, (0, 0, 0))


# ---------------------------------------------------------------------------
# Key strip rendering (no background fill — swatches drawn on canvas directly)
# ---------------------------------------------------------------------------

def _draw_swatch(draw: ImageDraw.ImageDraw,
                 x0: int, y0: int, sw: int,
                 color: tuple, brd: int) -> None:
    draw.rectangle([x0, y0, x0 + sw - 1, y0 + sw - 1],
                   fill=color, outline=_SWATCH_BORDER, width=brd)


def _draw_x_box(draw: ImageDraw.ImageDraw,
                x0: int, y0: int, sw: int, brd: int) -> None:
    draw.rectangle([x0, y0, x0 + sw - 1, y0 + sw - 1],
                   fill=_SWATCH_BG, outline=_SWATCH_BORDER, width=brd)
    pad = max(2, sw // 5)
    lw  = max(2, sw // 6)
    draw.line([(x0 + pad,          y0 + pad),
               (x0 + sw - 1 - pad, y0 + sw - 1 - pad)], fill=_X_COLOR, width=lw)
    draw.line([(x0 + sw - 1 - pad, y0 + pad),
               (x0 + pad,          y0 + sw - 1 - pad)], fill=_X_COLOR, width=lw)


def _draw_arrow_h(img: Image.Image, x0: int, cy: int, x1: int, sw: int) -> None:
    """Right-pointing arrow from x0 to x1, centered at cy."""
    aw = x1 - x0
    if aw < 4:
        return
    ah = max(4, min(aw // 2, sw // 2))
    ShapeInstance("arrow", x0, cy - ah // 2, aw, ah, 0.0, _SWATCH_BORDER).draw(img)


def _draw_arrow_v(img: Image.Image, cx: int, y0: int, y1: int, sw: int) -> None:
    """Downward-pointing arrow from y0 to y1, centered at cx.

    For ShapeInstance("arrow", x, y, W, H, 90):
      tip_y = (y + H/2) + W/2  →  y = y1 - H/2 - W/2
      tip_x = x + W/2          →  x = cx - W/2
    """
    al = y1 - y0   # arrow length  →  W in ShapeInstance
    if al < 4:
        return
    ah = max(4, min(al // 2, sw // 2))   # arrow width  →  H in ShapeInstance
    x  = cx - al // 2
    y  = y1 - ah // 2 - al // 2
    ShapeInstance("arrow", x, y, al, ah, 90.0, _SWATCH_BORDER).draw(img)


def _draw_key_strip(img: Image.Image, side: str,
                    entries: list, strip_size: int) -> None:
    """Draw color-key swatches and arrows along one edge of img (in-place).

    No background fill or dividing line — elements are drawn directly on the
    canvas, same style as counting-task tally marks.

    entries : [(src_rgb, dst_rgb_or_"X"), ...]
    """
    W, H  = img.size
    draw  = ImageDraw.Draw(img)
    n     = len(entries)

    if side == "top":
        bx0, by0, bx1, by1 = 0, 0, W, strip_size
        horiz = True
    elif side == "bottom":
        bx0, by0, bx1, by1 = 0, H - strip_size, W, H
        horiz = True
    elif side == "left":
        bx0, by0, bx1, by1 = 0, 0, strip_size, H
        horiz = False
    else:  # right
        bx0, by0, bx1, by1 = W - strip_size, 0, W, H
        horiz = False

    # White strip background
    draw.rectangle([bx0, by0, bx1 - 1, by1 - 1], fill=_SWATCH_BG)

    breadth   = strip_size
    span      = W if horiz else H
    cell_span = span / n
    sw  = max(8, min(int(breadth * 0.40), int(cell_span / 4.0)))
    arr = max(8, sw)    # arrow gets full swatch-width of room
    brd = max(1, sw // 12)

    for i, (src_c, dst) in enumerate(entries):
        if horiz:
            cc  = int(bx0 + cell_span * (i + 0.5))
            mid = (by0 + by1) // 2
            sx0 = cc - arr // 2 - sw
            dx0 = cc + arr // 2
            sy0 = dy0 = mid - sw // 2
            _draw_swatch(draw, sx0, sy0, sw, src_c, brd)
            _draw_arrow_h(img, sx0 + sw + 1, mid, dx0 - 1, sw)
            if dst == "X":
                _draw_x_box(draw, dx0, dy0, sw, brd)
            else:
                _draw_swatch(draw, dx0, dy0, sw, dst, brd)
        else:
            cc  = int(by0 + cell_span * (i + 0.5))
            mid = (bx0 + bx1) // 2
            sy0 = cc - arr // 2 - sw
            dy0 = cc + arr // 2
            sx0 = dx0 = mid - sw // 2
            _draw_swatch(draw, sx0, sy0, sw, src_c, brd)
            _draw_arrow_v(img, mid, sy0 + sw + 1, dy0 - 1, sw)
            if dst == "X":
                _draw_x_box(draw, dx0, dy0, sw, brd)
            else:
                _draw_swatch(draw, dx0, dy0, sw, dst, brd)


# ---------------------------------------------------------------------------
# Mapping builder
# ---------------------------------------------------------------------------

def _build_mapping(unique_colors: list, obj_colors: list,
                   rng: random.Random) -> dict:
    """Return {src_color: dst_color_or_"X"} for every unique source color.

    Guarantees at least one recolor and at least one X when n >= 2.
    """
    n       = len(unique_colors)
    palette = [tuple(c) for c in obj_colors]
    mapping = {}

    if n == 1:
        src = unique_colors[0]
        if rng.random() < 0.5:
            mapping[src] = "X"
        else:
            opts = [c for c in palette if c != src]
            mapping[src] = rng.choice(opts) if opts else rng.choice(palette)
    else:
        n_remove   = rng.randint(1, n - 1)
        remove_set = set(rng.sample(unique_colors, n_remove))
        for c in unique_colors:
            if c in remove_set:
                mapping[c] = "X"
            else:
                opts = [x for x in palette if x != c]
                mapping[c] = rng.choice(opts) if opts else rng.choice(palette)

    return mapping


# ---------------------------------------------------------------------------
# Task generator
# ---------------------------------------------------------------------------

def generate(seed: int, bg_spec, W: int, H: int,
             obj_colors: list, **kwargs) -> Problem:
    rng = random.Random(seed)
    p   = fill_params(kwargs, PARAMETERS, rng)
    n_min, n_max = p["n_min"], p["n_max"]

    side       = rng.choice(["top", "bottom", "left", "right"])
    strip_size = max(_STRIP_MIN_PX, int(_STRIP_FRAC * min(W, H)))
    blocker    = _strip_blocker(side, strip_size, W, H)

    # Shape size range — same formula as _make_scene / generate_scene
    n        = rng.randint(n_min, n_max)
    n_mid    = max(1, (n_min + n_max) / 2)
    short    = min(W, H)
    size_lo  = max(32.0, short * max(0.02, 0.18 / n_mid ** 0.5))
    size_hi  = max(size_lo, short * min(0.40, 0.55 / n_mid ** 0.5))

    # Place shapes explicitly, avoiding the key strip area.
    # Cap unique colors at 5 so the legend never gets too crowded.
    n_legend = min(rng.randint(2, 4), max(1, n), len(obj_colors))
    palette  = rng.sample([tuple(c) for c in obj_colors], n_legend)
    shapes   = []

    # Guarantee each palette color appears at least once
    for color in palette[:n]:
        sname = rng.choice(ALL_SHAPE_NAMES)
        inst  = _try_place(rng, sname, color, W, H, shapes, [blocker], size_lo, size_hi)
        if inst:
            shapes.append(inst)

    # Fill remaining slots from palette
    for _ in range(max(0, n - len(shapes))):
        sname = rng.choice(ALL_SHAPE_NAMES)
        color = rng.choice(palette)
        inst  = _try_place(rng, sname, color, W, H, shapes, [blocker], size_lo, size_hi)
        if inst:
            shapes.append(inst)

    if len(shapes) < n:
        raise RuntimeError(
            f"Could only place {len(shapes)}/{n} shapes on a {W}x{H} canvas"
        )

    bg = make_background(W, H, bg_spec)

    if not shapes:
        img = bg.copy()
        return Problem(img, "No shapes to map.", img.copy(), {"params": p}, error=True)

    # Build color mapping. sorted() pins iteration order so the subsequent
    # rng.shuffle (whose output depends on input order) is reproducible
    # across PYTHONHASHSEED values. {s.fill} (RGB tuples) is currently
    # hashseed-stable, but sort defensively.
    unique_colors = sorted({s.fill for s in shapes})
    rng.shuffle(unique_colors)
    mapping = _build_mapping(unique_colors, list(obj_colors), rng)
    entries = [(c, mapping[c]) for c in unique_colors]

    # Input: background + shapes + key strip
    input_img = bg.copy()
    for s in shapes:
        s.draw(input_img)
    _draw_key_strip(input_img, side, entries, strip_size)

    # Answer: apply mapping (recolor or remove), then draw same key strip
    answer_shapes = [copy.copy(s) for s in shapes if mapping[s.fill] != "X"]
    for s in answer_shapes:
        s.fill = mapping[s.fill]
    answer_img = bg.copy()
    for s in answer_shapes:
        s.draw(answer_img)
    _draw_key_strip(answer_img, side, entries, strip_size)

    instruction = (
        f"Apply the legend at the {side} of the image. "
        f"Recolor shapes whose color points to a new color, "
        f"and remove shapes whose color points to an X. "
        f"Keep the legend in place."
    )

    return Problem(
        input_image=input_img,
        instruction=instruction,
        answer_image=answer_img,
        metadata={
            "params":       p,
            "bg_colors":    _bg_colors_meta(bg_spec),
            "scene_shapes": _scene_shapes_meta(shapes),
            "side":         side,
            "mapping":      {_rgb_to_hex(k): (_rgb_to_hex(v) if v != "X" else "X")
                             for k, v in mapping.items()},
        },
    )
