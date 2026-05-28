"""Task 16 — Counting (Tally Adjustment).

A tally strip is drawn along one edge with one tally mark per shape on the
canvas.  The model must count the target shapes, then remove tally marks from
the specified end until the tally count matches.  The count is NOT revealed
in the instruction.

Scene generation is fully mode-independent: both modes produce the same input
image for the same seed.  k_color and k_shape are each sampled uniformly from
[1, n-1], independently.  The scene is constructed to satisfy both counts
simultaneously by pre-assigning shape types and colors to four groups:

    A  (overlap)               target_sname  +  target_color
    B  (k_shape - overlap)     target_sname  +  other color
    C  (k_color - overlap)     other shape   +  target_color
    D  (n - k_color - k_shape + overlap)     other shape   +  other color

where overlap ∈ [max(0, k_color+k_shape-n), min(k_color, k_shape)].

Modes (equal probability):
  "shape" — count shapes of a specific type  (answer removes n - k_shape tallies)
  "color" — count shapes of a specific color (answer removes n - k_color tallies)

Parameters
----------
n_min, n_max : total number of shapes on the canvas
"""
from __future__ import annotations
import math
import random

from .base import (
    Problem, fill_params, N_MIN_OPTIONS, N_MAX_OPTIONS,
    _shape_plural, _rgb_to_hex,
)
from core.background import make_background
from core.shapes import ShapeInstance, SHAPES, ALL_SHAPE_NAMES
from core.colors import name_of

NAME = "counting"

PARAMETERS = {
    "n_min":  N_MIN_OPTIONS,
    "n_max":  N_MAX_OPTIONS,
    "mode":   ["shape", "color"],
}

_CMARGIN = 0.05
_AR_RANGE = (0.4, 2.5)
_NEAR_SQUARE = {"rectangle", "ring", "diamond"}


def _sample_ar(rng: random.Random, sname: str) -> float:
    shape_def = SHAPES[sname]
    if not shape_def.SCALABLE_1D:
        return shape_def.ASPECT_RATIO
    for _ in range(100):
        ar = math.exp(rng.uniform(math.log(_AR_RANGE[0]), math.log(_AR_RANGE[1])))
        if sname not in _NEAR_SQUARE or not (0.8 <= ar <= 1.25):
            return ar
    return shape_def.ASPECT_RATIO


def _overlaps(a: ShapeInstance, b: ShapeInstance, pad: float = 6.0) -> bool:
    ax1, ay1, ax2, ay2 = a.axis_aligned_bbox()
    bx1, by1, bx2, by2 = b.axis_aligned_bbox()
    return not (ax2 + pad <= bx1 or bx2 + pad <= ax1 or
                ay2 + pad <= by1 or by2 + pad <= ay1)


def _try_place(rng: random.Random, sname: str, color: tuple,
               W: int, H: int, existing: list, blockers: list,
               size_lo: float, size_hi: float) -> ShapeInstance | None:
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


def _make_tally_insts(n: int, side: str, tally_size: float,
                      tally_shape: str, tally_ar: float,
                      tally_color: tuple, W: int, H: int) -> list[ShapeInstance]:
    insts = []
    for i in range(n):
        if side == "top":
            step = W / (n + 1); cx, cy = step * (i + 1), tally_size
        elif side == "bottom":
            step = W / (n + 1); cx, cy = step * (i + 1), H - tally_size
        elif side == "left":
            step = H / (n + 1); cx, cy = tally_size, step * (i + 1)
        else:  # right
            step = H / (n + 1); cx, cy = W - tally_size, step * (i + 1)
        tw = tally_size
        th = tally_size / tally_ar if tally_ar >= 1.0 else tally_size
        insts.append(ShapeInstance(tally_shape, cx - tw / 2, cy - th / 2,
                                   tw, th, 0.0, tally_color))
    return insts


def generate(seed: int, bg_spec, W: int, H: int,
             obj_colors: list, **kwargs) -> Problem:
    rng = random.Random(seed)
    p   = fill_params(kwargs, PARAMETERS, rng)
    n_min, n_max = p["n_min"], p["n_max"]
    mode = p["mode"]

    # ── Attribute RNG — seed-only, so target/tally attributes are stable
    #    across n variants for the same seed ─────────────────────────────────
    attr_rng     = random.Random(seed ^ 0xC07A11)
    target_sname = attr_rng.choice(ALL_SHAPE_NAMES)
    tally_shape  = attr_rng.choice([s for s in ALL_SHAPE_NAMES if s != target_sname])
    target_color = tuple(attr_rng.choice(obj_colors))
    tally_color  = tuple(attr_rng.choice([c for c in obj_colors
                                          if tuple(c) != target_color]))

    # ── Scene RNG — every decision that must be identical across modes ─────────
    scene_rng = random.Random(seed ^ 0x5CEE_DA1A)

    n = scene_rng.randint(n_min, n_max)
    if n < 2:
        raise RuntimeError("counting: need n >= 2 to have at least one distractor")

    side = scene_rng.choice(["top", "bottom", "left", "right"])

    n_mid   = max(1, (n_min + n_max) / 2)
    short   = min(W, H)
    size_lo = max(32.0, short * max(0.02, 0.18 / n_mid ** 0.5))
    size_hi = max(size_lo, short * min(0.40, 0.55 / n_mid ** 0.5))

    tally_ar   = SHAPES[tally_shape].ASPECT_RATIO
    tally_size = max(6.0, min(
        (W if side in ("top", "bottom") else H) / (n + 1) * 0.7,
        (H if side in ("top", "bottom") else W) * 0.06,
    ))

    other_snames  = [s for s in ALL_SHAPE_NAMES
                     if s != target_sname and s != tally_shape]
    other_colors  = [tuple(c) for c in obj_colors
                     if tuple(c) != target_color and tuple(c) != tally_color]

    if not other_colors:
        raise RuntimeError("counting: need at least 3 distinct palette colors")

    # k_color and k_shape each uniform in [1, n-1], independently.
    k_color = scene_rng.randint(1, n - 1)
    k_shape = scene_rng.randint(1, n - 1)

    # overlap = shapes with BOTH target_color AND target_sname.
    overlap_lo = max(0, k_color + k_shape - n)
    overlap_hi = min(k_color, k_shape)
    overlap    = scene_rng.randint(overlap_lo, overlap_hi)

    # Provisional tallies used only as spatial blockers during scene placement.
    tally_prov = _make_tally_insts(n, side, tally_size,
                                   tally_shape, tally_ar, tally_color, W, H)

    # Place shapes group by group.
    #   A: overlap            — (target_sname, target_color)
    #   B: k_shape - overlap  — (target_sname, other color)
    #   C: k_color - overlap  — (other shape,  target_color)
    #   D: remainder          — (other shape,  other color)
    groups: list[tuple[int, str | None, tuple | None]] = [
        (overlap,                          target_sname, target_color),
        (k_shape - overlap,                target_sname, None         ),
        (k_color - overlap,                None,         target_color ),
        (n - k_color - k_shape + overlap,  None,         None         ),
    ]

    shapes: list[ShapeInstance] = []
    for count, sn_fixed, sc_fixed in groups:
        for _ in range(count):
            sn   = sn_fixed if sn_fixed is not None else scene_rng.choice(other_snames)
            sc   = sc_fixed if sc_fixed is not None else scene_rng.choice(other_colors)
            inst = _try_place(scene_rng, sn, sc, W, H, shapes, tally_prov, size_lo, size_hi)
            if inst is None:
                raise RuntimeError(
                    f"counting: could not place shape {len(shapes) + 1}/{n} "
                    f"on a {W}x{H} canvas"
                )
            shapes.append(inst)

    # ── Mode-specific answer ───────────────────────────────────────────────────
    k        = k_color if mode == "color" else k_shape
    n_remove = n - k

    remove_end = rng.choice(["start", "end"])
    if remove_end == "end":
        keep_indices = list(range(k))
        end_desc = "right" if side in ("top", "bottom") else "bottom"
    else:
        keep_indices = list(range(n_remove, n))
        end_desc = "left" if side in ("top", "bottom") else "top"

    what = (f"{name_of(target_color)} shapes" if mode == "color"
            else _shape_plural(target_sname))
    instruction = (
        f"The {name_of(tally_color)} shapes arranged in a line on the {side} of the image "
        f"are used as tallies. Remove tallies from the {end_desc} so the number of tallies "
        f"equals the number of {what}."
    )

    tally_insts = _make_tally_insts(n, side, tally_size,
                                    tally_shape, tally_ar, tally_color, W, H)

    bg = make_background(W, H, bg_spec)

    input_img = bg.copy()
    for s in shapes:      s.draw(input_img)
    for t in tally_insts: t.draw(input_img)

    answer_img = bg.copy()
    for s in shapes:         s.draw(answer_img)
    for i in keep_indices:   tally_insts[i].draw(answer_img)

    return Problem(
        input_image=input_img,
        instruction=instruction,
        answer_image=answer_img,
        metadata={
            "params":        p,
            "bg_colors":     [_rgb_to_hex(c) for c in bg_spec.colors],
            "mode":          mode,
            "n":             n,
            "k":             k,
            "k_color":       k_color,
            "k_shape":       k_shape,
            "overlap":       overlap,
            "target_shape":  target_sname,
            "target_color":  _rgb_to_hex(target_color),
            "side":          side,
            "tally_shape":   tally_shape,
            "tally_color":   _rgb_to_hex(tally_color),
            "tally_size":    tally_size,
            "remove_end":    remove_end,
        },
    )
