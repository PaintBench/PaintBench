"""Task 15 — Pattern Completion.

Two pattern modes, chosen uniformly:

  grid     — A 2-D grid of shapes constructed from a repeating tiling pattern.
              Row period (Y) and column period (X) are each sampled uniformly
              from [1, 3].  Rotatable shapes receive a random per-tile rotation
              drawn from the pattern definition.

  circular — Shapes placed evenly around a circle using the first row of the
              grid pattern as the repeating sequence (period = period_c).
              This ensures grid and circular produce corresponding tile types
              for the same seed.

Visibility constraints (grid):
  - The upper-left (period_r+1) × (period_c+1) block is always shown — this
    uniquely determines the full repeating pattern.
  - The bottom-right corner cell is always shown — this reveals the grid extent.
  - All other cells are candidates for removal.

Visibility constraints (circular):
  - The first period_k+1 consecutive shapes are always shown.
  - The last shape is always shown.

n randomly chosen candidate cells / positions are left empty; the model
must fill them in.

Grid dimensions (rows, cols) are sampled uniformly from all combinations that
fit on the canvas with legible cell sizes and have at least n candidate cells.

Parameters
----------
n_min, n_max : number of cells to fill in (missing shapes, >= 1)
mode         : "grid" or "circular"
"""
from __future__ import annotations
import math
import random
from PIL import Image

from .base import Problem, fill_params, _rgb_to_hex
from core.background import make_background
from core.shapes import ShapeInstance, SHAPES, ALL_SHAPE_NAMES

NAME = "pattern"

_N_MIN_OPTIONS = [1, 2, 4,  6]
_N_MAX_OPTIONS = [3, 5, 8, 12]
_MAX_GRID      = 10
_MIN_CELL_PX   = 45.0   # shape inside cell >= ~32 px even at maximum gap (0.14)
_CELL_GAP_FRAC = 0.08   # fixed gap fraction — only grid position is randomised

PARAMETERS = {
    "n_min": _N_MIN_OPTIONS,
    "n_max": _N_MAX_OPTIONS,
    "mode":  ["grid", "circular"],
}


def generate(seed: int, bg_spec, W: int, H: int,
             obj_colors: list, **kwargs) -> Problem:
    rng = random.Random(seed)

    p = fill_params(kwargs, PARAMETERS, rng)
    n_min, n_max = p["n_min"], p["n_max"]
    n_min, n_max = min(n_min, n_max), max(n_min, n_max)
    mode = p["mode"]

    if n_max < 1:
        raise ValueError(f"n_max={n_max}: need at least 1 missing cell")

    n = rng.randint(n_min, n_max)

    # Sample the repeating pattern definition before mode dispatch so that grid
    # and circular produce corresponding tile types for the same seed.
    # Grid uses all period_r × period_c tiles; circular uses the first period_c
    # tiles (the "first row" of the grid pattern) as its repeating sequence.
    period_r = rng.randint(1, 3)
    period_c = rng.randint(1, 3)
    n_types  = period_r * period_c

    pattern_shapes = [rng.choice(ALL_SHAPE_NAMES)         for _ in range(n_types)]
    pattern_colors = [tuple(rng.choice(obj_colors))        for _ in range(n_types)]
    pattern_rots   = [
        rng.uniform(0, 360) if SHAPES[pattern_shapes[i]].ROTATABLE else 0.0
        for i in range(n_types)
    ]

    if mode == "grid":
        return _grid(rng, p, n, bg_spec, W, H, obj_colors,
                     period_r, period_c, pattern_shapes, pattern_colors, pattern_rots)
    else:
        # Circular period = period_c; tile types = first row of the grid pattern
        return _circular(rng, p, n, bg_spec, W, H, obj_colors,
                         period_c,
                         pattern_shapes[:period_c],
                         pattern_colors[:period_c],
                         pattern_rots[:period_c])


# ═══════════════════════════════════════════════════════════════════════════════
# Grid mode
# ═══════════════════════════════════════════════════════════════════════════════

def _grid(rng, p, n, bg_spec, W, H, obj_colors,
          period_r, period_c, pattern_shapes, pattern_colors, pattern_rots):

    def candidate_count(nr: int, nc: int) -> int:
        """Cells not in the always-shown upper-left block or bottom-right corner."""
        must = set()
        for r in range(period_r + 1):
            for c in range(period_c + 1):
                must.add((r, c))
        must.add((nr - 1, nc - 1))
        return nr * nc - len(must)

    # All (rows, cols) that fit on canvas and have enough candidate cells
    valid = []
    for nr in range(period_r + 1, _MAX_GRID + 1):
        for nc in range(period_c + 1, _MAX_GRID + 1):
            if min(W / nc, H / nr) * 0.9 < _MIN_CELL_PX:
                continue
            if candidate_count(nr, nc) >= n:
                valid.append((nr, nc))

    if not valid:
        raise ValueError(
            f"Cannot place {n} missing cells for period ({period_r},{period_c}) "
            f"on a {W}x{H} canvas"
        )

    rows, cols = rng.choice(valid)

    # Build must-keep set
    must_keep = set()
    for r in range(period_r + 1):
        for c in range(period_c + 1):
            must_keep.add((r, c))
    must_keep.add((rows - 1, cols - 1))

    candidates = [(r, c) for r in range(rows) for c in range(cols)
                  if (r, c) not in must_keep]
    removed = set(map(tuple, rng.sample(candidates, n)))

    def pattern_for(r: int, c: int):
        idx = (r % period_r) * period_c + (c % period_c)
        return pattern_shapes[idx], pattern_colors[idx], pattern_rots[idx]

    cell_size = min(W / cols, H / rows) * 0.9
    margin_x  = rng.uniform(0, W - cols * cell_size)
    margin_y  = rng.uniform(0, H - rows * cell_size)
    gap_x = gap_y = cell_size * _CELL_GAP_FRAC

    def draw_cell(img: Image.Image, r: int, c: int) -> None:
        sname, color, rot = pattern_for(r, c)
        ar   = SHAPES[sname].ASPECT_RATIO
        x    = margin_x + c * cell_size + gap_x
        y    = margin_y + r * cell_size + gap_y
        cw   = cell_size - 2 * gap_x   # available width in cell
        ch   = cell_size - 2 * gap_y   # available height in cell
        # Fit shape to the (possibly non-square) cell, preserving aspect ratio
        sh_w = min(cw, ch * ar)
        sh_h = sh_w / ar
        cx, cy = x + cw / 2, y + ch / 2
        ShapeInstance(sname, cx - sh_w / 2, cy - sh_h / 2,
                      sh_w, sh_h, rot, color).draw(img)

    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    bg = make_background(W, H, bg_spec)

    input_img = bg.copy()
    for r, c in all_cells:
        if (r, c) not in removed:
            draw_cell(input_img, r, c)

    answer_img = bg.copy()
    for r, c in all_cells:
        draw_cell(answer_img, r, c)

    instruction = f"Fill in the missing {'shape' if n == 1 else 'shapes'} in this {rows}×{cols} pattern."

    pattern_cells = [
        {"row": r, "col": c, "shape": sn, "color": _rgb_to_hex(col), "rotation": rot}
        for r in range(period_r)
        for c in range(period_c)
        for sn, col, rot in [pattern_for(r, c)]
    ]

    return Problem(
        input_image=input_img,
        instruction=instruction,
        answer_image=answer_img,
        metadata={
            "params":        p,
            "bg_colors":     [_rgb_to_hex(c) for c in bg_spec.colors],
            "rows":          rows,
            "cols":          cols,
            "period_r":      period_r,
            "period_c":      period_c,
            "n_removed":     n,
            "pattern_cells": pattern_cells,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Circular mode
# ═══════════════════════════════════════════════════════════════════════════════

def _circular(rng, p, n, bg_spec, W, H, obj_colors,
              period_k, pattern_shapes, pattern_colors, pattern_rots):

    def tile_for(i: int):
        idx = i % period_k
        return pattern_shapes[idx], pattern_colors[idx], pattern_rots[idx]

    # Shape size: use max bounding-box dimension as reference across all tile shapes
    canvas_half = min(W, H) / 2
    n_sh_min    = max(2 * n, period_k + 2)
    size_cap    = 0.85 * canvas_half / (n_sh_min * 1.3 / (2 * math.pi) + 0.5) * 0.999
    if size_cap < 32.0:
        raise ValueError(
            f"Canvas {W}x{H} is too small to fit {n_sh_min} shapes of >= 32 px"
        )
    shape_size = max(32.0, min(size_cap, min(W, H) * rng.uniform(0.07, 0.13)))

    max_r = canvas_half * 0.85 - shape_size / 2

    valid_n = []
    for n_sh in range(period_k + 2, 40):
        min_r = n_sh * shape_size * 1.3 / (2 * math.pi)
        if min_r > max_r:
            continue
        # n_shapes must be a multiple of period_k for clean full cycles
        if n_sh % period_k != 0:
            continue
        # must-keep: first period_k+1 shapes (one full cycle + first repeated) + last
        must  = set(range(period_k + 1)) | {n_sh - 1}
        n_can = n_sh - len(must)
        # At least half of all shapes must remain visible (n removed <= floor(n_sh/2))
        if n_can >= n and n <= n_sh // 2:
            valid_n.append(n_sh)

    if not valid_n:
        raise ValueError(
            f"Cannot place {n} missing shapes in circular mode on a {W}x{H} canvas"
        )

    n_shapes = rng.choice(valid_n)
    min_r    = n_shapes * shape_size * 1.3 / (2 * math.pi)
    circle_r = rng.uniform(min_r, max(min_r, max_r))

    must_keep_c = set(range(period_k + 1)) | {n_shapes - 1}
    candidates  = [i for i in range(n_shapes) if i not in must_keep_c]
    removed     = set(rng.sample(candidates, n))

    cx_c = W / 2
    cy_c = H / 2

    def draw_shape(img: Image.Image, i: int) -> None:
        sname, color, rot = tile_for(i)
        ar    = SHAPES[sname].ASPECT_RATIO
        w     = shape_size if ar >= 1.0 else shape_size * ar
        h     = w / ar
        angle = 2 * math.pi * i / n_shapes
        px    = cx_c + circle_r * math.cos(angle)
        py    = cy_c + circle_r * math.sin(angle)
        ShapeInstance(sname, px - w / 2, py - h / 2, w, h, rot, color).draw(img)

    bg = make_background(W, H, bg_spec)

    input_img = bg.copy()
    for i in range(n_shapes):
        if i not in removed:
            draw_shape(input_img, i)

    answer_img = bg.copy()
    for i in range(n_shapes):
        draw_shape(answer_img, i)

    instruction = f"Fill in the missing {'shape' if n == 1 else 'shapes'} in this circular pattern of {n_shapes} shapes."

    pattern_steps = [
        {"step": i, "shape": pattern_shapes[i], "color": _rgb_to_hex(pattern_colors[i]),
         "rotation": pattern_rots[i]}
        for i in range(period_k)
    ]

    return Problem(
        input_image=input_img,
        instruction=instruction,
        answer_image=answer_img,
        metadata={
            "params":        p,
            "bg_colors":     [_rgb_to_hex(c) for c in bg_spec.colors],
            "mode":          "circular",
            "n_shapes":      n_shapes,
            "period_k":      period_k,
            "circle_radius": circle_r,
            "n_removed":     n,
            "pattern_steps": pattern_steps,
        },
    )
