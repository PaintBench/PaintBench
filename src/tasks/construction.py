"""Task 07 — Construction.

A scene with background shapes is presented.  The model must draw a
primitive onto the scene (overlay or underlay).  All positions within a
problem are specified consistently: either all named control points (50%)
or all random canvas coordinates (50%).

Parameters
----------
n    : density bucket (controls number of background shapes)
mode : "line"    — draw a straight line between two points with a
                   circular brush of a given diameter
       "circle"  — draw a filled circle; sub-mode chosen uniformly (1/3 each):
                   "center_tangent"  — center point + a point on the circumference
                   "diameter"        — two diametrically opposite points
                   "center_radius"   — center point + radius as % of image width
       "polygon" — draw a filled polygon with n ∈ {3, 4, 5} vertices
"""
from __future__ import annotations
import random
import math
import itertools
from PIL import ImageDraw

from .base import (
    Problem, fill_params, N_MIN_OPTIONS, N_MAX_OPTIONS,
    _make_scene, _sample_point,
    _desc_color, _pick_unused_color, _overlay_str, _CLIP_CLAUSE,
    _scene_shapes_meta, _bg_colors_meta, _rgb_to_hex,
)

NAME = "construction"
PARAMETERS = {
    "n_min": N_MIN_OPTIONS,
    "n_max": N_MAX_OPTIONS,
    "mode": ["line", "circle", "polygon"],
}

_BRUSH_MIN    = 0.010
_BRUSH_MAX    = 0.030
_CIRCLE_R_MIN = 0.05
_CIRCLE_R_MAX = 0.25


def _draw_line_circular_brush(draw: ImageDraw.ImageDraw,
                               x1: float, y1: float,
                               x2: float, y2: float,
                               color: tuple, diameter_px: int) -> None:
    """Draw a line with circular end-caps (round brush)."""
    r = diameter_px // 2
    draw.line([(x1, y1), (x2, y2)], fill=color, width=diameter_px)
    draw.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=color)
    draw.ellipse([x2 - r, y2 - r, x2 + r, y2 + r], fill=color)


def generate(seed: int, bg_spec, W: int, H: int,
             obj_colors: list, **kwargs) -> Problem:
    rng  = random.Random(seed)
    p    = fill_params(kwargs, PARAMETERS, rng)
    n_min, n_max, mode = p["n_min"], p["n_max"], p["mode"]

    # ── Attribute RNG — seed-only, so these are stable across n variants ────────
    attr_rng           = random.Random(seed ^ 0xC04D1)
    color              = _pick_unused_color(list(bg_spec.colors), attr_rng, palette=list(obj_colors))
    overlay            = attr_rng.random() < 0.5
    use_control_points = attr_rng.random() < 0.5
    circle_mode        = attr_rng.choice(["center_tangent", "diameter", "center_radius"])
    n_vertices         = attr_rng.choice([3, 4, 5])
    bw_frac            = round(attr_rng.uniform(_BRUSH_MIN, _BRUSH_MAX) * 1000) / 1000
    r_frac             = round(attr_rng.uniform(_CIRCLE_R_MIN, _CIRCLE_R_MAX) * 1000) / 1000

    # Reserve the new shape's color so no background scene shape shares it.
    scene_colors = [c for c in obj_colors if tuple(c) != tuple(color)]
    scene = _make_scene(bg_spec, W, H, scene_colors, rng, n_min, n_max)

    # Input = background scene rendered
    input_img = scene.render()
    c_desc = _desc_color(color, is_new=True)

    # Build answer canvas
    if overlay:
        answer_img = input_img.copy()
    else:
        answer_img = scene.render_background()
    draw = ImageDraw.Draw(answer_img)

    meta = {
        "params":             p,
        "bg_colors":          _bg_colors_meta(bg_spec),
        "scene_shapes":       _scene_shapes_meta(scene.shapes),
        "overlay":            overlay,
        "use_control_points": use_control_points,
        "draw_color":         _rgb_to_hex(color),
    }

    # Pre-generate two shared control points used consistently across all modes:
    #   line          → pt1 and pt2 are the endpoints
    #   circle        → pt1 is the center (all sub-modes); pt2 is the outer/tangent
    #                   point (center_tangent) or second diameter endpoint (diameter)
    #   polygon       → pt1 and pt2 are the first two vertices
    min_dist_px = int(0.10 * min(W, H))
    pt1_desc, (x1, y1) = _sample_point(scene.shapes, W, H, rng, use_control_points)
    for _ in range(10):
        pt2_desc, (x2, y2) = _sample_point(scene.shapes, W, H, rng, use_control_points)
        if math.hypot(x2 - x1, y2 - y1) >= min_dist_px:
            break

    if mode == "line":
        bw_px   = max(2, int(round(bw_frac * W)))
        _draw_line_circular_brush(draw, x1, y1, x2, y2, color, bw_px)
        instruction = (
            f"Draw a {c_desc} line from {pt1_desc} to {pt2_desc}, "
            f"using a circular brush with a diameter of {bw_frac * 100:.2f}% image width."
            f" Place it {_overlay_str(overlay)} any existing shapes."
        )
        meta.update({"brush_diameter_frac": bw_frac,
                     "pt1_frac": [x1 / W, y1 / H], "pt2_frac": [x2 / W, y2 / H]})

    elif mode == "circle":

        if circle_mode == "center_tangent":
            r_px = max(4, int(round(math.hypot(x2 - x1, y2 - y1))))
            draw.ellipse([x1 - r_px, y1 - r_px, x1 + r_px, y1 + r_px], fill=color)
            instruction = (
                f"Draw a filled {c_desc} circle centered at {pt1_desc} "
                f"with its edge passing through {pt2_desc}."
                f" Place it {_overlay_str(overlay)} any existing shapes."
                f"{_CLIP_CLAUSE}"
            )
            meta.update({"circle_mode": circle_mode,
                         "center_frac": [x1 / W, y1 / H],
                         "outer_pt_frac": [x2 / W, y2 / H]})

        elif circle_mode == "diameter":
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            r_px   = max(4, int(round(math.hypot(x2 - x1, y2 - y1) / 2)))
            draw.ellipse([cx - r_px, cy - r_px, cx + r_px, cy + r_px], fill=color)
            instruction = (
                f"Draw a filled {c_desc} circle with diameter endpoints at "
                f"{pt1_desc} and {pt2_desc}."
                f" Place it {_overlay_str(overlay)} any existing shapes."
                f"{_CLIP_CLAUSE}"
            )
            meta.update({"circle_mode": circle_mode,
                         "pt1_frac": [x1 / W, y1 / H],
                         "pt2_frac": [x2 / W, y2 / H]})

        else:  # center_radius — pt2 generated for rng consistency but unused
            r_px   = max(4, int(round(r_frac * W)))
            draw.ellipse([x1 - r_px, y1 - r_px, x1 + r_px, y1 + r_px], fill=color)
            instruction = (
                f"Draw a filled {c_desc} circle centered at {pt1_desc} "
                f"with a radius of {r_frac * 100:.1f}% image width."
                f" Place it {_overlay_str(overlay)} any existing shapes."
                f"{_CLIP_CLAUSE}"
            )
            meta.update({"circle_mode": circle_mode,
                         "center_frac": [x1 / W, y1 / H],
                         "radius_frac": r_frac})

    else:  # polygon — pt1 and pt2 are the first two vertices; sample n-2 more
        n = n_vertices
        min_tri_area = 0.002 * W * H

        def _no_collinear_triplets(coords):
            for (ax, ay), (bx, by), (cx, cy) in itertools.combinations(coords, 3):
                area = abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) / 2
                if area < min_tri_area:
                    return False
            return True

        for _ in range(20):
            extra  = [_sample_point(scene.shapes, W, H, rng, use_control_points)
                      for _ in range(n - 2)]
            coords = [(x1, y1), (x2, y2)] + [(x, y) for _, (x, y) in extra]
            descs  = [pt1_desc, pt2_desc]  + [d       for d, _      in extra]
            if _no_collinear_triplets(coords):
                break

        # Sort vertices by angle around centroid to guarantee a simple polygon
        cx_mean = sum(x for x, _ in coords) / n
        cy_mean = sum(y for _, y in coords) / n
        order   = sorted(range(n), key=lambda i: math.atan2(coords[i][1] - cy_mean,
                                                             coords[i][0] - cx_mean))
        coords = [coords[i] for i in order]
        descs  = [descs[i]  for i in order]

        # Rotate so pt1 is listed first; if pt2 ends up last (adjacent to pt1 in
        # the other direction), reverse the remaining vertices to list pt2 second.
        # Reversing just flips CW↔CCW — the polygon stays non-self-intersecting.
        start  = next(i for i, c in enumerate(coords) if c == (x1, y1))
        coords = coords[start:] + coords[:start]
        descs  = descs[start:]  + descs[:start]
        if coords[-1] == (x2, y2):
            coords = [coords[0]] + list(reversed(coords[1:]))
            descs  = [descs[0]]  + list(reversed(descs[1:]))

        draw.polygon(coords, fill=color)

        vertex_list = ", ".join(descs[:-1]) + f", and {descs[-1]}"
        instruction = (
            f"Draw a filled {c_desc} polygon with vertices in order at {vertex_list}."
            f" Place it {_overlay_str(overlay)} any existing shapes."
        )
        meta.update({"n_vertices": n,
                     "vertices_frac": [[x / W, y / H] for x, y in coords]})

    # For underlay: draw background shapes on top after construction
    if not overlay:
        for s in scene.shapes:
            s.draw(answer_img)

    return Problem(
        input_image=input_img,
        instruction=instruction,
        answer_image=answer_img,
        metadata=meta,
    )
