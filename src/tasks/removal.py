"""Task 08 — Removal.

Parameters
----------
n    : density bucket
mode : "attribute" — remove shapes matching a shape type, colour, or
                     both (⅓ chance each); ½ chance of inverting
       "location"  — remove the shape whose control point is at a given
                     position; ½ chance of inverting
"""
from __future__ import annotations
import random

import numpy as np

from .base import (
    Problem, fill_params, N_MIN_OPTIONS, N_MAX_OPTIONS,
    _make_scene, _shape_label, _fpos, _shape_occupancy_mask,
    _scene_shapes_meta, _bg_colors_meta, _rgb_to_hex,
)
from core.colors import name_of

NAME = "removal"
PARAMETERS = {
    "n_min": N_MIN_OPTIONS,
    "n_max": N_MAX_OPTIONS,
    "mode":  ["attribute", "location"],
}


def generate(seed: int, bg_spec, W: int, H: int,
             obj_colors: list, **kwargs) -> Problem:
    rng = random.Random(seed)
    p   = fill_params(kwargs, PARAMETERS, rng)
    n_min, n_max, mode = p["n_min"], p["n_max"], p["mode"]

    scene     = _make_scene(bg_spec, W, H, obj_colors, rng, n_min, n_max)
    shapes    = scene.shapes
    input_img = scene.render()

    if not shapes:
        return Problem(input_img, "No shapes to remove.", input_img.copy(),
                       {"params": p}, error=True)

    invert = rng.random() < 0.5

    meta = {
        "params": p,
        "bg_colors": _bg_colors_meta(bg_spec),
        "scene_shapes": _scene_shapes_meta(shapes),
        "invert": invert,
    }

    if mode == "attribute":
        attr = rng.choice(["shape", "color", "both"])
        meta["attr"] = attr

        n = len(shapes)
        if attr == "shape":
            # sorted() pins iteration order so rng.choice is PYTHONHASHSEED-
            # independent. Set comprehensions over strings iterate in
            # hash-seed-dependent order.
            all_types = sorted({s.shape_name for s in shapes})
            # When inverting, avoid types shared by all shapes (would remove nothing)
            valid = [t for t in all_types
                     if any(s.shape_name != t for s in shapes)] if invert else all_types
            sname   = rng.choice(valid or all_types)
            matched = [s for s in shapes if s.shape_name == sname]
            label   = _shape_label(sname)
            meta["match_shape"] = sname
            if invert and len(matched) < n:
                instruction = f"Remove all shapes except those that are a {label}."
            elif len(matched) == 1:
                instruction = f"Remove the {label}."
            else:
                instruction = f"Remove all shapes that are a {label}."

        elif attr == "color":
            # sorted() for PYTHONHASHSEED-independence. RGB tuples are
            # currently hashseed-stable (int hash is identity), but sort
            # defensively in case ShapeInstance.fill ever becomes a
            # string-containing type.
            all_colors = sorted({s.fill for s in shapes})
            # When inverting, avoid colors shared by all shapes (would remove nothing)
            valid = [c for c in all_colors
                     if any(s.fill != c for s in shapes)] if invert else all_colors
            color   = rng.choice(valid or all_colors)
            matched = [s for s in shapes if s.fill == color]
            cname   = name_of(color)
            meta["match_color"] = _rgb_to_hex(color)
            if invert and len(matched) < n:
                instruction = f"Remove all shapes except the {cname} ones."
            elif len(matched) == 1:
                instruction = f"Remove the {cname} shape."
            else:
                instruction = f"Remove all {cname} shapes."

        else:  # both
            # sorted() pins iteration order; sets of (str, tuple) iterate
            # in PYTHONHASHSEED-dependent order.
            combos  = sorted({(s.shape_name, s.fill) for s in shapes})
            sname, color = rng.choice(combos)
            matched = [s for s in shapes
                       if s.shape_name == sname and s.fill == color]
            label, cname = _shape_label(sname), name_of(color)
            meta["match_shape"] = sname
            meta["match_color"] = _rgb_to_hex(color)
            if invert:
                instruction = f"Remove all shapes except those that are a {cname} {label}."
            elif len(matched) == 1:
                instruction = f"Remove the {cname} {label}."
            else:
                instruction = f"Remove all shapes that are a {cname} {label}."

        action_ids = ({id(s) for s in shapes if s not in matched}
                      if invert else {id(s) for s in matched})

    else:  # location
        target = rng.choice(shapes)
        mask   = _shape_occupancy_mask([target], W, H)
        ys, xs = np.where(np.array(mask) > 128)
        if len(xs) > 0:
            idx = rng.randint(0, len(xs) - 1)
            px, py = float(xs[idx]), float(ys[idx])
        else:
            px, py = target.cx, target.cy  # fallback to center
        loc_desc = _fpos(px, py, W, H)
        if invert:
            action_ids  = {id(s) for s in shapes if s is not target}
            instruction = f"Remove all shapes except the one at {loc_desc}."
        else:
            action_ids  = {id(target)}
            instruction = f"Remove the shape at {loc_desc}."

    meta["n_removed"] = len(action_ids)
    keep_idxs     = [i for i, s in enumerate(shapes) if id(s) not in action_ids]
    answer        = scene.copy()
    answer.shapes = [answer.shapes[i] for i in keep_idxs]
    answer_img    = answer.render()

    return Problem(
        input_image=input_img,
        instruction=instruction,
        answer_image=answer_img,
        metadata=meta,
    )
