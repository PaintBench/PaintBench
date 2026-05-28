"""Task 09 — Recolor.

Parameters
----------
n    : density bucket
mode : "color_code" — change shapes to a new palette color
                   Target selection (1/3 each):
                     • one specific shape (by description)
                     • all shapes of a given color
                     • all shapes of a given shape type
                   1/2 chance of inverting the target set.
       "dropper" — sample the color at a random point on the body of a shape,
                   then recolor a target set to that sampled color.
"""
from __future__ import annotations
import random

import numpy as np

from .base import (
    Problem, fill_params, N_MIN_OPTIONS, N_MAX_OPTIONS,
    _make_scene, _unique_desc, _shape_label, _shape_plural,
    _desc_color, _pick_unused_color, _fpos, _shape_occupancy_mask,
    _scene_shapes_meta, _bg_colors_meta, _rgb_to_hex,
)
from core.colors import name_of

NAME = "recolor"
PARAMETERS = {
    "n_min": N_MIN_OPTIONS,
    "n_max": N_MAX_OPTIONS,
    "mode":  ["color_code", "dropper"],
}


def _select_target(shapes, rng):
    """Return (target_ids: set, target_desc: str, attr: str, invert: bool).

    Selects 1/3 by single shape, 1/3 by color, 1/3 by shape type.
    1/2 chance of inversion.
    """
    attr   = rng.choice(["single", "color", "type"])
    invert = rng.random() < 0.5

    if attr == "single":
        t = rng.choice(shapes)
        desc_t = _unique_desc(t, shapes, rng)
        matched = {id(t)}
        if invert:
            return {id(s) for s in shapes if id(s) not in matched}, \
                   f"all shapes except {desc_t}", attr, True
        else:
            return matched, desc_t, attr, False

    elif attr == "color":
        # sorted() for PYTHONHASHSEED-independence. RGB tuples are currently
        # hashseed-stable, but sort defensively for uniformity with the
        # {s.shape_name} site below.
        color = rng.choice(sorted({s.fill for s in shapes}))
        matched = {id(s) for s in shapes if s.fill == color}
        cname   = name_of(color)
        if invert:
            return {id(s) for s in shapes if s.fill != color}, \
                   f"all shapes that are not {cname}", attr, True
        else:
            label = "the" if len(matched) == 1 else "all"
            return matched, f"{label} {cname} shape{'s' if len(matched) != 1 else ''}", attr, False

    else:  # type
        # sorted() pins iteration order; set comprehensions over strings
        # iterate in PYTHONHASHSEED-dependent order.
        sname   = rng.choice(sorted({s.shape_name for s in shapes}))
        matched = {id(s) for s in shapes if s.shape_name == sname}
        slabel  = _shape_label(sname)
        splural = _shape_plural(sname)
        if invert:
            return {id(s) for s in shapes if s.shape_name != sname}, \
                   f"all shapes that are not a {slabel}", attr, True
        else:
            if len(matched) == 1:
                return matched, f"the {slabel}", attr, False
            else:
                return matched, f"all {splural}", attr, False


def generate(seed: int, bg_spec, W: int, H: int,
             obj_colors: list, **kwargs) -> Problem:
    rng = random.Random(seed)
    p   = fill_params(kwargs, PARAMETERS, rng)
    n_min, n_max, mode = p["n_min"], p["n_max"], p["mode"]

    # ── Attribute RNG — seed-only, so new color is stable across n variants ─────
    attr_rng  = random.Random(seed ^ 0xC010B)
    new_color = _pick_unused_color(list(bg_spec.colors), attr_rng, palette=list(obj_colors))

    # Reserve the new color so no scene shape shares it — guarantees recoloring
    # always produces a visible change and new_color stays unambiguous.
    scene_colors = [c for c in obj_colors if tuple(c) != tuple(new_color)]
    scene     = _make_scene(bg_spec, W, H, scene_colors, rng, n_min, n_max)
    shapes    = scene.shapes
    input_img = scene.render()

    if not shapes:
        return Problem(input_img, "No shapes to recolor.", input_img.copy(),
                       {"params": p}, error=True)

    # Target selection is shared across modes so the same seed always targets
    # the same shapes — the mode only determines how the new color is specified.
    target_ids, target_desc, attr, invert = _select_target(shapes, rng)
    target_idxs = {i for i, s in enumerate(shapes) if id(s) in target_ids}

    base_meta = {
        "params":       p,
        "bg_colors":    _bg_colors_meta(bg_spec),
        "scene_shapes": _scene_shapes_meta(shapes),
        "attr":         attr,
        "invert":       invert,
    }

    if mode == "color_code":
        new_desc  = _desc_color(new_color, is_new=True)

        answer = scene.copy()
        for i, s in enumerate(answer.shapes):
            if i in target_idxs:
                s.fill = new_color

        instruction = f"Recolor {target_desc} to {new_desc}."
        meta = {**base_meta, "new_color": _rgb_to_hex(new_color)}

    else:  # dropper — source shape must be outside the target set
        non_targets = [s for s in shapes if id(s) not in target_ids]
        if not non_targets:
            # All shapes are targets; exclude one to serve as the dropper source
            spare       = rng.choice(shapes)
            target_ids  = target_ids  - {id(spare)}
            target_idxs = {i for i, s in enumerate(shapes) if id(s) in target_ids}
            non_targets = [spare]
        src_shape     = rng.choice(non_targets)
        mask          = _shape_occupancy_mask([src_shape], W, H)
        ys, xs        = np.where(np.array(mask) > 128)
        if len(xs) > 0:
            idx = rng.randint(0, len(xs) - 1)
            px, py = float(xs[idx]), float(ys[idx])
        else:
            px, py = src_shape.cx, src_shape.cy
        sampled_color = src_shape.fill
        coords_desc   = _fpos(px, py, W, H)

        answer = scene.copy()
        for i, s in enumerate(answer.shapes):
            if i in target_idxs:
                s.fill = sampled_color

        instruction = f"Recolor {target_desc} to the color of the shape at {coords_desc}."
        meta = {**base_meta, "sampled_color": _rgb_to_hex(sampled_color)}

    answer_img = answer.render()
    return Problem(
        input_image=input_img,
        instruction=instruction,
        answer_image=answer_img,
        metadata=meta,
    )
