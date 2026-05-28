"""Scene management: non-overlapping shape placement on a background.

All public APIs work in pixel space internally.  Task generators use the
rel_pos / rel_size helpers on ShapeInstance to produce instruction text
in fractional coordinates.
"""
from __future__ import annotations
import copy
import math
import random
from PIL import Image
from .background import BackgroundSpec, make_background
from .shapes import ShapeInstance, ALL_SHAPE_NAMES, SHAPES


# ---------------------------------------------------------------------------
# Size range from density
# ---------------------------------------------------------------------------

def _size_range_from_n(n_min: int, n_max: int) -> tuple[float, float]:
    """Return (lo_frac, hi_frac) shape size relative to canvas width."""
    n_mid = max(1, (n_min + n_max) / 2)
    lo = max(0.02, 0.18 / math.sqrt(n_mid))
    hi = min(0.40, 0.55 / math.sqrt(n_mid))
    return lo, hi


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

class Scene:
    def __init__(self, width: int, height: int, bg_spec: BackgroundSpec):
        self.width = width
        self.height = height
        self.bg_spec = bg_spec
        self.shapes: list[ShapeInstance] = []

    def render(self) -> Image.Image:
        img = make_background(self.width, self.height, self.bg_spec)
        for shape in self.shapes:
            shape.draw(img)
        return img

    def render_background(self) -> Image.Image:
        return make_background(self.width, self.height, self.bg_spec)

    def copy(self) -> "Scene":
        s = Scene(self.width, self.height, self.bg_spec)
        s.shapes = [copy.copy(sh) for sh in self.shapes]
        return s


# ---------------------------------------------------------------------------
# Overlap detection
# ---------------------------------------------------------------------------

def _aabb_overlap(s1: ShapeInstance, s2: ShapeInstance, margin: float = 4) -> bool:
    ax1, ay1, ax2, ay2 = s1.axis_aligned_bbox()
    bx1, by1, bx2, by2 = s2.axis_aligned_bbox()
    return not (ax2 + margin < bx1 or bx2 + margin < ax1 or
                ay2 + margin < by1 or by2 + margin < ay1)


def _bbox_overlap(a: tuple, b: tuple, margin: float = 4) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 + margin < bx1 or bx2 + margin < ax1 or
                ay2 + margin < by1 or by2 + margin < ay1)


# ---------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------

def generate_scene(
    width: int,
    height: int,
    bg_spec: BackgroundSpec,
    shape_colors: list,
    n_min: int = 4,
    n_max: int = 15,
    shape_types: list[str] = None,
    rng: random.Random = None,
    min_size_px: float = None,
    max_size_px: float = None,
    edge_margin_frac: float = 0.02,
    gap_margin_px: float = 4.0,
    allow_rotation: bool = None,   # None → use per-shape ROTATABLE flag
    fixed_n: int = None,           # override count (ignores n_min/n_max)
    aspect_range: tuple = (0.4, 2.5),  # (lo, hi) for 1D-scalable shapes; None → use default AR
    no_same_combo: bool = False,   # if True, no two shapes share (shape_name, color)
) -> Scene:
    """Generate a scene with non-overlapping shapes.

    Args:
        n_min, n_max  : count range; a random integer in [n_min, n_max] is sampled.
        shape_types   : which shape names to use (default: all ten).
        allow_rotation: True/False overrides per-shape ROTATABLE flag.
        fixed_n       : exact count, bypasses n_min/n_max sampling.
        aspect_range  : (lo, hi) aspect ratio range for 1D-scalable shapes
                        (default (0.4, 2.5), sampled log-uniformly); pass None
                        to use each shape's fixed default aspect ratio.
                        Rectangles, rings, and diamonds avoid the near-square band [0.8, 1.25].
    """
    if rng is None:
        # Fail loudly: every call site in src/tasks/ passes a seeded rng. A
        # fallback to random.Random() would seed from OS entropy and silently
        # produce non-deterministic scenes. Catching the missing-rng case here
        # prevents a whole class of "make generate produces different output
        # every run" regressions.
        raise ValueError(
            "generate_scene requires a seeded rng — pass rng=random.Random(seed)"
        )
    if shape_types is None:
        shape_types = ALL_SHAPE_NAMES
    if not shape_colors:
        raise ValueError("shape_colors must be provided — no global palette fallback")

    n_min, n_max = min(n_min, n_max), max(n_min, n_max)

    # Exclude bg colours from shape colour pool
    bg_set = {tuple(c) for c in bg_spec.colors}
    available = [c for c in shape_colors if tuple(c) not in bg_set] or list(shape_colors)

    n = fixed_n if fixed_n is not None else rng.randint(n_min, n_max)
    short = min(width, height)   # scale sizes off the shorter dimension so
                                 # portrait and landscape scenes are comparable
    edge_px = edge_margin_frac * short

    if min_size_px is None or max_size_px is None:
        lo_f, hi_f = _size_range_from_n(n_min, n_max)
        if min_size_px is None:
            min_size_px = lo_f * short
        if max_size_px is None:
            max_size_px = hi_f * short
    min_size_px = max(min_size_px, 32.0)
    max_size_px = max(max_size_px, min_size_px)

    scene = Scene(width, height, bg_spec)

    max_same_color = max(2, math.ceil(n / 3))
    max_same_type  = max(2, math.ceil(n / max(1, len(shape_types))))
    color_counts: dict = {}
    type_counts:  dict = {}
    used_combos:  set  = set()
    placed_bboxes: list = []   # cached bboxes of placed shapes
    _NEAR_SQUARE_SHAPES = {"rectangle", "ring", "diamond"}

    for _ in range(n):
        for _attempt in range(100):
            sname = rng.choice(shape_types)
            color = rng.choice(available)
            size  = rng.uniform(min_size_px, max_size_px)

            if color_counts.get(tuple(color), 0) >= max_same_color:
                continue
            if type_counts.get(sname, 0) >= max_same_type:
                continue
            if no_same_combo and (sname, tuple(color)) in used_combos:
                continue

            shape_def = SHAPES[sname]

            # Determine aspect ratio
            if shape_def.SCALABLE_1D and aspect_range is not None:
                for _ in range(100):
                    ar = math.exp(rng.uniform(math.log(aspect_range[0]), math.log(aspect_range[1])))
                    if sname not in _NEAR_SQUARE_SHAPES or not (0.8 <= ar <= 1.25):
                        break
            else:
                ar = shape_def.ASPECT_RATIO

            if ar >= 1.0:
                w, h = size, size / ar
            else:
                h, w = size, size * ar

            # Rotation
            if allow_rotation is True or (allow_rotation is None and shape_def.ROTATABLE):
                rot = rng.uniform(0, 360)
            else:
                rot = 0.0

            half_diag = math.ceil(math.hypot(w, h) / 2) + edge_px
            x_lo = half_diag
            y_lo = half_diag
            x_hi = width  - half_diag
            y_hi = height - half_diag
            if x_hi <= x_lo or y_hi <= y_lo:
                continue

            cx = rng.uniform(x_lo, x_hi)
            cy = rng.uniform(y_lo, y_hi)
            candidate = ShapeInstance(sname, cx - w / 2, cy - h / 2,
                                      w, h, rot, tuple(color))

            candidate_bbox = candidate.axis_aligned_bbox()
            if not any(_bbox_overlap(candidate_bbox, bb, gap_margin_px)
                       for bb in placed_bboxes):
                scene.shapes.append(candidate)
                placed_bboxes.append(candidate_bbox)
                color_counts[tuple(color)] = color_counts.get(tuple(color), 0) + 1
                type_counts[sname]         = type_counts.get(sname, 0)         + 1
                used_combos.add((sname, tuple(color)))
                break

    if len(scene.shapes) < n:
        raise RuntimeError(
            f"generate_scene: could only place {len(scene.shapes)}/{n} shapes "
            f"after 500 attempts each. "
            f"Try fewer shapes, more colors, a larger canvas, or a different seed."
        )

    return scene
