"""Base types and shared helpers for task generators.

Each task module exports:
    NAME        : str              — unique task identifier
    PARAMETERS  : dict[str, list]  — {"n_min": [...], "n_max": [...]} plus optional "mode": [...] for multi-mode tasks
    generate(seed, bg_spec, W, H, obj_colors, **kwargs) -> Problem
"""
from __future__ import annotations
import copy
import random
from dataclasses import dataclass, field
from PIL import Image, ImageChops, ImageDraw

from core.background import BackgroundSpec
from core.canvas import Scene, generate_scene
from core.colors import name_of
from core.shapes import ShapeInstance, CANVAS_CONTROL_POINTS

# ---------------------------------------------------------------------------
# n parameter options (shared by all tasks)
# ---------------------------------------------------------------------------

N_MIN_OPTIONS = [1, 4, 16, 64]
N_MAX_OPTIONS = [3, 15, 63, 100]


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    input_image:  Image.Image
    instruction:  str
    answer_image: Image.Image
    metadata:     dict = field(default_factory=dict)
    error:        bool = False

    def save(self, input_path: str, answer_path: str) -> None:
        self.input_image.save(input_path)
        self.answer_image.save(answer_path)


# ---------------------------------------------------------------------------
# Relative-coordinate formatting helpers
# ---------------------------------------------------------------------------

def _fpos(x: float, y: float, W: int, H: int) -> str:
    """Format a pixel position as '(P%, P%)'."""
    return f"({x / W * 100:.2f}%, {y / H * 100:.2f}%)"


def _flen(px: float, W: int) -> str:
    """Format a pixel length as a percentage of canvas width."""
    return f"{px / W * 100:.2f}% of the image width"


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _rgb_to_hex(rgb: tuple) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _desc_color(rgb: tuple, is_new: bool = False) -> str:
    """'name' for existing colours; 'name (#HEX)' for colours to draw."""
    n = name_of(rgb)
    return f"{n} ({_rgb_to_hex(rgb)})" if is_new else n


def _pick_unused_color(used_rgbs: list[tuple], rng: random.Random,
                       palette: list[tuple]) -> tuple:
    """Pick a color from palette not already in used_rgbs."""
    used = {tuple(c) for c in used_rgbs}
    pool = [c for c in palette if tuple(c) not in used]
    return rng.choice(pool) if pool else rng.choice(palette)


# ---------------------------------------------------------------------------
# Scene-building helper
# ---------------------------------------------------------------------------

def _make_scene(bg_spec: BackgroundSpec, W: int, H: int,
                obj_colors: list[tuple], rng: random.Random,
                n_min: int = 4, n_max: int = 15,
                no_same_combo: bool = True, **kwargs) -> Scene:
    return generate_scene(W, H, bg_spec,
                          shape_colors=obj_colors,
                          n_min=n_min, n_max=n_max,
                          rng=rng,
                          no_same_combo=no_same_combo,
                          **kwargs)


# ---------------------------------------------------------------------------
# Shape description helpers
# ---------------------------------------------------------------------------

def _shape_label(shape_name: str) -> str:
    return shape_name.replace("_", " ")


def _shape_plural(shape_name: str) -> str:
    """Return a plural descriptor for the shape name."""
    return _shape_label(shape_name) + " shapes"


def _unique_desc(target: ShapeInstance, shapes: list[ShapeInstance],
                  rng: random.Random = None) -> str:
    """Return a textual description of *target*.

    When the target's colour is unique, picks uniformly among the forms that
    are unambiguous: by-colour ("the red shape"), by-both ("the red circle"),
    and — only when the shape type is also unique — by-geometry ("the circle").
    Falls back to colour+geometry when the colour is not unique.
    """
    color = name_of(target.fill)
    label = _shape_label(target.shape_name)
    unique_color = sum(1 for s in shapes if s.fill == target.fill) == 1
    unique_type  = sum(1 for s in shapes if s.shape_name == target.shape_name) == 1
    if unique_color:
        if rng is not None:
            options = [f"the {color} shape", f"the {color} {label}"]
            if unique_type:
                options.append(f"the {label}")
            return rng.choice(options)
        return f"the {color} shape"
    return f"the {color} {label}"


# ---------------------------------------------------------------------------
# Control-point sampling helpers
# ---------------------------------------------------------------------------

def _sample_shape_control_point(shape: ShapeInstance, rng: random.Random) -> tuple[str, tuple]:
    """Sample an intrinsic control point from *shape* (excludes bounding-box points).

    Returns (control_point_name, (px_x, px_y)).
    """
    all_pts = shape.control_points()
    intrinsic = {k: v for k, v in all_pts.items() if not k.startswith("bounding box")}
    return rng.choice(list(intrinsic.items()))


def _sample_control_point(shapes: list[ShapeInstance], W: int, H: int,
               rng: random.Random,
               exclude: tuple = None) -> tuple:
    """Sample a control point with equal probability per shape.

    Canvas is treated as an additional shape.  If *exclude* is provided as
    (shape_ref_or_None, control_point_name), that exact (shape, name) combo is skipped.

    Returns (shape_or_None, control_point_name, (px_x, px_y)).
    ``shape_or_None`` is None when the canvas was chosen.
    """
    # Build pool: list of (shape_or_None, control_point_name, (px, py))
    # First collect per-source entries
    sources: list = list(shapes) + [None]   # None = canvas
    while True:
        src = rng.choice(sources)
        if src is None:
            control_point_dict = {name: (rx * W, ry * H)
                                  for name, (rx, ry) in CANVAS_CONTROL_POINTS.items()}
            control_point_name, control_point_px = rng.choice(list(control_point_dict.items()))
        else:
            control_point_name, control_point_px = _sample_shape_control_point(src, rng)
        if exclude is not None:
            ex_src, ex_name = exclude
            if ex_src is src and ex_name == control_point_name:
                continue
        return src, control_point_name, control_point_px


def _sample_point(shapes: list[ShapeInstance], W: int, H: int,
                  rng: random.Random,
                  use_control_points: bool = True) -> tuple[str, tuple]:
    """Sample a point as either a named control point or a random canvas location.

    Returns (description_str, (px, py)).
    """
    if use_control_points:
        src, control_point_name, control_point_px = _sample_control_point(shapes, W, H, rng)
        if src is None:
            label = f"the {control_point_name} of the image"
        else:
            same_type = [s for s in shapes if s.shape_name == src.shape_name]
            if len(same_type) == 1:
                shape_desc = f"the {src.shape_name}"
            else:
                shape_desc = f"the {name_of(src.fill)} {src.shape_name}"
            label = f"the {control_point_name} of {shape_desc}"
        return label, control_point_px
    else:
        px = rng.uniform(0.05 * W, 0.95 * W)
        py = rng.uniform(0.05 * H, 0.95 * H)
        return _fpos(px, py, W, H), (px, py)


# ---------------------------------------------------------------------------
# Polygon region helpers (flood fill, blending, gradient)
# ---------------------------------------------------------------------------


def _poly_overlaps_shapes(poly: list, shapes: list) -> bool:
    """Return True if the polygon contains at least one point from any shape.

    Checks the centre and four corners of each shape's bounding box.
    """
    for s in shapes:
        cx, cy = s.x + s.w / 2, s.y + s.h / 2
        for px, py in [(cx, cy),
                       (s.x,       s.y),
                       (s.x + s.w, s.y),
                       (s.x,       s.y + s.h),
                       (s.x + s.w, s.y + s.h)]:
            if _point_in_poly(px, py, poly):
                return True
    return False


def _point_in_poly(px: float, py: float, poly: list) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and px < (xj - xi) * (py - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _draw_poly_outline(img: Image.Image, poly: list,
                       color: tuple, width: int = None) -> None:
    if width is None:
        width = max(3, img.width // 100)
    draw = ImageDraw.Draw(img)
    pts = [(int(round(x)), int(round(y))) for x, y in poly]
    draw.polygon(pts, outline=color, width=width)


def _make_poly_mask(W: int, H: int, poly: list) -> Image.Image:
    mask = Image.new("L", (W, H), 0)
    pts = [(int(round(x)), int(round(y))) for x, y in poly]
    if len(pts) >= 3:
        ImageDraw.Draw(mask).polygon(pts, fill=255)
    return mask


def _shape_occupancy_mask(shapes: list, W: int, H: int) -> Image.Image:
    """Return an L-mode mask that is 255 at shape pixels, 0 at background."""
    rgb = Image.new("RGB", (W, H), (0, 0, 0))
    for s in shapes:
        s_white = copy.copy(s)
        s_white.fill = (255, 255, 255)
        s_white.draw(rgb)
    return rgb.split()[0]


def _region_coverage(poly_mask: Image.Image,
                     shape_mask: Image.Image) -> tuple[float, float]:
    """Return (fg_frac, bg_frac) of the polygon interior.

    fg_frac : fraction of interior pixels that are shape pixels
    bg_frac : fraction of interior pixels that are background pixels
    """
    total = poly_mask.histogram()[255]
    if total == 0:
        return 0.0, 0.0
    fg = ImageChops.multiply(poly_mask, shape_mask).histogram()[255]
    return fg / total, (total - fg) / total


def _cross(O: tuple, A: tuple, B: tuple) -> float:
    return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])


def _convex_hull(points: list[tuple]) -> list[tuple[float, float]]:
    """Andrew's monotone chain convex hull. Returns vertices in CCW order."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts
    lower: list = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _shape_polygon(rng: random.Random, shapes: list, W: int, H: int,
                   n_pts: int = 12,
                   min_fg_frac: float = 0.05,
                   min_bg_frac: float = 0.10) -> list[tuple[float, float]]:
    """Convex polygon anchored around a randomly chosen scene shape.

    Scatters *n_pts* points within a padded bounding box of the chosen shape
    and returns their convex hull.  Retries up to 20 times to ensure the
    region contains at least *min_fg_frac* shape pixels and *min_bg_frac*
    background pixels (as fractions of the polygon interior).
    """
    shape_mask = _shape_occupancy_mask(shapes, W, H)
    poly = None
    for _ in range(20):
        shape = rng.choice(shapes)
        pad = max(
            rng.uniform(0.3, 0.8) * max(shape.w, shape.h),
            0.05 * min(W, H),
        )
        mg = 0.04
        x_min = max(mg * W, shape.x - pad)
        x_max = min((1 - mg) * W, shape.x + shape.w + pad)
        y_min = max(mg * H, shape.y - pad)
        y_max = min((1 - mg) * H, shape.y + shape.h + pad)
        bbox_pts = [
            (shape.x,           shape.y),
            (shape.x + shape.w, shape.y),
            (shape.x + shape.w, shape.y + shape.h),
            (shape.x,           shape.y + shape.h),
        ]
        random_pts = [(rng.uniform(x_min, x_max), rng.uniform(y_min, y_max))
                      for _ in range(n_pts)]
        poly = _convex_hull(bbox_pts + random_pts)
        fg_frac, bg_frac = _region_coverage(_make_poly_mask(W, H, poly), shape_mask)
        if fg_frac >= min_fg_frac and bg_frac >= min_bg_frac:
            return poly
    return poly  # best effort after 20 attempts


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _overlay_str(overlay: bool) -> str:
    return "on top of" if overlay else "underneath"


_CLIP_CLAUSE         = " Clip any parts that may extend beyond the image boundary."
_KEEP_OUTLINE_CLAUSE = " Keep the outline as is."


def _order_clause(overlay: bool) -> str:
    return (f" Place the transformed shape {_overlay_str(overlay)}"
            f" any possible overlapping shapes.")


# ---------------------------------------------------------------------------
# Parallelogram region (for gradient / blending tasks)
# ---------------------------------------------------------------------------

def _random_parallelogram(rng: random.Random, W: int, H: int,
                          orientation: str) -> list[tuple[float, float]]:
    """Random axis-aligned parallelogram, clamped to canvas.

    orientation : "horizontal" (wider than tall, straight top/bottom edges)
                  "vertical"   (taller than wide, straight left/right edges)
    """
    mg = 0.04  # margin as fraction
    if orientation == "horizontal":
        cx = rng.uniform(0.3 * W, 0.7 * W)
        cy = rng.uniform(0.3 * H, 0.7 * H)
        half_w = rng.uniform(0.18 * W, 0.36 * W)
        half_h = rng.uniform(0.06 * H, 0.16 * H)
        slant  = rng.uniform(-0.12 * W, 0.12 * W)
        tl = (cx - half_w + slant, cy - half_h)
        tr = (cx + half_w + slant, cy - half_h)
        br = (cx + half_w - slant, cy + half_h)
        bl = (cx - half_w - slant, cy + half_h)
    else:
        cx = rng.uniform(0.3 * W, 0.7 * W)
        cy = rng.uniform(0.3 * H, 0.7 * H)
        half_w = rng.uniform(0.06 * W, 0.16 * W)
        half_h = rng.uniform(0.18 * H, 0.36 * H)
        slant  = rng.uniform(-0.12 * H, 0.12 * H)
        tl = (cx - half_w, cy - half_h + slant)
        tr = (cx + half_w, cy - half_h - slant)
        br = (cx + half_w, cy + half_h - slant)
        bl = (cx - half_w, cy + half_h + slant)
    # Clamp all points to canvas with margin
    pts = [tl, tr, br, bl]
    return [(max(mg * W, min((1 - mg) * W, x)),
             max(mg * H, min((1 - mg) * H, y))) for x, y in pts]


def _para_mask(W: int, H: int, para: list) -> Image.Image:
    mask = Image.new("L", (W, H), 0)
    pts = [(int(round(x)), int(round(y))) for x, y in para]
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    return mask


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _scene_shapes_meta(shapes: list) -> list:
    """Return [{shape, color}] for every shape in the scene."""
    return [{"shape": s.shape_name, "color": _rgb_to_hex(s.fill)} for s in shapes]


def _bg_colors_meta(bg_spec) -> list:
    """Return background color(s) as hex strings."""
    return [_rgb_to_hex(c) for c in bg_spec.colors]


# ---------------------------------------------------------------------------
# Parameter filling
# ---------------------------------------------------------------------------

def fill_params(kwargs: dict, parameters: dict, rng: random.Random) -> dict:
    """Fill missing parameter keys with random choices.

    Coerces string kwargs to the type of the first option so that values
    stringified by the web visualizer (e.g. "4" vs 4) still match.
    """
    out = {}
    for key, options in parameters.items():
        val = kwargs.get(key)
        if val is not None and options:
            target_type = type(options[0])
            if not isinstance(val, target_type):
                try:
                    val = target_type(val)
                except (ValueError, TypeError):
                    val = None
        if val is not None:
            out[key] = val
        else:
            out[key] = rng.choice(options)
    return out
