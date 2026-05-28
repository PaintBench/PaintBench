"""Shape definitions.

Twelve shapes, each tagged with:
    ROTATABLE   — whether the shape can be randomly rotated in input scenes.
    SCALABLE_1D — whether width and height may be set independently
                  (i.e. the aspect ratio may vary).  False → w/h locked to
                  ASPECT_RATIO at all times.

Geometry is always defined in normalised [0, 1] × [0, 1] coords and scaled
to the actual (w, h) bounding box at draw time.
"""
from __future__ import annotations
import copy
import math
from dataclasses import dataclass
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _rotate_pts(pts: list, cx: float, cy: float, deg: float) -> list:
    if deg == 0:
        return list(pts)
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    return [(cx + (px - cx) * ca - (py - cy) * sa,
             cy + (px - cx) * sa + (py - cy) * ca)
            for px, py in pts]


def _scale_pts(pts: list, x: float, y: float, w: float, h: float) -> list:
    return [(x + px * w, y + py * h) for px, py in pts]


def _ellipse_pts(cx: float, cy: float, rx: float, ry: float, n: int = 128) -> list:
    return [(cx + rx * math.cos(2 * math.pi * k / n),
             cy + ry * math.sin(2 * math.pi * k / n))
            for k in range(n)]


def _regular_ngon(n: int, cx: float = 0.0, cy: float = 0.0,
                  r: float = 1.0) -> list:
    """Vertices of a regular n-gon.
    Odd n  → one vertex at top  (start = -π/2).
    Even n → flat top/bottom edges (start = -π/2 + π/n).
    """
    start = -math.pi / 2 if n % 2 == 1 else -math.pi / 2 + math.pi / n
    return [(cx + r * math.cos(start + 2 * math.pi * k / n),
             cy + r * math.sin(start + 2 * math.pi * k / n))
            for k in range(n)]


def _normalize(pts: list) -> tuple[list, float]:
    """Normalise pts to [0,1]×[0,1]; return (normalised_pts, aspect_ratio w/h)."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = max_x - min_x or 1e-9
    h = max_y - min_y or 1e-9
    return [((p[0] - min_x) / w, (p[1] - min_y) / h) for p in pts], w / h


def _poly_fill_mask(W: int, H: int, pts: list) -> np.ndarray:
    """Exact polygon rasterization via ray-casting (even-odd rule).

    Tests the centre of each pixel (x+0.5, y+0.5) against the polygon.
    Works correctly for convex, concave, and star-shaped polygons at any
    size and any floating-point vertex position.  Only iterates over the
    shape's bounding box, so it is efficient even on large canvases.

    Returns a boolean array of shape (H, W).
    """
    if not pts:
        return np.zeros((H, W), dtype=bool)

    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    n  = len(pts)

    x0 = max(0, int(np.floor(xs.min())))
    x1 = min(W, int(np.ceil(xs.max())) + 1)
    y0 = max(0, int(np.floor(ys.min())))
    y1 = min(H, int(np.ceil(ys.max())) + 1)
    if x0 >= x1 or y0 >= y1:
        return np.zeros((H, W), dtype=bool)

    # Pixel-centre grids over the bounding box
    px2d, py2d = np.meshgrid(
        np.arange(x0, x1, dtype=np.float64) + 0.5,
        np.arange(y0, y1, dtype=np.float64) + 0.5,
    )

    inside = np.zeros(px2d.shape, dtype=bool)
    xj, yj = xs[-1], ys[-1]
    for i in range(n):
        xi, yi = xj, yj
        xj, yj = xs[i], ys[i]
        dy = yj - yi
        if abs(dy) < 1e-12:   # horizontal edge — no vertical crossing
            continue
        # Top-left fill convention: count edge if yi <= py < yj (or reversed)
        crosses = ((yi <= py2d) & (yj > py2d)) | ((yj <= py2d) & (yi > py2d))
        x_cross = xi + (py2d - yi) * (xj - xi) / dy
        inside  ^= crosses & (px2d < x_cross)

    mask = np.zeros((H, W), dtype=bool)
    mask[y0:y1, x0:x1] = inside
    return mask


def _draw_polygon_aa(img: "Image.Image", pts: list, fill: tuple) -> None:
    """Fill a polygon using exact ray-casting rasterization (no PIL scanline)."""
    if not pts or not fill:
        return
    W, H = img.size
    mask = _poly_fill_mask(W, H, pts).view(np.uint8) * 255
    img.paste(Image.new("RGB", (W, H), fill), mask=Image.fromarray(mask, mode="L"))


def _draw_ring_aa(img: "Image.Image", outer_pts: list, inner_pts: list,
                  fill: tuple) -> None:
    """Fill the annular region between an outer and inner polygon."""
    if not outer_pts or not inner_pts or not fill:
        return
    W, H = img.size
    ring = (_poly_fill_mask(W, H, outer_pts) &
            ~_poly_fill_mask(W, H, inner_pts))
    mask = Image.fromarray(ring.view(np.uint8) * 255, mode="L")
    img.paste(Image.new("RGB", (W, H), fill), mask=mask)


def _edge_mid(a: tuple, b: tuple) -> tuple:
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


# ---------------------------------------------------------------------------
# Shape base class
# ---------------------------------------------------------------------------

class Shape:
    NAME: str         = ""
    ASPECT_RATIO: float = 1.0   # default w/h; used when SCALABLE_1D is False
    ROTATABLE: bool   = True    # may be randomly rotated in input scenes
    SCALABLE_1D: bool = False   # w and h may differ (else w/h = ASPECT_RATIO)

    def polygon(self) -> list[tuple[float, float]]:
        """Normalised polygon vertices in [0,1]×[0,1]."""
        raise NotImplementedError

    def control_points(self) -> dict[str, tuple[float, float]]:
        """Named control points in normalised [0,1]×[0,1] coordinates."""
        raise NotImplementedError

    def draw_on(self, si: "ShapeInstance", img: "Image.Image") -> None:
        """Render this shape instance onto *img*.  Default: filled polygon."""
        if si.fill:
            _draw_polygon_aa(img, si._canvas_pts(), si.fill)


# ---------------------------------------------------------------------------
# The ten shapes
# ---------------------------------------------------------------------------

class Circle(Shape):
    """Perfect circle.  Not rotatable (rotation has no visible effect).
    Not 1D-scalable (would become an ellipse, not a circle)."""
    NAME         = "circle"
    ASPECT_RATIO = 1.0
    ROTATABLE    = False
    SCALABLE_1D  = False

    def polygon(self):
        return _ellipse_pts(0.5, 0.5, 0.5, 0.5, 128)

    def control_points(self):
        return {
            "center":          (0.5, 0.5),
            "rightmost point": (1.0, 0.5),
            "leftmost point":  (0.0, 0.5),
            "highest point":   (0.5, 0.0),
            "lowest point":    (0.5, 1.0),
        }


class Rectangle(Shape):
    """Axis-aligned rectangle.  Not rotatable.  1D-scalable (w and h free)."""
    NAME         = "rectangle"
    ASPECT_RATIO = 1.5   # default when used as a single size parameter
    ROTATABLE    = False
    SCALABLE_1D  = True

    def polygon(self):
        return [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

    def control_points(self):
        return {
            "center":                (0.5, 0.5),
            "top-left corner":       (0.0, 0.0),
            "top-right corner":      (1.0, 0.0),
            "bottom-right corner":   (1.0, 1.0),
            "bottom-left corner":    (0.0, 1.0),
            "top edge midpoint":     (0.5, 0.0),
            "bottom edge midpoint":  (0.5, 1.0),
            "left edge midpoint":    (0.0, 0.5),
            "right edge midpoint":   (1.0, 0.5),
        }


class Cloud(Shape):
    """Cloud shape built from a truncated square + semicircular lobes.

    Construction (raw coords before normalisation):

      Pentagon base: (0.25,0) → (1.25,0) → (1.25,1) → (0,1) → (0,0.5)
      (a square with the top-left triangle cut off at the diagonal (0,0.5)→(0.25,0))

      Semicircles bulge outward on every edge EXCEPT the flat bottom:
        Edge 1 (top):   (0.25,0)→(1.25,0), centre (0.75,0),  r=0.50 → peak (0.75,−0.50)
        Edge 2 (right): (1.25,0)→(1.25,1), centre (1.25,0.5),r=0.50 → peak (1.75, 0.50)
        Edge 4 (left-lower): (0,1)→(0,0.5),centre (0,0.75), r=0.25 → peak (−0.25,0.75)
        Edge 5 (diagonal): (0,0.5)→(0.25,0), semicircle bulging upper-left

    Raw bbox: x∈[−0.25, 1.75], y∈[−0.50, 1.00]
    Not rotatable.  1D-scalable.
    """
    NAME        = "cloud"
    ROTATABLE   = False
    SCALABLE_1D = False

    # Raw bounding-box extents (precomputed for control_point normalisation)
    _RAW_XMIN = -0.25   # edge-4 lobe leftmost
    _RAW_XMAX =  1.75   # edge-2 lobe rightmost (centre 1.25 + r 0.50)
    _RAW_YMIN = -0.50   # edge-1 lobe topmost   (centre y=0  − r 0.50)
    _RAW_YMAX =  1.00   # flat base

    def __init__(self):
        pts = self._build_pts()
        self._norm, self.ASPECT_RATIO = _normalize(pts)

    def _build_pts(self, n: int = 128):
        pts = []

        # Edge 1: (0.25,0)→(1.25,0)  outward = up (−y)
        # Arc centre (0.75, 0), r=0.50, from π→2π via 3π/2
        cx, cy, r = 0.75, 0.0, 0.50
        for k in range(n + 1):
            theta = math.pi + math.pi * k / n
            pts.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
        # pts[0]=(0.25,0)  pts[-1]=(1.25,0) ✓

        # Edge 2: (1.25,0)→(1.25,1)  outward = right (+x)
        # Arc centre (1.25, 0.5), r=0.50, from −π/2→π/2 via 0
        cx, cy, r = 1.25, 0.5, 0.50
        for k in range(1, n + 1):
            theta = -math.pi / 2 + math.pi * k / n
            pts.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
        # ends at (1.25, 1) ✓

        # Bottom edge: (1.25,1)→(0,1) — straight, no lobe
        pts.append((0.0, 1.0))

        # Edge 4: (0,1)→(0,0.5)  outward = left (−x)
        # Arc centre (0, 0.75), r=0.25, from π/2→3π/2 via π
        cx, cy, r = 0.0, 0.75, 0.25
        for k in range(1, n + 1):
            theta = math.pi / 2 + math.pi * k / n
            pts.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
        # ends at (0, 0.5) ✓

        # Edge 5: (0,0.5)→(0.25,0)  outward = upper-left
        # Centre = midpoint (0.125, 0.25), r = half-chord-length = sqrt(0.3125)/2
        cx = 0.125
        cy = 0.25
        r  = math.sqrt(0.0625 + 0.25) / 2   # = sqrt(0.3125) / 2
        a_start = math.atan2(0.5  - cy, 0.00  - cx)  # angle to (0,   0.5)
        a_end   = math.atan2(0.0  - cy, 0.25  - cx)  # angle to (0.25,0  )
        span    = (a_end - a_start) % (2 * math.pi)  # CCW span ≈ π
        for k in range(1, n + 1):
            theta = a_start + span * k / n
            pts.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
        # last point ≈ (0.25, 0) → closes to start of edge-1 arc ✓

        return pts

    def polygon(self):
        return self._norm

    def control_points(self):
        xmin, xmax = self._RAW_XMIN, self._RAW_XMAX
        ymin, ymax = self._RAW_YMIN, self._RAW_YMAX
        w = xmax - xmin   # 2.0
        h = ymax - ymin   # 1.5

        def n(rx, ry):
            return ((rx - xmin) / w, (ry - ymin) / h)

        return {
            # top of edge-1 semicircle peak: (0.75, −0.50)
            "highest point":   n(0.75, -0.50),  # = (0.5,  0.0)
            # leftmost of edge-4 semicircle peak: (−0.25, 0.75)
            "leftmost point":  n(-0.25, 0.75),  # = (0.0,  0.833)
            # rightmost of edge-2 semicircle peak: (1.75, 0.50)
            "rightmost point": n(1.75,   0.50), # = (1.0,  0.667)
        }



class Hexagon(Shape):
    """Regular hexagon.  Rotatable, not 1D-scalable."""
    NAME        = "hexagon"
    ROTATABLE   = True
    SCALABLE_1D = False

    def __init__(self):
        pts = _regular_ngon(6)
        self._norm, self.ASPECT_RATIO = _normalize(pts)

    def polygon(self):
        return self._norm

    def control_points(self):
        return {"center": (0.5, 0.5)}


class Triangle(Shape):
    """30-60-90 right triangle.

    Placed with:
      • right angle (90°) at bottom-left
      • 60° angle   at top-left   (opposite the long leg)
      • 30° angle   at bottom-right (opposite the short leg)

    Short leg  (opposite 30°) is vertical  → length 1
    Long leg   (opposite 60°) is horizontal → length √3
    Hypotenuse connects 60° corner to 30° corner.

    Rotatable, not 1D-scalable (ratio fixed by the 30-60-90 constraint).
    """
    NAME         = "triangle"
    ROTATABLE    = True
    SCALABLE_1D  = False

    _SQRT3 = math.sqrt(3)

    def __init__(self):
        # Raw vertices in natural coords
        #   corner_90: (0, 1)  corner_60: (0, 0)  corner_30: (√3, 1)
        raw = [(0.0, 1.0), (0.0, 0.0), (self._SQRT3, 1.0)]
        norm, ar = _normalize(raw)
        self._norm = norm
        self.ASPECT_RATIO = ar   # √3 ≈ 1.732

    def polygon(self):
        return self._norm   # [(0,1),(0,0),(1,1)] after normalisation

    def control_points(self):
        right_vertex     = self._norm[0]   # (0, 1)
        sixty_vertex     = self._norm[1]   # (0, 0)
        thirty_vertex    = self._norm[2]   # (1, 1)
        return {
            "90-degree vertex":    right_vertex,
            "60-degree vertex":    sixty_vertex,
            "30-degree vertex":    thirty_vertex,
            "short leg midpoint":  _edge_mid(right_vertex, sixty_vertex),
            "long leg midpoint":   _edge_mid(right_vertex, thirty_vertex),
            "hypotenuse midpoint": _edge_mid(sixty_vertex, thirty_vertex),
        }


class Arrow(Shape):
    """Right-pointing arrow (shaft + triangular head).
    Rotatable and 1D-scalable (shaft length can vary)."""
    NAME         = "arrow"
    ASPECT_RATIO = 2.0
    ROTATABLE    = True
    SCALABLE_1D  = True

    def polygon(self):
        # Standard 7-point right-pointing arrow:
        # shaft (left 60%, 40% tall) + triangular head (right 40%, full height)
        return [
            (0.00, 0.30),
            (0.60, 0.30),
            (0.60, 0.00),
            (1.00, 0.50),
            (0.60, 1.00),
            (0.60, 0.70),
            (0.00, 0.70),
        ]

    def control_points(self):
        return {"tip": (1.0, 0.5)}


class Heart(Shape):
    """Heart shape (cleft at top, pointy apex at bottom).
    Rotatable and 1D-scalable."""
    NAME        = "heart"
    ROTATABLE   = True
    SCALABLE_1D = False

    def __init__(self):
        pts = self._raw_pts()
        self._norm, self.ASPECT_RATIO = _normalize(pts)
        # Pre-compute apex index
        n = len(self._norm)
        self._apex_i = max(range(n), key=lambda i: self._norm[i][1])

    def _raw_pts(self, n: int = 128):
        pts = []
        for k in range(n):
            t = 2 * math.pi * k / n
            x = 16 * math.sin(t) ** 3
            y = -(13 * math.cos(t)
                  - 5 * math.cos(2 * t)
                  - 2 * math.cos(3 * t)
                  -     math.cos(4 * t))
            pts.append((x, y))
        return pts

    def polygon(self):
        return self._norm

    def control_points(self):
        return {"pointy tip": self._norm[self._apex_i]}


class Star(Shape):
    """Five-pointed star.  Rotatable, not 1D-scalable."""
    NAME        = "star"
    ROTATABLE   = True
    SCALABLE_1D = False
    _INNER_RATIO = 0.382

    def __init__(self):
        raw = self._raw_pts()
        self._norm, self.ASPECT_RATIO = _normalize(raw)
        # Circumcenter is the raw origin (0,0); compute its normalised position
        xs = [p[0] for p in raw]
        ys = [p[1] for p in raw]
        self._cc = ((0 - min(xs)) / (max(xs) - min(xs)),
                    (0 - min(ys)) / (max(ys) - min(ys)))

    def _raw_pts(self):
        outer = _regular_ngon(5, r=1.0)
        a = math.pi / 5
        ca, sa = math.cos(a), math.sin(a)
        inner = [(p[0] * ca - p[1] * sa, p[0] * sa + p[1] * ca)
                 for p in _regular_ngon(5, r=self._INNER_RATIO)]
        pts = []
        for i in range(5):
            pts.append(outer[i])
            pts.append(inner[i])
        return pts

    def polygon(self):
        return self._norm

    def control_points(self):
        return {"center": self._cc}


class Semicircle(Shape):
    """Semicircle — flat base at bottom, curved arc at top.
    Rotatable, not 1D-scalable."""
    NAME         = "semicircle"
    ASPECT_RATIO = 2.0   # diameter : radius
    ROTATABLE    = True
    SCALABLE_1D  = False

    def __init__(self):
        cx, cy, r = 0.5, 1.0, 0.5
        N = 129
        pts = [(cx + r * math.cos(math.pi - math.pi * k / (N - 1)),
                cy - r * math.sin(math.pi - math.pi * k / (N - 1)))
               for k in range(N)]
        self._norm, _ = _normalize(pts)

    def polygon(self):
        return self._norm

    def control_points(self):
        return {
            "arc midpoint":      (0.5, 0.0),
            "diameter midpoint": (0.5, 1.0),
        }


class Cross(Shape):
    """Equal-arm plus/cross.  Not rotatable.  1D-scalable (arms can differ)."""
    NAME         = "cross"
    ASPECT_RATIO = 1.0
    ROTATABLE    = False
    SCALABLE_1D  = True
    _T = 1/3   # arm width = 1-2T = T, so all arms have equal length and width

    def polygon(self):
        t = self._T
        return [
            (  t, 0.0), (1-t, 0.0),
            (1-t,   t), (1.0,   t),
            (1.0, 1-t), (1-t, 1-t),
            (1-t, 1.0), (  t, 1.0),
            (  t, 1-t), (0.0, 1-t),
            (0.0,   t), (  t,   t),
        ]

    def control_points(self):
        return {
            "center":                         (0.5, 0.5),
            "rightmost edge midpoint": (1.0, 0.5),
            "leftmost edge midpoint":  (0.0, 0.5),
            "highest edge midpoint":   (0.5, 0.0),
            "lowest edge midpoint":    (0.5, 1.0),
        }


class Ring(Shape):
    """Elliptical ring (annulus): solid band between an outer and inner ellipse.

    The outer ellipse fills the bounding box.  The inner ellipse is scaled
    toward the centre by _INNER_RATIO (inner radius = 60 % of outer radius),
    giving a band that is 40 % of the outer semi-axis wide.

    Rotatable (orientation of an elliptical ring is visible) and 1D-scalable
    (ring can be circular or elliptical).  Single control point: center.

    Rendering uses a mask-composite approach so the interior is truly hollow
    (background shows through), regardless of PIL polygon limitations.
    """
    NAME         = "ring"
    ROTATABLE    = True
    SCALABLE_1D  = True
    ASPECT_RATIO = 1.5   # default when used with a single size parameter

    _INNER_RATIO = 0.60  # inner radius = 60 % of outer

    def __init__(self):
        pts = _ellipse_pts(0.5, 0.5, 0.5, 0.5, 128)
        self._norm, self.ASPECT_RATIO = _normalize(pts)
        # ASPECT_RATIO == 1.0 for a unit circle; kept for 1D-scalable support

    def polygon(self):
        return self._norm   # outer ellipse (used for bbox / overlap tests)

    def control_points(self):
        return {"center": (0.5, 0.5)}

    def inner_pts(self, outer_pts: list, cx: float, cy: float) -> list:
        """Shrink outer polygon toward (cx, cy) to produce the inner boundary."""
        r = self._INNER_RATIO
        return [(cx + (px - cx) * r, cy + (py - cy) * r) for px, py in outer_pts]

    def draw_on(self, si: "ShapeInstance", img: "Image.Image") -> None:
        outer_pts = si._canvas_pts()
        _draw_ring_aa(img, outer_pts, self.inner_pts(outer_pts, si.cx, si.cy), si.fill)


class Diamond(Shape):
    """Axis-aligned rhombus (diamond): four vertices at the midpoints of the
    bounding box edges.  Rotatable and 1D-scalable (wide vs tall diamond).
    Near-square aspect ratios [0.8, 1.25] are excluded at generation time so
    it never looks like a tilted square."""
    NAME         = "diamond"
    ROTATABLE    = True
    SCALABLE_1D  = True
    ASPECT_RATIO = 1.5   # default when used with a single size parameter

    def __init__(self):
        # Raw vertices: top, right, bottom, left (unit bounding box)
        raw = [(0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5)]
        self._norm, self.ASPECT_RATIO = _normalize(raw)

    def polygon(self):
        return self._norm

    def control_points(self):
        return {"center": (0.5, 0.5)}


# ---------------------------------------------------------------------------
# Shape registry
# ---------------------------------------------------------------------------

_ALL_CLASSES = [
    Circle, Triangle, Rectangle, Diamond, Hexagon, Semicircle,
    Heart, Star, Arrow, Cross, Cloud, Ring,
]

SHAPES: dict[str, Shape] = {}
for _cls in _ALL_CLASSES:
    _inst = _cls()
    SHAPES[_inst.NAME] = _inst

ALL_SHAPE_NAMES: list[str] = list(SHAPES.keys())
SCALABLE_1D_SHAPES: list[str] = [n for n, s in SHAPES.items() if s.SCALABLE_1D]


# ---------------------------------------------------------------------------
# Canvas control points (relative coords in [0,1]×[0,1])
# ---------------------------------------------------------------------------

CANVAS_CONTROL_POINTS: dict[str, tuple[float, float]] = {
    "center":                (0.5, 0.5),
    "top-left corner":       (0.0, 0.0),
    "top-right corner":      (1.0, 0.0),
    "bottom-left corner":    (0.0, 1.0),
    "bottom-right corner":   (1.0, 1.0),
    "top edge midpoint":     (0.5, 0.0),
    "bottom edge midpoint":  (0.5, 1.0),
    "left edge midpoint":    (0.0, 0.5),
    "right edge midpoint":   (1.0, 0.5),
}


# ---------------------------------------------------------------------------
# ShapeInstance — a placed, coloured, (optionally rotated) shape on the canvas
# ---------------------------------------------------------------------------

@dataclass
class ShapeInstance:
    """A concrete shape placed on the canvas, stored in pixel coordinates."""
    shape_name: str
    x: float          # left edge of unrotated bounding box (pixels)
    y: float          # top edge of unrotated bounding box (pixels)
    w: float          # bounding box width (pixels)
    h: float          # bounding box height (pixels)
    rotation: float   # degrees CCW around bbox centre (0 for non-rotatable)
    fill: tuple       # RGB fill colour

    @property
    def shape(self) -> Shape:
        return SHAPES[self.shape_name]

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def _canvas_pts(self) -> list[tuple[float, float]]:
        pts = _scale_pts(self.shape.polygon(), self.x, self.y, self.w, self.h)
        return _rotate_pts(pts, self.cx, self.cy, self.rotation)

    def draw(self, img: Image.Image) -> None:
        self.shape.draw_on(self, img)

    def control_points(self) -> dict[str, tuple[float, float]]:
        """Named control points in canvas pixel coordinates."""
        raw = self.shape.control_points()
        result = {}
        for name, (nx, ny) in raw.items():
            px = self.x + nx * self.w
            py = self.y + ny * self.h
            result[name] = _rotate_pts([(px, py)], self.cx, self.cy, self.rotation)[0]
        # Add axis-aligned bounding box control points (rotation-dependent)
        ax1, ay1, ax2, ay2 = self.axis_aligned_bbox()
        mx, my = (ax1 + ax2) / 2, (ay1 + ay2) / 2
        result.update({
            "bounding box top-left corner":      (ax1, ay1),
            "bounding box top edge midpoint":    (mx,  ay1),
            "bounding box top-right corner":     (ax2, ay1),
            "bounding box left edge midpoint":   (ax1, my),
            "bounding box center":               (mx,  my),
            "bounding box right edge midpoint":  (ax2, my),
            "bounding box bottom-left corner":   (ax1, ay2),
            "bounding box bottom edge midpoint": (mx,  ay2),
            "bounding box bottom-right corner":  (ax2, ay2),
        })
        return result

    def axis_aligned_bbox(self) -> tuple[float, float, float, float]:
        """(min_x, min_y, max_x, max_y) of the rotated shape."""
        pts = self._canvas_pts()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    def rel_pos(self, W: int, H: int) -> tuple[float, float]:
        """Center position as (frac_of_width, frac_of_height)."""
        return self.cx / W, self.cy / H

    def rel_size(self, W: int) -> tuple[float, float]:
        """(w/W, h/W) — sizes as fraction of canvas width."""
        return self.w / W, self.h / W

    def copy(self) -> "ShapeInstance":
        return copy.copy(self)


