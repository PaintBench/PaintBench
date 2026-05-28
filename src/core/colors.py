"""Color palettes for PaintBench."""
from __future__ import annotations
import random as _random

# ---------------------------------------------------------------------------
# PaintBench benchmark palettes
# ---------------------------------------------------------------------------

# Standard palette — common web colors (exact hex codes)
STANDARD_PALETTE: dict[str, tuple[int, int, int]] = {
    "red":        (255,   0,   0),   # #FF0000
    "orange":     (255, 165,   0),   # #FFA500
    "yellow":     (255, 255,   0),   # #FFFF00
    "green":      (  0, 255,   0),   # #00FF00
    "blue":       (  0,   0, 255),   # #0000FF
    "purple":     (128,   0, 128),   # #800080
    "pink":       (255, 192, 203),   # #FFC0CB
    "brown":      (139,  69,  19),   # #8B4513
    "black":      (  0,   0,   0),   # #000000
    "gray":       (128, 128, 128),   # #808080
    "white":      (255, 255, 255),   # #FFFFFF
}

# Nonstandard palette — perceptually distinct variants with uncommon hex codes
NONSTANDARD_PALETTE: dict[str, tuple[int, int, int]] = {
    "crimson":          (195,  27,  55),   # #C31B37
    "tangerine-colored":(244, 123,  22),   # #F47B16
    "gold":             (228, 186,  24),   # #E4BA18
    "olive-colored":    (113, 122,  30),   # #717A1E
    "cyan":             ( 15, 225, 223),   # #0FE1DF
    "lavender":         (217, 210, 233),   # #D9D2E9
    "magenta":          (242,  13, 216),   # #F20DD8
    "tan-colored":      (203, 170, 133),   # #CBAA85
    "jet black":        ( 16,  18,  17),   # #101211
    "silver":           (187, 188, 186),   # #BBBCBA
    "ivory white":      (248, 246, 232),   # #F8F6E8
}

# ---------------------------------------------------------------------------
# Internal scene palette (used by visualizer / random generation)
# ---------------------------------------------------------------------------

# Named color palette: name -> (R, G, B)
PALETTE: dict[str, tuple[int, int, int]] = {
    "red":        (220,  50,  50),
    "orange":     (230, 130,  40),
    "yellow":     (240, 210,  50),
    "lime":       (100, 200,  60),
    "green":      ( 40, 160,  80),
    "teal":       ( 40, 150, 150),
    "cyan":       ( 50, 190, 200),
    "sky":        (120, 190, 230),
    "blue":       ( 50, 100, 200),
    "navy":       ( 30,  50, 140),
    "indigo":     ( 80,  60, 190),
    "violet":     (160,  60, 200),
    "pink":       (230,  80, 150),
    "coral":      (230, 110, 100),
    "brown":      (140,  90,  50),
    "tan":        (200, 170, 120),
    "gold":       (210, 170,  40),
    "olive":      (110, 130,  40),
    "maroon":     (130,  30,  30),
    "white":      (250, 250, 250),
    "light_gray": (190, 190, 190),
    "gray":       (120, 120, 120),
    "dark_gray":  ( 60,  60,  60),
    "black":      ( 20,  20,  20),
    "lavender":   (214, 197, 235),
}

COLOR_NAMES: dict[tuple[int, int, int], str] = {
    **{v: k for k, v in PALETTE.items()},
    **{v: k for k, v in STANDARD_PALETTE.items()},
    **{v: k for k, v in NONSTANDARD_PALETTE.items()},
}


def name_of(rgb: tuple[int, int, int]) -> str:
    return COLOR_NAMES.get(tuple(rgb), f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}")


def sample_colors(n: int, rng: _random.Random,
                  exclude: list[tuple] = None) -> list[tuple[int, int, int]]:
    """Sample n distinct colors from the palette, excluding listed RGB tuples."""
    pool = [c for c in PALETTE.values()
            if (exclude is None or tuple(c) not in {tuple(e) for e in exclude})]
    return rng.sample(pool, min(n, len(pool)))


def contrasting_color(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return black or white, whichever contrasts more with the given color."""
    luminance = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    return PALETTE["black"] if luminance > 128 else PALETTE["white"]


def lerp_color(c1: tuple, c2: tuple, t: float) -> tuple[int, int, int]:
    """Linearly interpolate between two RGB colors."""
    return tuple(int(round(c1[i] + (c2[i] - c1[i]) * t)) for i in range(3))


def rgb_to_hex(rgb: tuple) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)
