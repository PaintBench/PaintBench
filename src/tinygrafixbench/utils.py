"""Shared helpers for TinyGrafixBench generators.

Each generator in this package is a deterministic function of a single integer
seed. To keep font rendering identical across machines, we copy matplotlib's
bundled DejaVuSans.ttf into the repo on first import and register it.
"""

import os
import shutil
import string

import matplotlib
import matplotlib.font_manager as fm
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(_HERE, "fonts")
FONT_PATH = os.path.join(FONT_DIR, "DejaVuSans.ttf")


def _ensure_font():
    if not os.path.exists(FONT_PATH):
        os.makedirs(FONT_DIR, exist_ok=True)
        src = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
        shutil.copyfile(src, FONT_PATH)
    fm.fontManager.addfont(FONT_PATH)
    matplotlib.rcParams["font.family"] = "DejaVu Sans"


_ensure_font()


def make_rng(seed):
    return np.random.default_rng(int(seed))


def round_sig(x, n=3):
    """Round x to n significant figures, as a float. Used to keep instruction
    text and answer rendering exactly consistent: any value printed with
    `:.{n}g` should first pass through this so the stored value equals the
    displayed one."""
    if x == 0:
        return 0.0
    return float(f"{x:.{n}g}")


def random_magnitude(rng, min_exp=-3, max_exp=3):
    """Pick an order-of-magnitude multiplier (a power of ten) for a problem's
    numerical ranges. Defaults span 0.001–1000× the chart's base scale."""
    k = int(rng.integers(min_exp, max_exp + 1))
    return 10.0 ** k


def random_string(rng, min_len=3, max_len=8):
    n = int(rng.integers(min_len, max_len + 1))
    alphabet = list(string.ascii_letters)
    return "".join(rng.choice(alphabet, size=n))


def random_title(rng, min_words=1, max_words=3, min_word_len=3, max_word_len=8):
    """Multi-word title: 1–3 space-separated gibberish words."""
    n_words = int(rng.integers(min_words, max_words + 1))
    return " ".join(random_string(rng, min_word_len, max_word_len) for _ in range(n_words))


def random_axis_label(rng, min_words=1, max_words=3, min_word_len=3, max_word_len=8):
    """1–3 word axis label."""
    n_words = int(rng.integers(min_words, max_words + 1))
    return " ".join(random_string(rng, min_word_len, max_word_len) for _ in range(n_words))


# sRGB D65 conversion matrices (Bradford-adapted, IEC 61966-2-1).
_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)
_D65 = np.array([0.95047, 1.0, 1.08883])


def _srgb_to_linear(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * (np.abs(c) ** (1.0 / 2.4)) - 0.055)


def _xyz_to_lab(xyz):
    t = np.asarray(xyz, dtype=float) / _D65
    delta = 6.0 / 29.0
    f = np.where(t > delta ** 3, np.cbrt(t), t / (3.0 * delta ** 2) + 4.0 / 29.0)
    return np.array([116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), 200.0 * (f[1] - f[2])])


def _lab_to_xyz(lab):
    L, a, b = (float(x) for x in lab)
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    delta = 6.0 / 29.0
    f = np.array([fx, fy, fz])
    t = np.where(f > delta, f ** 3, 3.0 * delta ** 2 * (f - 4.0 / 29.0))
    return t * _D65


def _rgb_to_lab(rgb):
    return _xyz_to_lab(_RGB_TO_XYZ @ _srgb_to_linear(rgb))


def _lab_to_rgb(lab):
    return _linear_to_srgb(_XYZ_TO_RGB @ _lab_to_xyz(lab))


def _delta_e76(lab1, lab2):
    return float(np.linalg.norm(np.asarray(lab1) - np.asarray(lab2)))


def rgb_to_hex(c):
    """Format an RGB tuple in [0, 1]^3 as a `#RRGGBB` string."""
    r, g, b = (int(round(x * 255)) for x in c)
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def random_color(rng, avoid=(), min_delta_e=20.0, max_attempts=2000):
    """Sample a color uniformly in CIE Lab, rejecting out-of-sRGB-gamut draws
    and any draw within `min_delta_e` (ΔE76) of every color in `avoid`.

    `avoid` is an iterable of RGB tuples in [0,1]^3. Returns an RGB tuple.
    """
    avoid_lab = [_rgb_to_lab(c) for c in avoid]
    tol = 1e-6
    for _ in range(max_attempts):
        lab = np.array([
            rng.uniform(0.0, 100.0),
            rng.uniform(-100.0, 100.0),
            rng.uniform(-100.0, 100.0),
        ])
        rgb = _lab_to_rgb(lab)
        if np.any(rgb < -tol) or np.any(rgb > 1.0 + tol):
            continue
        if any(_delta_e76(lab, la) <= min_delta_e for la in avoid_lab):
            continue
        return tuple(float(x) for x in np.clip(rgb, 0.0, 1.0))
    raise RuntimeError(
        f"random_color: could not satisfy gamut + ΔE76>{min_delta_e} "
        f"against {len(avoid)} colors in {max_attempts} attempts"
    )


def random_palette(rng, n, avoid=(), min_delta_e=20.0):
    """Sample n colors with pairwise ΔE76 > `min_delta_e`, and each kept color
    also >min_delta_e from every color in `avoid` (typically the background)."""
    result = []
    for _ in range(n):
        result.append(random_color(
            rng, avoid=list(avoid) + result, min_delta_e=min_delta_e,
        ))
    return result


def random_bg_and_text(rng):
    """Return (bg_rgb, text_rgb) with enough luminance contrast."""
    if rng.random() < 0.3:
        bg = rng.random(3) * 0.35
        text = rng.random(3) * 0.3 + 0.7
    else:
        bg = rng.random(3) * 0.35 + 0.6
        text = rng.random(3) * 0.3
    return tuple(float(x) for x in bg), tuple(float(x) for x in text)


# Font sizes used across all generators. Title is large enough to dominate,
# axis labels sit below it, tick labels smallest. Tuned for the 768x576
# canvas used by every generator: these give a clear visual hierarchy while
# remaining comfortably readable at full resolution.
FONT_SIZES = {
    "title": 18,
    "axis_label": 14,
    "tick": 11,
    "legend": 11,
}

# Constrained-layout padding (inches) around the axes. Default is 4/72 ≈
# 0.0556" — too tight to breathe on a 768x576 canvas. 0.15" gives a clear
# rim on all four sides without shrinking the plot area meaningfully.
LAYOUT_PAD = 0.15
TITLE_PAD = 10  # pt; default is 6


def tighten_layout(fig):
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(w_pad=LAYOUT_PAD, h_pad=LAYOUT_PAD)


def apply_theme(fig, ax, bg, text):
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_color(text)
    ax.tick_params(colors=text, which="both", labelsize=FONT_SIZES["tick"])
    ax.xaxis.label.set_color(text)
    ax.xaxis.label.set_size(FONT_SIZES["axis_label"])
    ax.yaxis.label.set_color(text)
    ax.yaxis.label.set_size(FONT_SIZES["axis_label"])
    ax.title.set_color(text)
    ax.set_title(ax.get_title(), pad=TITLE_PAD, fontsize=FONT_SIZES["title"])
    tighten_layout(fig)
