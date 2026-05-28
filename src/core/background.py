"""Background image generator with waveform band support.

Supports single colour, stripes, or one of four waveforms
(sine, square, triangle, sawtooth) at arbitrary rotation and amplitude.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from PIL import Image


@dataclass
class BackgroundSpec:
    """Specification for background rendering.

    colors    : list of RGB tuples (1–4).  Single entry → solid colour.
    band_width: fraction of canvas width per colour band (relative, 0–1).
    waveform  : boundary waveform — "line" | "sine" | "square" |
                "triangle" | "sawtooth".
    rotation  : degrees — direction of band boundaries (0° = vertical bands;
                90° = horizontal bands).
    amplitude : fraction of canvas width — peak waveform displacement.
    frequency : cycles of the waveform per canvas width along the band axis.
    """
    colors:     list          # list of RGB tuples
    band_width: float = 0.08  # relative to canvas width
    waveform:   str   = "line"
    rotation:   float = 0.0
    amplitude:  float = 0.0
    frequency:  float = 1.0


def make_background(width: int, height: int, spec: BackgroundSpec) -> Image.Image:
    """Render a background image from a BackgroundSpec."""
    n_colors = len(spec.colors)
    if n_colors == 1:
        return Image.new("RGB", (width, height), spec.colors[0])

    # Build pixel-coordinate grids
    xs = np.arange(width,  dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    px, py = np.meshgrid(xs, ys)   # shape (H, W)

    cx, cy = width / 2.0, height / 2.0
    angle_rad = math.radians(spec.rotation)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # d: signed distance along the band-normal direction (perpendicular to bands)
    # s: coordinate along the band axis
    d = (px - cx) * cos_a + (py - cy) * sin_a
    s = -(px - cx) * sin_a + (py - cy) * cos_a

    band_px = spec.band_width * width
    amp_px  = spec.amplitude  * width

    # Waveform shift applied per pixel along s
    if amp_px > 0 and spec.waveform != "line":
        phase = spec.frequency * s / width  # shape (H, W)
        t = phase % 1.0
        wf = spec.waveform
        if wf == "sine":
            shift = amp_px * np.sin(2 * math.pi * t)
        elif wf == "square":
            shift = np.where(t < 0.5, amp_px, -amp_px).astype(np.float32)
        elif wf == "triangle":
            shift = amp_px * np.where(
                t < 0.5,
                1.0 - 4.0 * np.abs(t - 0.25),
                -1.0 + 4.0 * np.abs(t - 0.75),
            )
        else:  # sawtooth
            shift = amp_px * (2.0 * t - 1.0)
        d_eff = d - shift
    else:
        d_eff = d

    # Assign each pixel to a band index → color index
    # numpy % follows Python convention (result ≥ 0 when divisor > 0)
    band_idx  = np.floor(d_eff / max(band_px, 1e-6)).astype(np.int32)
    color_idx = band_idx % n_colors

    # Build the output image from the color index array
    palette = np.array(spec.colors, dtype=np.uint8)   # (n_colors, 3)
    rgb = palette[color_idx]                           # (H, W, 3)
    return Image.fromarray(rgb, mode="RGB")
