"""Shared test fixtures and helpers.

The tests live outside src/ but most need to import from there. Pytest's
`pythonpath = ["src"]` (pyproject.toml) handles that.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

# Same default W/H/N as src/generate_benchmark.py. Tests should never need
# to change these; if the benchmark's base config changes, update here
# consciously.
CANVAS_W = 1024
CANVAS_H = 1024
N_SHAPES = 3


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def gen_one(task_name: str, mode: str | None, seed: int):
    """Generate a single problem — mirrors _render_main / _gen from
    generate_benchmark.py, but inline so tests don't depend on private
    helpers they don't care about."""
    import generate_benchmark as gb  # via pyproject pythonpath=["src"]
    from core.background import BackgroundSpec
    from core.colors import STANDARD_PALETTE

    mod = importlib.import_module(f"tasks.{task_name}")
    bg_rgb, _, obj_colors = gb._color_split(STANDARD_PALETTE, seed)
    bg_spec = BackgroundSpec(colors=[bg_rgb])
    kwargs: dict = {"n_min": N_SHAPES, "n_max": N_SHAPES}
    if mode is not None:
        kwargs["mode"] = mode
    return mod.generate(
        seed=seed, bg_spec=bg_spec, W=CANVAS_W, H=CANVAS_H,
        obj_colors=obj_colors, **kwargs,
    )
