"""CIE76 math and stats on synthetic inputs. Tests the invariants that
make the benchmark meaningful: identity → perfect score, random noise →
imperfect score, and the pixel accounting adds up."""
from __future__ import annotations

import numpy as np

from eval import compute_problem_stats, normalize_output, _rgb_to_lab
from PIL import Image


def _rand_img(w=32, h=32, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_identity_problem_perfect_score():
    """I == A and O == A → every edit_accuracy + preservation_accuracy = 1.0"""
    I = _rand_img(seed=1)
    A = I.copy()
    O = I.copy()
    stats, _, _ = compute_problem_stats(I, A, O)
    # Identity problem: nothing changes, so edit_pixels=0, preservation_pixels=all
    assert stats["edit_pixels"] == 0
    assert stats["preservation_pixels"] == 32 * 32
    # By convention (see eval.py) edit_accuracy=1.0 when edit_pixels=0 — the
    # model "succeeded" at the zero required edits.
    for t in range(11):
        cell = stats["cie76_threshold"][str(t)]
        assert cell["edit_accuracy"]         == 1.0
        assert cell["preservation_accuracy"] == 1.0


def test_perfect_output_matches_answer():
    """I != A but O == A → edit and preservation accuracies both 1.0."""
    I = _rand_img(seed=1)
    A = _rand_img(seed=2)
    O = A.copy()
    stats, _, _ = compute_problem_stats(I, A, O)
    for t in range(11):
        cell = stats["cie76_threshold"][str(t)]
        assert cell["edit_accuracy"]         == 1.0
        assert cell["preservation_accuracy"] == 1.0


def test_unchanged_output_has_zero_edit_accuracy():
    """I != A but O == I (model did nothing) → edit_accuracy = 0,
    preservation_accuracy = 1 (everything that should have stayed did)."""
    I = _rand_img(seed=1)
    A = _rand_img(seed=2)
    O = I.copy()
    stats, _, _ = compute_problem_stats(I, A, O)
    # edit_pixels > 0 by construction (different random images)
    assert stats["edit_pixels"] > 0
    cell0 = stats["cie76_threshold"]["0"]
    assert cell0["edit_accuracy"] == 0.0
    assert cell0["preservation_accuracy"] == 1.0


def test_pixel_accounting_adds_up():
    """edit_pixels + preservation_pixels == H * W for every image."""
    I = _rand_img(seed=3)
    A = _rand_img(seed=4)
    O = _rand_img(seed=5)
    stats, _, _ = compute_problem_stats(I, A, O)
    assert stats["edit_pixels"] + stats["preservation_pixels"] == 32 * 32


def test_normalize_output_passthrough():
    """Output already matches answer size → identity pass-through."""
    A = Image.fromarray(_rand_img(w=64, h=48))
    O = Image.fromarray(_rand_img(w=64, h=48))
    assert normalize_output(O, A).size == A.size


def test_normalize_output_resizes_to_match():
    A = Image.fromarray(_rand_img(w=32, h=32))
    O = Image.fromarray(_rand_img(w=128, h=96))
    norm = normalize_output(O, A)
    assert norm.size == A.size


def test_rgb_to_lab_shape_and_identity():
    """Lab conversion is shape-preserving; black → L=0."""
    I = np.zeros((4, 4, 3), dtype=np.uint8)
    lab = _rgb_to_lab(I)
    assert lab.shape == (4, 4, 3)
    # Black: L=0, a=0, b=0
    assert np.allclose(lab, 0.0, atol=1e-5)
