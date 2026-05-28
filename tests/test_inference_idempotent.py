"""Tests for incremental inference reruns (cache-hit short-circuit).

Inference is incremental by default: existing output PNGs in
``--out-dir`` are reused after a Pillow ``.load()`` decode check, so
reruns after a cancel only redo the missing problems. ``--overwrite``
disables the short-circuit and unconditionally re-invokes the model.

The helper (``_build_skipped_result``) is the load-bearing piece —
once it correctly returns a synthetic result for a usable cached PNG
(and ``None`` for a missing / corrupt one), the orchestrators in both
the sync and async paths plumb it through unchanged. So unit tests on
the helper plus a regression for the not-corrupt-cache contract are
sufficient.

No model loading, no API calls — runs in a few ms."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

import inference
from benchmark_source import Problem


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_problem(tmp_path):
    """Build a minimal :class:`Problem` with a valid input PNG on disk.

    Mirrors the shape produced by
    :class:`benchmark_source.LocalBenchmarkSource` — pid, instruction,
    input/answer paths held lazily. Enough for ``_build_skipped_result``
    to read sizes via the cheap header-only path.
    """
    input_path = tmp_path / "000_input.png"
    Image.new("RGB", (32, 24), (10, 20, 30)).save(input_path)
    answer_path = tmp_path / "000_answer.png"
    Image.new("RGB", (32, 24), (40, 50, 60)).save(answer_path)
    return Problem(
        pid=0,
        task="recolor",
        mode="default",
        visual_condition="baseline",
        instruction="Recolor the red square to blue.",
        metadata={},
        _input=input_path,
        _answer=answer_path,
    )


def _write_png(path: Path, size=(32, 24), color=(200, 100, 50)) -> None:
    Image.new("RGB", size, color).save(path)


# ─── _build_skipped_result ───────────────────────────────────────────────────

def test_returns_none_when_no_cache(fake_problem, tmp_path):
    """No PNG on disk → fall through to model invocation."""
    out = tmp_path / "out" / "000_output.png"
    assert inference._build_skipped_result(fake_problem, out) is None


def test_returns_none_when_cache_is_empty_file(fake_problem, tmp_path):
    """A 0-byte PNG (write was killed before any data flushed) is invalid;
    fall through and redo. Catches the ``-rw-r--r--  0 ... 011_input.png``
    corruption mode that motivated the regenerate feature."""
    out = tmp_path / "out" / "000_output.png"
    out.parent.mkdir(parents=True)
    out.touch()
    assert out.stat().st_size == 0
    assert inference._build_skipped_result(fake_problem, out) is None


def test_returns_none_when_cache_is_truncated(fake_problem, tmp_path):
    """A partial PNG that has bytes but doesn't decode → fall through. This
    is the realistic 'killed mid-save' corruption — non-zero size, valid
    PNG header, but truncated body. Pillow's lazy loading misses this; we
    rely on .load() to force a real decode."""
    out = tmp_path / "out" / "000_output.png"
    out.parent.mkdir(parents=True)
    full = BytesIO()
    Image.new("RGB", (64, 64), (255, 0, 0)).save(full, format="PNG")
    # Write only the first ~third of the bytes (well past the header but
    # short of the IDAT terminator).
    out.write_bytes(full.getvalue()[: len(full.getvalue()) // 3])
    assert out.stat().st_size > 0
    assert inference._build_skipped_result(fake_problem, out) is None


def test_returns_skipped_dict_when_cache_is_valid(fake_problem, tmp_path):
    """Healthy cached PNG → synthetic result with ``skipped=True``,
    ``success=True``, and ``inference_time_s=None`` (so summary timing
    means filter it out via the ``is not None`` predicate)."""
    out = tmp_path / "out" / "000_output.png"
    out.parent.mkdir(parents=True)
    _write_png(out, size=(48, 36))

    result = inference._build_skipped_result(fake_problem, out)
    assert result is not None
    assert result["success"] is True
    assert result["skipped"] is True
    assert result["inference_time_s"] is None
    assert result["index"] == 0
    assert result["instruction"] == "Recolor the red square to blue."
    # Identifier tuple lets downstream tooling locate the source problem
    # without needing the local path (HF source has no on-disk path).
    assert result["task"] == "recolor"
    assert result["mode"] == "default"
    assert result["visual_condition"] == "baseline"
    assert result["output_path"] == str(out)
    assert result["input_size_wh"] == [32, 24]
    assert result["output_size_wh"] == [48, 36]


def test_skipped_result_picks_up_reasoning_text_when_present(fake_problem, tmp_path):
    """If a sibling .txt with the model's reasoning trace was saved
    alongside the PNG (Gemini models do this), surface it. Without the
    .txt, leave reasoning fields as None."""
    out = tmp_path / "out" / "000_output.png"
    out.parent.mkdir(parents=True)
    _write_png(out)

    reasoning_path = out.with_suffix(".txt")
    reasoning_path.write_text("[REASONING] swap red→blue", encoding="utf-8")

    result = inference._build_skipped_result(fake_problem, out)
    assert result is not None
    assert result["reasoning_text"] == "[REASONING] swap red→blue"
    assert result["reasoning_path"] == str(reasoning_path)


def test_skipped_result_handles_missing_reasoning_text(fake_problem, tmp_path):
    """Most models don't emit a reasoning trace — those runs leave the
    .txt absent. The helper should still succeed with reasoning fields
    set to None."""
    out = tmp_path / "out" / "000_output.png"
    out.parent.mkdir(parents=True)
    _write_png(out)

    result = inference._build_skipped_result(fake_problem, out)
    assert result is not None
    assert result["reasoning_text"] is None
    assert result["reasoning_path"] is None


def test_skipped_result_returns_none_when_input_missing(tmp_path):
    """If the *input* PNG is gone (benchmark moved/deleted) we can't
    reconstruct ``input_size_wh``, so fall through. This is a degenerate
    case — the run would fail anyway — but the helper shouldn't raise."""
    # No W/H in metadata → input_size_wh falls back to header-read on the
    # missing file and raises, which the helper catches.
    problem = Problem(
        pid=0,
        task="x", mode="", visual_condition="",
        instruction="x",
        metadata={},
        _input=tmp_path / "missing_input.png",
        _answer=None,
    )
    out = tmp_path / "out" / "000_output.png"
    out.parent.mkdir(parents=True)
    _write_png(out)
    assert inference._build_skipped_result(problem, out) is None


def test_skipped_result_excluded_from_timing_means(fake_problem, tmp_path):
    """The ``_mean(all_times)`` summary in main filters via
    ``inference_time_s is not None``. Skipped results set it to None so
    they don't poison the mean (a skipped 0.0s would tank the avg and
    make a resumed run look much faster than the original)."""
    out = tmp_path / "out" / "000_output.png"
    out.parent.mkdir(parents=True)
    _write_png(out)

    result = inference._build_skipped_result(fake_problem, out)
    assert result is not None and result["inference_time_s"] is None

    # Mirror the comprehension in save_metrics: skipped should be filtered.
    fake_problems = [
        {"inference_time_s": 1.0},
        {"inference_time_s": 2.0},
        result,
    ]
    times = [p["inference_time_s"] for p in fake_problems if p.get("inference_time_s") is not None]
    assert times == [1.0, 2.0]
    assert inference._mean(times) == 1.5
