"""Cache invariants for src/eval.py.

Mirrors the inference-cache behaviour: per-problem sidecar JSONs at
``<results>/<model>/<benchmark>/<task>/<NNNN>_stats.json`` get reused
on rerun unless their mtime falls behind a source PNG, or
``--overwrite`` is passed. These tests use the trivial input-as-output
baseline (same as test_smoke_pipeline) so they're self-contained.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import gen_one

_ROOT = Path(__file__).resolve().parent.parent


def _save_problem(prob, task_dir: Path, pid: int):
    prefix = task_dir / f"{pid:03d}"
    prob.input_image.save(f"{prefix}_input.png")
    prob.answer_image.save(f"{prefix}_answer.png")
    with open(f"{prefix}.json", "w") as f:
        json.dump({"instruction": prob.instruction,
                   "task": "translation", "mode": "align", "visual_condition": "baseline",
                   "problem_id": pid}, f)


@pytest.fixture
def tiny_setup(tmp_path):
    """2-problem benchmark + trivial model outputs (input copy)."""
    bench_root   = tmp_path / "benchmarks"
    bench_dir    = bench_root / "PaintBench"
    task_dir     = bench_dir  / "translation"
    task_dir.mkdir(parents=True)

    for pid, seed in enumerate([42, 123]):
        prob = gen_one("translation", "align", seed)
        assert not prob.error
        _save_problem(prob, task_dir, pid)

    outputs_root = tmp_path / "results"
    model_task   = outputs_root / "trivial" / "PaintBench" / "translation"
    model_task.mkdir(parents=True)
    for pid in range(2):
        shutil.copy(task_dir / f"{pid:03d}_input.png",
                    model_task / f"{pid:04d}_output.png")

    results_root = tmp_path / "eval_results"
    return bench_root, outputs_root, results_root, model_task


def _run_eval(bench_root, outputs_root, results_root, *extra) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "src/eval.py",
         "--benchmarks",    str(bench_root),
         "--model-outputs", str(outputs_root),
         "--eval-outputs",  str(results_root),
         "--workers",       "1",
         *extra],
        cwd=_ROOT, capture_output=True, text=True, check=True,
    )


def _read_records(results_root: Path) -> list[dict]:
    return [json.loads(l) for l in
            (results_root / "problem_stats.jsonl").read_text().splitlines()]


def _sidecar(results_root: Path, idx: int) -> Path:
    return (results_root / "trivial" / "PaintBench" / "translation"
            / f"{idx:04d}_stats.json")


def test_first_run_writes_sidecars(tiny_setup):
    """First eval has no cache — all problems fresh, sidecars get written."""
    bench_root, outputs_root, results_root, _ = tiny_setup
    out = _run_eval(bench_root, outputs_root, results_root)

    assert "All " not in out.stdout, "first run should not report all-cached"
    assert "(cached: 0, recomputed: 2)" in out.stdout

    for pid in range(2):
        side = _sidecar(results_root, pid)
        assert side.exists(), f"sidecar missing: {side}"
        rec = json.loads(side.read_text())
        assert rec["idx"] == pid
        assert rec["model"] == "trivial"


def test_second_run_uses_cache(tiny_setup):
    """Second eval with no input changes: every problem cached, zero recomputation."""
    bench_root, outputs_root, results_root, _ = tiny_setup
    _run_eval(bench_root, outputs_root, results_root)

    first  = _read_records(results_root)
    sidecar_mtimes = [_sidecar(results_root, i).stat().st_mtime for i in range(2)]

    # Sleep so any rewrite would have a strictly newer mtime (mtime
    # resolution on macOS APFS is ~1ns but we want a clear signal even
    # on filesystems with second-resolution mtime).
    time.sleep(1.1)

    out = _run_eval(bench_root, outputs_root, results_root)
    assert "All 2 problems cached" in out.stdout
    assert "(cached: 2, recomputed: 0)" in out.stdout

    second = _read_records(results_root)
    assert first == second, "cached run must reproduce the same records"

    # Sidecars not rewritten (cache hit shouldn't touch them).
    for i in range(2):
        assert _sidecar(results_root, i).stat().st_mtime == sidecar_mtimes[i]


def test_overwrite_forces_recompute(tiny_setup):
    """--overwrite invalidates the sidecar cache and rewrites every record."""
    bench_root, outputs_root, results_root, _ = tiny_setup
    _run_eval(bench_root, outputs_root, results_root)
    first_mtimes = [_sidecar(results_root, i).stat().st_mtime for i in range(2)]

    time.sleep(1.1)

    out = _run_eval(bench_root, outputs_root, results_root, "--overwrite")
    assert "(cached: 0, recomputed: 2)" in out.stdout

    second_mtimes = [_sidecar(results_root, i).stat().st_mtime for i in range(2)]
    for f, s in zip(first_mtimes, second_mtimes):
        assert s > f, "--overwrite should refresh every sidecar"


def test_stale_output_invalidates_cache(tiny_setup):
    """Touching a model output PNG (post-eval) invalidates only that
    problem's cache; the other one stays cached."""
    bench_root, outputs_root, results_root, model_task = tiny_setup
    _run_eval(bench_root, outputs_root, results_root)

    time.sleep(1.1)
    # Bump only problem 0's output mtime — simulates a re-inference run.
    os.utime(model_task / "0000_output.png", None)

    out = _run_eval(bench_root, outputs_root, results_root)
    assert "(cached: 1, recomputed: 1)" in out.stdout, out.stdout


def test_corrupt_sidecar_falls_through(tiny_setup):
    """A truncated sidecar (size>0 + bad JSON) passes the cheap stat
    check in _is_cached but fails the json.loads in _load_cached_record;
    main() must promote those jobs back to fresh and report the count.
    Mirrors inference.py's _print_corrupt_cache_note for truncated
    output PNGs."""
    bench_root, outputs_root, results_root, _ = tiny_setup
    _run_eval(bench_root, outputs_root, results_root)

    # Truncate problem 0's sidecar: nonzero size, invalid JSON.
    side0 = _sidecar(results_root, 0)
    side0.write_text('{"model": "trivial", "be')
    assert side0.stat().st_size > 0, "size>0 needed to bypass the empty-file fast-path"

    out = _run_eval(bench_root, outputs_root, results_root)
    # Plan forecasts both as cached; load-time validation catches the
    # corruption and rebuilds the bad one. Both prints land in stdout.
    assert "All 2 problems cached" in out.stdout, out.stdout
    assert "1 forecast-cached sidecar was corrupt and rerun" in out.stdout, out.stdout

    # And the rebuilt sidecar is now valid and consistent with the
    # other one.
    records = _read_records(results_root)
    assert len(records) == 2
    assert all(r["model"] == "trivial" for r in records)


def test_default_eval_writes_diff_pngs(tiny_setup):
    """Default `make eval` writes the 4 ΔE diff PNGs that the visualizer
    Eval tab consumes — input/answer/output PNGs are NOT copied (they
    live in --benchmarks / --model-outputs, not eval_outputs/)."""
    bench_root, outputs_root, results_root, _ = tiny_setup
    _run_eval(bench_root, outputs_root, results_root)

    task_results = results_root / "trivial" / "PaintBench" / "translation"
    # Exactly the 4 IMAGE_THRESHOLDS — nothing for de1/de3/de4/de6/de7/de8/de9.
    for t in (0, 2, 5, 10):
        assert (task_results / f"0000_diff_cie76_{t}.png").exists(), f"missing diff_de_{t}"
    for t in (1, 3, 4, 6, 7, 8, 9):
        assert not (task_results / f"0000_diff_cie76_{t}.png").exists(), \
            f"unexpectedly wrote diff_de_{t} (only IMAGE_THRESHOLDS should be saved)"
    # No input/answer/output copies in eval_outputs/.
    assert not (task_results / "0000_input.png").exists()
    assert not (task_results / "0000_answer.png").exists()
    assert not (task_results / "0000_output.png").exists()
    # Trivial baseline (output == input) means O_norm is O_raw, so no
    # normalized PNG should be written.
    assert not (task_results / "0000_normalized_output.png").exists()


def test_no_save_images_flag(tiny_setup):
    """--no-save-images opts out of the diff-PNG cache; sidecars + JSONL
    still get written. Subsequent default-`make eval` rerun then sees
    the missing diff PNGs and recomputes everything."""
    bench_root, outputs_root, results_root, _ = tiny_setup
    _run_eval(bench_root, outputs_root, results_root, "--no-save-images")

    diff_png = (results_root / "trivial" / "PaintBench" / "translation"
                / "0000_diff_cie76_10.png")
    sidecar = _sidecar(results_root, 0)
    assert not diff_png.exists(), "--no-save-images: diff PNG shouldn't be written"
    assert sidecar.exists(),      "--no-save-images: sidecar still written"

    time.sleep(1.1)
    out = _run_eval(bench_root, outputs_root, results_root)
    # Default save-images=True needs the diff PNGs, which are missing → all rerun.
    assert "(cached: 0, recomputed: 2)" in out.stdout, out.stdout
    assert diff_png.exists()

    # Third run, also default: cache fully populated, nothing to do.
    out = _run_eval(bench_root, outputs_root, results_root)
    assert "All 2 problems cached" in out.stdout

    # And a --no-save-images rerun is also cached: a sidecar from a
    # prior save-images run is still good when the current run doesn't
    # need the diff PNGs.
    out = _run_eval(bench_root, outputs_root, results_root, "--no-save-images")
    assert "All 2 problems cached" in out.stdout


def test_missing_output_problems_cache_under_default(tiny_setup):
    """Regression: under the new ``--save-images`` default, problems
    with NO model output PNG must still cache on the sidecar mtime
    alone — the diff-PNG marker would never exist for them (the write
    loop in ``_process_one_problem`` is guarded by ``if O is not None``),
    so gating on it would force every ``make eval`` to recompute every
    missing-output problem.

    The bug this regression-tests: pre-fix, the ``_save_images_marker``
    was passed unconditionally when ``save_images=True``, so a partial
    inference run (some outputs present, others not) would burn full
    eval time on every rerun for the missing ones.
    """
    bench_root, outputs_root, results_root, model_task = tiny_setup
    # Simulate a partial inference: drop problem 1's output PNG.
    # Problem 0 keeps its trivial input-as-output; problem 1 has no output.
    (model_task / "0001_output.png").unlink()

    # First eval: problem 0 produces a diff-PNG marker, problem 1 does not.
    _run_eval(bench_root, outputs_root, results_root)
    side0 = _sidecar(results_root, 0)
    side1 = _sidecar(results_root, 1)
    assert side0.exists() and side1.exists()
    diff0 = (model_task.parent.parent.parent.parent / "eval_results" / "trivial" /
             "PaintBench" / "translation" / "0000_diff_cie76_10.png")
    diff1 = diff0.with_name("0001_diff_cie76_10.png")
    assert diff0.exists(), "problem 0 (with output) must have its diff-PNG marker"
    assert not diff1.exists(), "problem 1 (no output) must not have a diff-PNG"

    time.sleep(1.1)

    # Second eval: BOTH problems should cache. Pre-fix, problem 1 would
    # be recomputed because its missing diff-PNG marker tripped
    # ``_is_cached``. Post-fix, the marker isn't required when there's
    # no output to diff in the first place.
    out = _run_eval(bench_root, outputs_root, results_root)
    assert "All 2 problems cached" in out.stdout, (
        "missing-output problem should cache on sidecar mtime alone "
        f"(saw: {out.stdout!r})"
    )
    assert "(cached: 2, recomputed: 0)" in out.stdout
