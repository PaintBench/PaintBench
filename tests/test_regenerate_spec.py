"""Tests for the targeted-regeneration helpers in generate_benchmark.py.

Covers:
  * the spec-parser (`_parse_regenerate_spec`)
  * task registry invariants (unique names, n-level coverage)
  * the end-to-end replay path (`_regenerate_problems`) — pins the
    "byte-identical output by recorded seed" contract that's the core
    claim of the feature.
"""
from __future__ import annotations

import importlib
import json
import os

import pytest

import generate_benchmark as gb


# ─── _parse_regenerate_spec ──────────────────────────────────────────────────

def test_parse_single_pair():
    assert gb._parse_regenerate_spec("translation:11") == [("translation", 11)]


def test_parse_multiple_pairs():
    assert gb._parse_regenerate_spec("a:1,b:2,c:3") == [("a", 1), ("b", 2), ("c", 3)]


def test_parse_tolerates_whitespace():
    assert gb._parse_regenerate_spec(" a : 1 , b: 2") == [("a", 1), ("b", 2)]


def test_parse_skips_empty_entries():
    """Trailing/leading commas + double-commas shouldn't produce phantom pairs."""
    assert gb._parse_regenerate_spec(",a:1,,b:2,") == [("a", 1), ("b", 2)]


def test_parse_task_name_with_underscores():
    """Task names can contain underscores (e.g. point_operations, flood_fill).
    The id split must use the LAST colon, not the first."""
    assert gb._parse_regenerate_spec("point_operations:7") == [("point_operations", 7)]


def test_parse_rejects_missing_colon():
    with pytest.raises(ValueError, match="missing ':'"):
        gb._parse_regenerate_spec("just_a_task")


def test_parse_rejects_non_integer_id():
    with pytest.raises(ValueError, match="id must be an integer"):
        gb._parse_regenerate_spec("blending:five")


# ─── Task registry ────────────────────────────────────────────────────────────

def test_task_names_are_unique():
    """Task names double as folder names, so duplicates would silently clobber output."""
    names = [t for _, t in gb.TASKS]
    assert len(names) == len(set(names))


def test_n_levels_for_all_tasks():
    """Every task must resolve to a 4-element list of positive integers."""
    for _, task_name in gb.TASKS:
        levels = gb._n_levels_for(task_name)
        assert len(levels) == 4, f"{task_name}: expected 4 n-levels, got {len(levels)}"
        assert all(isinstance(n, int) and n >= 1 for n in levels), \
            f"{task_name}: non-positive n in levels {levels}"


# ─── _regenerate_problems end-to-end ─────────────────────────────────────────

def _seed_one_problem(out_dir, task_name="blending", pid=0):
    """Generate a single problem the canonical way, returning the byte
    contents of its three artefacts so the test can compare."""
    modes    = importlib.import_module(f"tasks.{task_name}").PARAMETERS.get("mode") or [None]
    mode     = modes[0]
    cond     = gb.VISUAL_CONDITIONS[0]  # baseline
    seed     = gb._make_seed(task_name, cond["name"], mode or "", pid, attempt=0)
    task_dir = os.path.join(out_dir, task_name)
    os.makedirs(task_dir, exist_ok=True)
    result = gb._render_main((task_name, mode, task_dir, pid, seed, cond))
    assert result is not None, f"could not generate test problem ({task_name}/{pid})"
    prefix = os.path.join(task_dir, f"{pid:03d}")
    return {
        "input":  open(f"{prefix}_input.png",  "rb").read(),
        "answer": open(f"{prefix}_answer.png", "rb").read(),
        "json":   open(f"{prefix}.json").read(),
    }


def test_regenerate_problems_is_byte_identical(tmp_path):
    """Replay path: corrupt input.png → call _regenerate_problems → BOTH
    input.png and answer.png match the original bytes exactly. Pins the
    headline "byte-identical output by recorded seed" contract for both
    artefacts (any non-determinism in answer rendering would slip past
    an input-only check)."""
    task_name, pid = "blending", 0
    out_dir = str(tmp_path / "PaintBench")
    original = _seed_one_problem(out_dir, task_name, pid)

    # Simulate the wild-corruption case: truncate input.png to 0 bytes.
    input_path  = tmp_path / "PaintBench" / task_name / f"{pid:03d}_input.png"
    answer_path = tmp_path / "PaintBench" / task_name / f"{pid:03d}_answer.png"
    input_path.write_bytes(b"")
    assert input_path.stat().st_size == 0

    n_ok = gb._regenerate_problems(out_dir, [(task_name, pid)])

    assert n_ok == 1
    assert input_path.read_bytes()  == original["input"]
    assert answer_path.read_bytes() == original["answer"]


def test_regenerate_problems_skips_unknown_task(tmp_path, capsys):
    """Logic check fires before the disk hit: an unknown task name produces an
    'unknown task' message, not a misleading 'no .json' message."""
    n_ok = gb._regenerate_problems(str(tmp_path), [("definitely_not_a_task", 5)])
    assert n_ok == 0
    out = capsys.readouterr().out
    assert "unknown task" in out
    assert "no .json" not in out


def test_regenerate_problems_skips_missing_json(tmp_path, capsys):
    """A real task name but no problem at that index: clean SKIP, not crash."""
    task_name = "blending"
    (tmp_path / task_name).mkdir()
    n_ok = gb._regenerate_problems(str(tmp_path), [(task_name, 999)])
    assert n_ok == 0
    assert "no .json" in capsys.readouterr().out


def test_regenerate_problems_skips_unknown_visual_condition(tmp_path, capsys):
    """If the stored visual_condition name is not in VISUAL_CONDITIONS, skip gracefully."""
    task_name, pid = "blending", 0
    out_dir = str(tmp_path / "PaintBench")
    _seed_one_problem(out_dir, task_name, pid)

    json_path = tmp_path / "PaintBench" / task_name / f"{pid:03d}.json"
    meta = json.loads(json_path.read_text())
    meta["visual_condition"] = "obsolete_visual_condition"
    json_path.write_text(json.dumps(meta))

    n_ok = gb._regenerate_problems(out_dir, [(task_name, pid)])
    assert n_ok == 0
    assert "unknown visual_condition" in capsys.readouterr().out
