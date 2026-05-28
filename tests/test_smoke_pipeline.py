"""End-to-end smoke: tiny benchmark → eval → stats → report.

Runs the full pipeline with 2 problems on a single task, using copies of
the input as fake "model outputs" (the trivial baseline). Fast and
catches integration regressions (file layouts, JSON schemas, CLI flags)
that unit tests miss.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import gen_one

_ROOT = Path(__file__).resolve().parent.parent


def _save_problem(prob, task_dir: Path, pid: int, extra_meta: dict):
    prefix = task_dir / f"{pid:03d}"
    prob.input_image.save(f"{prefix}_input.png")
    prob.answer_image.save(f"{prefix}_answer.png")
    # Match _save() in generate_benchmark.py — flatten metadata into the JSON
    with open(f"{prefix}.json", "w") as f:
        json.dump({"instruction": prob.instruction, **extra_meta}, f)


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    """Run a src/ CLI as a subprocess. Using subprocess (not import+main)
    because argparse scripts like to sys.exit on unknown args and we want
    real subprocess semantics for a true end-to-end test."""
    return subprocess.run(
        [sys.executable, *argv], cwd=_ROOT, capture_output=True, text=True, check=True,
    )


@pytest.fixture
def tiny_benchmark(tmp_path):
    """Build a 2-problem, 1-task minimal benchmark layout on disk.

    Uses the new folder-per-task format: folder = task name, mode and
    visual_condition stored in the per-problem JSON (not the folder name).
    """
    bench_root = tmp_path / "benchmarks"
    bench_dir  = bench_root / "PaintBench"
    task_dir   = bench_dir / "translation"   # new: folder = task name only
    task_dir.mkdir(parents=True)

    meta_list = []
    for pid, seed in enumerate([42, 123]):
        prob = gen_one("translation", "align", seed)
        assert not prob.error
        meta = {
            "task":       "translation", "mode": "align",
            "visual_condition":  "baseline",
            "problem_id": pid, "seed": seed, "n": 3, "W": 1024, "H": 1024,
        }
        _save_problem(prob, task_dir, pid, meta)
        meta_list.append({"task": "translation", "problem_id": pid,
                          "instruction": prob.instruction})

    with open(bench_dir / "problems.jsonl", "w") as f:
        for entry in meta_list:
            f.write(json.dumps(entry) + "\n")

    return bench_root


def test_pipeline_eval_stats_report(tiny_benchmark, tmp_path):
    """Copy each input as the model output (trivial baseline), then run
    eval → stats → report and verify each step lands the expected file."""
    bench_root   = tiny_benchmark
    outputs_root = tmp_path / "results"
    results_root = tmp_path / "eval_results"

    # Fake model outputs = copies of inputs (trivial "do nothing" baseline)
    model_task = outputs_root / "trivial" / "PaintBench" / "translation"
    model_task.mkdir(parents=True)
    src_task = bench_root / "PaintBench" / "translation"
    for pid in range(2):
        # eval.py expects 4-digit output filenames
        shutil.copy(src_task / f"{pid:03d}_input.png",
                    model_task / f"{pid:04d}_output.png")

    # --- eval ---
    _run([
        "src/eval.py",
        "--benchmarks",    str(bench_root),
        "--model-outputs", str(outputs_root),
        "--eval-outputs",  str(results_root),
        "--workers",       "1",
    ])
    problem_stats = results_root / "problem_stats.jsonl"
    assert problem_stats.exists()
    records = [json.loads(l) for l in problem_stats.read_text().splitlines()]
    assert len(records) == 2
    assert all(r["model"] == "trivial" for r in records)
    assert all(r["benchmark"] == "PaintBench" for r in records)

    # --- stats ---
    aggregate = results_root / "aggregate_stats.jsonl"
    _run([
        "src/stats.py",
        "--input",  str(problem_stats),
        "--output", str(aggregate),
    ])
    assert aggregate.exists()
    rows = [json.loads(l) for l in aggregate.read_text().splitlines()]
    levels = {r["level"] for r in rows}
    # translation/align is a single (task, mode) key → no mode row.
    # One visual_condition (baseline) → one visual_condition row.
    assert "task"      in levels
    assert "category"  in levels
    assert "visual_condition" in levels
    assert "benchmark" in levels

    cat_row = next(r for r in rows if r["level"] == "category")
    assert cat_row["category"] == "geometric_transformation"

    cond_row = next(r for r in rows if r["level"] == "visual_condition")
    assert cond_row["visual_condition"] == "baseline"

    # --- report ---
    report_path = tmp_path / "report.html"
    _run([
        "src/report.py",
        "--input",  str(aggregate),
        "--output", str(report_path),
    ])
    assert report_path.exists()
    # Report is self-contained HTML with the benchmark name in the title / body
    html = report_path.read_text()
    assert "PaintBench" in html
