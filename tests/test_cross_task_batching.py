"""Tests for cross-task batching.

Pre-fix (per-task ``asyncio.gather``): with N tasks each holding 1 to-run
problem of latency T, total wall is N×T because every task waits on its
own gather barrier before the next task's coros even start.

Post-fix (one global ``Semaphore`` + ``asyncio.as_completed``): all N
problems are in flight together; total wall is ~max(latencies), bounded
by ``concurrency``. For N tasks with concurrency >= N, that's ~T.

These tests are integration-level — they exercise ``_run_global_async``
directly with a fake evaluator that ``await asyncio.sleep(T)``s instead
of calling a real model. No model loads, no API calls, no subprocess.
Total runtime is dominated by the sleeps the test itself sets (kept
small).
"""
from __future__ import annotations

import asyncio
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple

import pytest
from PIL import Image

import inference
from benchmark_source import Problem


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_problem(tmp_path: Path, task_id: str, idx: int) -> Tuple[str, Problem, Path, bool]:
    """Build a (task_id, problem, output_path, is_cached) plan entry. Writes
    the input PNG to disk so generate_async can open it."""
    task_dir = tmp_path / "bench" / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    input_path = task_dir / f"{idx:03d}_input.png"
    Image.new("RGB", (16, 12), (10 + idx, 20, 30)).save(input_path)

    problem = Problem(
        pid=idx,
        task=task_id,
        mode="",
        visual_condition="",
        instruction=f"Edit instruction for {task_id}/{idx}",
        metadata={},
        _input=input_path,
        _answer=None,
    )

    out_dir = tmp_path / "out" / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{idx:04d}_output.png"

    return (task_id, problem, output_path, False)  # not cached


class _SleepyEvaluator:
    """Fake evaluator with ``generate_async`` that sleeps for a configured
    duration, so we can prove parallelism timing without real models."""

    def __init__(self, *, concurrency: int, latency_s: float):
        self.concurrency = concurrency
        self.latency_s = latency_s
        self.MODEL_ID = "sleepy-test"

    def load_model(self) -> None:
        pass

    async def generate_async(self, image: Image.Image, instruction: str) -> Tuple[Image.Image, str]:
        await asyncio.sleep(self.latency_s)
        # Return the input as a stand-in output (orchestrator only saves it).
        buf = BytesIO()
        image.save(buf, format="PNG")
        return Image.open(BytesIO(buf.getvalue())), f"slept {self.latency_s}s"


class _PerTaskLatencyEvaluator:
    """Like _SleepyEvaluator but the latency depends on the instruction's
    leading task name. Models the real workload where some tasks are slow
    and some are fast — the pre-fix bug was that the per-task barrier
    serialised the slow-task tail across the whole run."""

    def __init__(self, *, concurrency: int, latencies: Dict[str, float]):
        self.concurrency = concurrency
        self.latencies = latencies
        self.MODEL_ID = "per-task-latency"

    def load_model(self) -> None:
        pass

    async def generate_async(self, image: Image.Image, instruction: str) -> Tuple[Image.Image, str]:
        # Lookup task by prefix in instruction.
        for task, lat in self.latencies.items():
            if instruction.startswith(f"Edit instruction for {task}/"):
                await asyncio.sleep(lat)
                buf = BytesIO()
                image.save(buf, format="PNG")
                return Image.open(BytesIO(buf.getvalue())), f"slept {lat}s ({task})"
        raise AssertionError(f"unmapped instruction: {instruction!r}")


# ─── Wall-time tests ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("n_tasks", [3, 6])
def test_n_tasks_one_problem_each_runs_in_parallel(tmp_path, n_tasks):
    """Pre-fix: N tasks × 1 problem × latency T → wall ≈ N*T (per-task barrier).
    Post-fix: same shape → wall ≈ T (single batch under one semaphore).

    Lock the invariant by asserting wall < (n_tasks * latency * 0.6). Generous
    upper bound (0.6) to absorb test scheduler jitter while still catching a
    re-introduction of the per-task barrier (which would be ~n_tasks * latency).
    """
    latency_s = 0.20
    plan = [
        _make_problem(tmp_path, f"task_{i}", 0)
        for i in range(n_tasks)
    ]
    evaluator = _SleepyEvaluator(concurrency=n_tasks, latency_s=latency_s)

    async def _drive():
        results = []
        async for task_id, result in inference._run_global_async(evaluator, plan):
            results.append((task_id, result))
        return results

    t0 = time.perf_counter()
    results = asyncio.run(_drive())
    wall = time.perf_counter() - t0

    assert len(results) == n_tasks
    assert all(r[1].get("success") for r in results)

    # The pre-fix would have been ~n_tasks * latency_s. We expect ~latency_s.
    pre_fix_lower_bound = n_tasks * latency_s * 0.6
    assert wall < pre_fix_lower_bound, (
        f"Expected ~{latency_s}s wall (single batch, {n_tasks} tasks in parallel) "
        f"but got {wall:.3f}s — looks like the per-task gather barrier is back."
    )
    # And it must take at least ~latency_s (one round of sleeps).
    assert wall >= latency_s * 0.8


def test_slow_task_does_not_block_fast_tasks(tmp_path):
    """The motivating case: a slow task in the middle of the schedule
    shouldn't gate the fast tasks before/after it. Post-fix all tasks
    complete in ~max(latency); pre-fix it'd be sum-of-maxes."""
    latencies = {
        "fast_a": 0.05,
        "slow_mid": 0.30,
        "fast_b": 0.05,
    }
    plan = [
        _make_problem(tmp_path, task, 0)
        for task in latencies
    ]
    evaluator = _PerTaskLatencyEvaluator(concurrency=10, latencies=latencies)

    async def _drive():
        results = []
        async for task_id, result in inference._run_global_async(evaluator, plan):
            results.append((task_id, result))
        return results

    t0 = time.perf_counter()
    results = asyncio.run(_drive())
    wall = time.perf_counter() - t0

    assert len(results) == 3
    # Wall must be close to the slow task's latency, not sum-of-maxes.
    # Phrased as "halfway between max and sum" to absorb scheduler jitter
    # while still catching a re-introduction of the per-task barrier
    # (which would push wall toward sum-of-maxes).
    max_lat = max(latencies.values())
    sum_lat = sum(latencies.values())
    overhead = wall - max_lat
    barrier_excess = sum_lat - max_lat
    assert overhead < barrier_excess * 0.5, (
        f"Pre-fix wall would be ~{sum_lat}s (per-task barrier); "
        f"post-fix should be ~{max_lat}s (one global pool). Got {wall:.3f}s, "
        f"overhead {overhead:.3f}s vs barrier-excess budget "
        f"{barrier_excess * 0.5:.3f}s."
    )
    assert wall >= max_lat * 0.8


def test_concurrency_bound_respected(tmp_path):
    """``Semaphore(concurrency)`` must still cap in-flight work. With 6
    problems × latency T and concurrency=2, wall ≈ 3*T (3 batches of 2),
    not T (would imply unbounded parallelism)."""
    latency_s = 0.10
    plan = [
        _make_problem(tmp_path, f"task_{i}", 0)
        for i in range(6)
    ]
    evaluator = _SleepyEvaluator(concurrency=2, latency_s=latency_s)

    async def _drive():
        results = []
        async for task_id, result in inference._run_global_async(evaluator, plan):
            results.append((task_id, result))
        return results

    t0 = time.perf_counter()
    results = asyncio.run(_drive())
    wall = time.perf_counter() - t0

    assert len(results) == 6
    # 3 batches of 2 → wall ≈ 3*latency. Allow 25% jitter band.
    assert 3 * latency_s * 0.75 < wall < 3 * latency_s * 1.5, (
        f"Expected ~{3*latency_s}s for 3 batches of concurrency=2, got {wall:.3f}s."
    )


def test_completion_order_independent_of_problem_index(tmp_path):
    """The first-completed problem isn't necessarily the lowest-index one.
    With per-problem latencies (10s, 5s, 1s) on indices (0, 1, 2), results
    yield in order 2, 1, 0 — but ``_finalize_task_results`` re-sorts by
    index so the metrics JSON stays deterministic across runs."""
    latencies_by_idx = {0: 0.20, 1: 0.10, 2: 0.05}

    plan = []
    for idx, _lat in latencies_by_idx.items():
        plan.append(_make_problem(tmp_path, "ordering", idx))

    class _IdxLatency:
        concurrency = 10
        MODEL_ID = "idx-latency"
        def load_model(self): pass
        async def generate_async(self, image, instruction):
            idx = int(instruction.rsplit("/", 1)[-1])
            await asyncio.sleep(latencies_by_idx[idx])
            buf = BytesIO()
            image.save(buf, format="PNG")
            return Image.open(BytesIO(buf.getvalue())), None

    evaluator = _IdxLatency()
    task_results: Dict = {"ordering": {"task_id": "ordering", "problems": []}}

    async def _drive():
        async for task_id, result in inference._run_global_async(evaluator, plan):
            task_results[task_id]["problems"].append(result)

    asyncio.run(_drive())

    # Completion order must be 2, 1, 0 (fastest → slowest). Verify directly:
    pre_sort = [r["index"] for r in task_results["ordering"]["problems"]]
    assert pre_sort == [2, 1, 0], (
        f"Completion order should follow latency, got {pre_sort}"
    )

    # ``_finalize_task_results`` then sorts by index for deterministic JSON.
    inference._finalize_task_results(task_results)
    post_sort = [r["index"] for r in task_results["ordering"]["problems"]]
    assert post_sort == [0, 1, 2], "_finalize_task_results must sort by index"


# ─── Plan / summary tests ────────────────────────────────────────────────────

def test_build_global_plan_applies_max_problems(tmp_path):
    """``--max-problems`` truncation happens in the plan, not at the runner.
    Plan should hold exactly ``max_problems`` per task."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def _p(task: str, i: int) -> Problem:
        return Problem(
            pid=i, task=task, mode="", visual_condition="",
            instruction=f"x{i}", metadata={},
            _input=Path("p"), _answer=None,
        )

    tasks = {
        "blending": [_p("blending", i) for i in range(5)],
        "rotation": [_p("rotation", i) for i in range(7)],
    }
    plan, summary = inference._build_global_plan(tasks, max_problems=2, out_dir=out_dir, overwrite=False)
    assert len(plan) == 4  # 2 per task × 2 tasks
    assert summary == {"blending": (2, 0), "rotation": (2, 0)}


def test_build_global_plan_marks_cached_correctly(tmp_path):
    """``is_cached`` is set when the output PNG already exists on disk
    (cheap stat). Decode validation happens later in
    ``_build_skipped_result`` (a forecast-cached file that fails to decode
    falls through to a real run)."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    task_dir = out_dir / "blending"
    task_dir.mkdir()
    # Cache problem 0001 (existence only).
    (task_dir / "0001_output.png").write_bytes(b"junk-png")

    tasks = {
        "blending": [
            Problem(
                pid=i, task="blending", mode="", visual_condition="",
                instruction="x", metadata={},
                _input=Path("p"), _answer=None,
            )
            for i in range(3)
        ],
    }
    plan, summary = inference._build_global_plan(tasks, max_problems=None, out_dir=out_dir, overwrite=False)
    cached = [is_cached for _, _, _, is_cached in plan]
    assert cached == [False, True, False]
    assert summary == {"blending": (3, 1)}


def test_build_global_plan_overwrite_disables_cache(tmp_path):
    """``--overwrite`` invalidates the cache: nothing in the plan should
    be marked is_cached even if PNGs exist."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    task_dir = out_dir / "blending"
    task_dir.mkdir()
    (task_dir / "0000_output.png").write_bytes(b"junk-png")

    tasks = {"blending": [Problem(
        pid=0, task="blending", mode="", visual_condition="",
        instruction="x", metadata={},
        _input=Path("p"), _answer=None,
    )]}
    plan, summary = inference._build_global_plan(tasks, max_problems=None, out_dir=out_dir, overwrite=True)

    cached = [is_cached for _, _, _, is_cached in plan]
    assert cached == [False]
    assert summary == {"blending": (1, 0)}
