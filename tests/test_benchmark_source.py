"""Tests for the BenchmarkSource adapter (Phase 1a).

Covers both sources and the CLI dispatcher:

* :class:`LocalBenchmarkSource` — preserves the on-disk layout that
  pre-Phase-1 ``inference.load_benchmark`` consumed; problems iterate in
  pid order; image accessors return decoded RGB images.
* :class:`HfBenchmarkSource` — loads from the live
  ``PaintBench/PaintBench`` dataset on the Hub. Network-dependent test;
  skipped when offline or unreachable.
* :func:`parse_benchmark_arg` — argparse-side dispatcher; verifies
  local-path vs ``hf:...`` branching and revision parsing.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from benchmark_source import (
    BenchmarkSource,
    HfBenchmarkSource,
    LocalBenchmarkSource,
    Problem,
    parse_benchmark_arg,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_BENCH_ROOT = _REPO_ROOT / "benchmarks" / "PaintBench"

# Sentinel task used by every "load one task" assertion. Picked because
# `blending` is small (96 problems), present in the local benchmark, and
# also appears in the HF `dev` split — so the same row can be looked up
# in both sources for the local↔HF parity test.
_SENTINEL_TASK = "blending"


# ─── LocalBenchmarkSource ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def local_paintbench() -> LocalBenchmarkSource:
    """Point at the live ``benchmarks/PaintBench`` tree (regenerated as
    part of every ``make generate-all``). Skipped if the user hasn't
    generated it locally — keeps CI / fresh-checkout runs green."""
    if not _LOCAL_BENCH_ROOT.exists():
        pytest.skip(
            f"Local benchmark tree not generated: {_LOCAL_BENCH_ROOT}. "
            "Run `make generate-all` to populate it."
        )
    return LocalBenchmarkSource(_LOCAL_BENCH_ROOT, name="PaintBench")


def test_local_source_preserves_existing_behavior(local_paintbench):
    """Smoke check on a real local benchmark task.

    Verifies the source's contract end-to-end: ``name()`` /
    ``revision()`` surface what inference.py prints in its banner;
    ``iter_tasks()`` returns the task folder names; ``iter_problems()``
    yields well-formed :class:`Problem`s whose image accessors decode to
    1024×1024 RGB PIL images (matches the local PNG dimensions).
    """
    src = local_paintbench
    assert src.name() == "PaintBench"
    assert src.revision() == str(_LOCAL_BENCH_ROOT)

    tasks = list(src.iter_tasks())
    assert _SENTINEL_TASK in tasks, f"Expected '{_SENTINEL_TASK}' in {tasks}"

    problems = list(src.iter_problems(_SENTINEL_TASK))
    assert problems, f"No problems iterated for task {_SENTINEL_TASK!r}"

    p = problems[0]
    assert isinstance(p, Problem)
    assert p.task == _SENTINEL_TASK
    assert p.instruction, "instruction must be non-empty"

    # Image accessors return decoded RGB PIL images sized to match the
    # canvas the generator wrote (1024×1024 for PaintBench).
    input_img = p.input_image
    assert isinstance(input_img, Image.Image)
    assert input_img.mode == "RGB"
    assert input_img.size == (1024, 1024)

    assert p.answer_image is not None, "blending problems all have an answer image"
    assert p.answer_image.size == (1024, 1024)

    # Cheap header-only size accessor agrees with the decoded image.
    assert tuple(p.input_size_wh) == input_img.size


def test_local_source_pid_ordering(local_paintbench):
    """Within a task, problems must come out in pid order — the metrics
    JSON downstream sorts by ``index`` for stable diffs across runs, so
    iteration order should already be monotonic to make that a no-op."""
    pids = [p.pid for p in local_paintbench.iter_problems(_SENTINEL_TASK)]
    assert pids == sorted(pids), f"Problems out of pid order: {pids}"
    # Lock the contract: blending starts at pid 0 and increments by 1.
    assert pids[0] == 0
    assert pids == list(range(len(pids)))


def test_local_source_problem_dict_access(local_paintbench):
    """The Problem dataclass exposes a dict-style ``__getitem__`` so the
    inference orchestrator (which treats problems as dicts) keeps working
    unchanged. Verify the keys inference.py actually consumes."""
    p = next(local_paintbench.iter_problems(_SENTINEL_TASK))
    # Hot-path keys consumed by run_single / _build_skipped_result.
    for key in ("index", "instruction", "task", "mode", "visual_condition",
                "input_image", "input_size_wh"):
        assert key in p, f"Problem missing dict-style key {key!r}"
    assert p["index"] == p.pid
    assert p["task"] == p.task
    assert isinstance(p["input_image"], Image.Image)
    assert p.get("nonexistent", "default") == "default"


# ─── HfBenchmarkSource ──────────────────────────────────────────────────────

def _build_hf_source_or_skip() -> HfBenchmarkSource:
    """Build a real :class:`HfBenchmarkSource` against the live
    ``PaintBench/PaintBench`` repo, or :func:`pytest.skip` if the network
    or cache is unavailable.

    Uses the ``dev`` split (280 rows) so the download/loader is cheap.
    Honours ``HF_HUB_OFFLINE`` so air-gapped CI doesn't fight the test.
    """
    if os.environ.get("HF_HUB_OFFLINE"):
        pytest.skip("HF_HUB_OFFLINE set — skipping network-dependent test")
    try:
        return HfBenchmarkSource(
            "PaintBench/PaintBench",
            "PaintBench",
            split="dev",
        )
    except Exception as exc:  # noqa: BLE001 — broad catch is intentional
        pytest.skip(f"HF unreachable: {exc}")


def test_hf_source_loads_one_problem(local_paintbench):
    """Round-trip parity: the same problem loaded from the local tree
    and from the HF dev split should agree on identity (task, mode,
    visual_condition, problem_id) and on the instruction text.

    This pins the contract that the HF dataset is a faithful mirror of
    the regenerated local benchmark — same generator seeds → same
    instructions → reproducible scores across both code paths. Skipped
    when HF is unreachable so offline / air-gapped CI stays green.
    """
    hf_src = _build_hf_source_or_skip()
    assert hf_src.name() == "PaintBench"
    rev = hf_src.revision()
    assert rev.startswith("hf:PaintBench/PaintBench@")
    assert "split=dev" in rev

    assert _SENTINEL_TASK in list(hf_src.iter_tasks())

    hf_problems = list(hf_src.iter_problems(_SENTINEL_TASK))
    assert hf_problems, "HF dev split should have at least one blending problem"

    # Compare HF row 0 of blending with the matching local problem
    # (same task + mode + visual_condition + pid). The dev split is a
    # stratified one-per-cell subsample so look the cell up by tuple.
    hp = hf_problems[0]
    local_match = next(
        (lp for lp in local_paintbench.iter_problems(_SENTINEL_TASK)
         if (lp.task, lp.mode, lp.visual_condition, lp.pid)
         == (hp.task, hp.mode, hp.visual_condition, hp.pid)),
        None,
    )
    assert local_match is not None, (
        f"Local benchmark missing HF cell "
        f"({hp.task}, {hp.mode}, {hp.visual_condition}, pid={hp.pid})"
    )

    # Identity matches across sources.
    assert hp.task == local_match.task
    assert hp.mode == local_match.mode
    assert hp.visual_condition == local_match.visual_condition
    assert hp.pid == local_match.pid
    assert hp.instruction == local_match.instruction

    # HF image is already in memory (datasets library decoded it).
    assert isinstance(hp.input_image, Image.Image)
    assert hp.input_image.mode == "RGB"
    assert hp.input_size_wh == local_match.input_size_wh


# ─── parse_benchmark_arg (CLI dispatcher) ───────────────────────────────────

def test_cli_arg_parsing(tmp_path):
    """Unit test on the local-vs-HF dispatch logic without touching the
    HF network. We mock out the HF constructor by patching the symbol the
    dispatcher imports under, so we can assert the right class is built
    with the right args without actually loading a dataset."""
    # Local path → LocalBenchmarkSource.
    fake_root = tmp_path / "benchmarks" / "PaintBench"
    (fake_root / "blending").mkdir(parents=True)
    src = parse_benchmark_arg(str(fake_root))
    assert isinstance(src, LocalBenchmarkSource)
    assert src.name() == "PaintBench"

    # Local path + --benchmark-config override → name comes from config.
    src = parse_benchmark_arg(str(fake_root), config="custom-name")
    assert isinstance(src, LocalBenchmarkSource)
    assert src.name() == "custom-name"

    # HF spec requires --benchmark-config.
    with pytest.raises(ValueError, match="--benchmark-config is required"):
        parse_benchmark_arg("hf:PaintBench/PaintBench")

    # HF dispatch: stub out HfBenchmarkSource so we don't hit the network.
    calls: list[dict] = []

    class _FakeHfBenchmarkSource(BenchmarkSource):
        def __init__(self, repo_id, config, *, split="test", revision=None):
            calls.append({
                "repo_id": repo_id, "config": config,
                "split": split, "revision": revision,
            })
            self.repo_id = repo_id
            self.config = config
            self.split = split
            self.revision_spec = revision

        def name(self): return self.config
        def revision(self): return f"hf:{self.repo_id}@{self.revision_spec or 'main'}"
        def iter_tasks(self): return iter(())
        def iter_problems(self, task): return iter(())

    import benchmark_source as bs
    monkey_orig = bs.HfBenchmarkSource
    bs.HfBenchmarkSource = _FakeHfBenchmarkSource  # type: ignore[assignment]
    try:
        # Plain hf:<repo>
        src = bs.parse_benchmark_arg(
            "hf:PaintBench/PaintBench",
            config="PaintBench",
            split="dev",
        )
        assert isinstance(src, _FakeHfBenchmarkSource)
        assert calls[-1] == {
            "repo_id": "PaintBench/PaintBench",
            "config": "PaintBench",
            "split": "dev",
            "revision": None,
        }

        # hf:<repo>@<revision>
        bs.parse_benchmark_arg(
            "hf:PaintBench/PaintBench@2cb941fd",
            config="TinyGrafixBench",
        )
        assert calls[-1] == {
            "repo_id": "PaintBench/PaintBench",
            "config": "TinyGrafixBench",
            "split": "test",        # default
            "revision": "2cb941fd",
        }

        # Malformed hf:<empty>
        with pytest.raises(ValueError, match="Malformed --benchmark spec"):
            bs.parse_benchmark_arg("hf:", config="PaintBench")
    finally:
        bs.HfBenchmarkSource = monkey_orig  # type: ignore[assignment]


def test_local_source_rejects_missing_dir(tmp_path):
    """Constructor should fail loudly for a non-existent path, not
    silently produce an empty source."""
    with pytest.raises(FileNotFoundError):
        LocalBenchmarkSource(tmp_path / "does-not-exist", name="x")


def test_problem_input_size_wh_prefers_metadata():
    """The cheap ``input_size_wh`` accessor reads W/H from metadata when
    present so the inference cache-check fast path doesn't need to open
    the PNG. Tested with a Path that doesn't even exist on disk."""
    p = Problem(
        pid=0, task="t", mode="", visual_condition="",
        instruction="x", metadata={"W": 640, "H": 480},
        _input=Path("/does/not/exist.png"), _answer=None,
    )
    assert p.input_size_wh == (640, 480)
