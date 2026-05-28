"""Same seed → byte-identical problem. This is the load-bearing invariant
of the benchmark — if it silently breaks, every reported number drifts."""
from __future__ import annotations

import importlib
import os
import subprocess
import sys

import pytest

from conftest import gen_one


@pytest.mark.parametrize("task,mode,seed", [
    ("translation",  "align",     42),
    ("translation",  "amount",    42),
    ("rotation",     None,        123),
    ("construction", None,        7),
    ("recolor",      "color_code", 99),
    ("removal",      "attribute", 31415),
    ("cropping",     "straight",  2718),
])
def test_paintbench_determinism(task, mode, seed):
    """Same seed → same instruction + same image bytes, twice in a row."""
    p1 = gen_one(task, mode, seed)
    p2 = gen_one(task, mode, seed)

    assert not p1.error, f"problem erred: {p1.error}"
    assert p1.instruction == p2.instruction
    assert p1.input_image.tobytes()  == p2.input_image.tobytes()
    assert p1.answer_image.tobytes() == p2.answer_image.tobytes()


# Cases that previously hit set-comprehension paths in removal.py /
# recolor.py / legend.py whose iteration order could differ across Python
# processes with different PYTHONHASHSEED values. Each case must be
# picked so the relevant code branch is actually exercised (e.g. removal
# "both" mode needs n_shapes ≥ 2 with shared color or shape).
#
# Coverage note: the `legend/None/99` case exercises a set of RGB tuples
# (`{s.fill}`) which is currently PYTHONHASHSEED-stable in CPython
# (int hash is identity), so this case would pass on the *unfixed* code
# too. It's a defensive guard: if ShapeInstance.fill ever changes to a
# string-containing type, this test will start catching the regression.
# The removal/attribute and recolor/color_code cases above exercise sets
# of strings / (str, tuple) and DO fail on unfixed code (verified).
_CROSS_HASHSEED_CASES = [
    ("removal",  "attribute", 31415),  # exercises L60/L76/L92 set-over-string
    ("recolor",  "color_code",   42),  # exercises L56/L67 set-over-string
    ("legend",   None,           99),  # exercises L296 (defensive — see note)
]


# Sentinel prefix so the JSON fingerprint is unambiguously identifiable in
# stdout, even if a future dependency emits deprecation warnings or other
# print noise from the subprocess.
_FP_SENTINEL = "__PAINTBENCH_FP__"


def _gen_fingerprint(task: str, mode: str | None, seed: int,
                     hashseed: str) -> tuple[str, str, str]:
    """Generate `(task, mode, seed)` in a *fresh* Python process with
    PYTHONHASHSEED=`hashseed`, and return (instruction, sha(input),
    sha(answer)). Fresh processes are required because PYTHONHASHSEED is
    only read once at interpreter startup."""
    script = (
        "import sys, hashlib, json\n"
        "from conftest import gen_one\n"
        "task, mode_str, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])\n"
        "mode = mode_str or None\n"
        "p = gen_one(task, mode, seed)\n"
        "assert not p.error, f'problem erred during fingerprint: {p.error}'\n"
        "rec = json.dumps({\n"
        "    'instruction': p.instruction,\n"
        "    'input_sha':  hashlib.sha256(p.input_image.tobytes()).hexdigest(),\n"
        "    'answer_sha': hashlib.sha256(p.answer_image.tobytes()).hexdigest(),\n"
        "})\n"
        f"print({_FP_SENTINEL!r} + rec)\n"
    )
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env = {**os.environ,
           "PYTHONHASHSEED": hashseed,
           "PYTHONPATH":     os.pathsep.join([
               os.path.join(repo_root, "src"),
               os.path.join(repo_root, "tests"),
               os.environ.get("PYTHONPATH", ""),
           ])}
    out = subprocess.check_output(
        [sys.executable, "-c", script, task, mode or "", str(seed)],
        env=env, cwd=repo_root,
    ).decode()
    import json as _json
    fp_lines = [ln for ln in out.splitlines() if ln.startswith(_FP_SENTINEL)]
    assert len(fp_lines) == 1, (
        f"expected exactly 1 fingerprint line in subprocess stdout, got "
        f"{len(fp_lines)}. Full stdout:\n{out}"
    )
    rec = _json.loads(fp_lines[0][len(_FP_SENTINEL):])
    return rec["instruction"], rec["input_sha"], rec["answer_sha"]


@pytest.mark.parametrize("task,mode,seed", _CROSS_HASHSEED_CASES)
def test_paintbench_determinism_across_hashseeds(task, mode, seed):
    """Benchmark generation must be byte-identical across PYTHONHASHSEED.

    Set comprehensions over strings (or string-containing tuples) iterate
    in hash-seed-dependent order. Feeding such a list to ``rng.choice`` /
    ``rng.shuffle`` then picks a different element per Python process,
    silently producing different answers + instructions across machines.
    Fix: wrap in ``sorted(...)`` (and pin PYTHONHASHSEED=0 in
    ``make generate`` as belt-and-suspenders).
    """
    fps = [_gen_fingerprint(task, mode, seed, hs) for hs in ("0", "1", "12345")]
    assert fps[0] == fps[1] == fps[2], (
        f"non-deterministic across PYTHONHASHSEED for {task}/{mode}/{seed}:\n"
        f"  PYTHONHASHSEED=0     → {fps[0]}\n"
        f"  PYTHONHASHSEED=1     → {fps[1]}\n"
        f"  PYTHONHASHSEED=12345 → {fps[2]}"
    )


@pytest.mark.parametrize("graph,task,seed", [
    ("bar_chart",    "recolor_bar",        42),
    ("bar_chart",    "sort_bars",          7),
    ("heatmap",      "shift_heatmap",      99),
    ("line_chart",   "normalize_series",   123),
    ("network",      "remove_node",        31415),
    ("scatter_plot", "draw_best_fit_line", 2718),
])
def test_tinygrafixbench_determinism(graph, task, seed):
    """Same seed → same instruction for every (chart, task) pair."""
    import matplotlib.pyplot as plt
    mod = importlib.import_module(f"tinygrafixbench.{graph}")
    try:
        _, _, instr1 = mod.generate_task(seed, task)
        _, _, instr2 = mod.generate_task(seed, task)
        assert instr1 == instr2
    finally:
        plt.close("all")


# Mirror of _CROSS_HASHSEED_CASES for TGF. No actual set-of-strings bugs
# exist in TGF today; these are defensive guards against a future regression
# of the same class. One case per chart module so the test exercises every
# TGF code path.
_TGF_CROSS_HASHSEED_CASES = [
    ("bar_chart",    "recolor_bar",        42),
    ("heatmap",      "shift_heatmap",      99),
    ("line_chart",   "normalize_series",   123),
    ("network",      "remove_node",        31415),
    ("scatter_plot", "draw_best_fit_line", 2718),
]

_TGF_FP_SENTINEL = "__TINYGRAFIXBENCH_FP__"


def _gen_tgf_fingerprint(graph: str, task: str, seed: int,
                         hashseed: str) -> tuple[str, str, str]:
    """Generate a TGF (graph, task, seed) in a fresh Python process with
    PYTHONHASHSEED=`hashseed`, render both figures to PNG bytes, and return
    (instruction, sha(input_png), sha(answer_png))."""
    script = (
        "import sys, io, hashlib, json\n"
        "import importlib\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "graph, task, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])\n"
        "mod = importlib.import_module(f'tinygrafixbench.{graph}')\n"
        "input_fig, answer_fig, instruction = mod.generate_task(seed, task)\n"
        "def fig_sha(fig):\n"
        "    buf = io.BytesIO()\n"
        "    fig.savefig(buf, format='png')\n"
        "    return hashlib.sha256(buf.getvalue()).hexdigest()\n"
        "rec = json.dumps({\n"
        "    'instruction': instruction,\n"
        "    'input_sha':   fig_sha(input_fig),\n"
        "    'answer_sha':  fig_sha(answer_fig),\n"
        "})\n"
        "plt.close('all')\n"
        f"print({_TGF_FP_SENTINEL!r} + rec)\n"
    )
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env = {**os.environ,
           "PYTHONHASHSEED": hashseed,
           "PYTHONPATH":     os.pathsep.join([
               os.path.join(repo_root, "src"),
               os.path.join(repo_root, "tests"),
               os.environ.get("PYTHONPATH", ""),
           ])}
    out = subprocess.check_output(
        [sys.executable, "-c", script, graph, task, str(seed)],
        env=env, cwd=repo_root,
    ).decode()
    import json as _json
    fp_lines = [ln for ln in out.splitlines() if ln.startswith(_TGF_FP_SENTINEL)]
    assert len(fp_lines) == 1, (
        f"expected exactly 1 fingerprint line in subprocess stdout, got "
        f"{len(fp_lines)}. Full stdout:\n{out}"
    )
    rec = _json.loads(fp_lines[0][len(_TGF_FP_SENTINEL):])
    return rec["instruction"], rec["input_sha"], rec["answer_sha"]


@pytest.mark.parametrize("graph,task,seed", _TGF_CROSS_HASHSEED_CASES)
def test_tinygrafixbench_determinism_across_hashseeds(graph, task, seed):
    """TGF benchmark generation must be byte-identical across PYTHONHASHSEED.

    No actual set-of-strings non-determinism exists in src/tinygrafixbench/
    today, so this test passes on current code. It exists as a defensive
    guard: if a future contributor adds a `list({...string-set})` pattern
    in a chart generator, this test will fail.
    """
    fps = [_gen_tgf_fingerprint(graph, task, seed, hs) for hs in ("0", "1", "12345")]
    assert fps[0] == fps[1] == fps[2], (
        f"non-deterministic across PYTHONHASHSEED for {graph}/{task}/{seed}:\n"
        f"  PYTHONHASHSEED=0     → {fps[0]}\n"
        f"  PYTHONHASHSEED=1     → {fps[1]}\n"
        f"  PYTHONHASHSEED=12345 → {fps[2]}"
    )


def test_different_seeds_differ():
    """Sanity check: the RNG isn't a global constant masquerading as seeded."""
    a = gen_one("translation", "align", 1)
    b = gen_one("translation", "align", 2)
    assert a.input_image.tobytes() != b.input_image.tobytes()
