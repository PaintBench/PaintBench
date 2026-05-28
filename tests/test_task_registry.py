"""Every task registered in generate_benchmark.TASKS must export the
expected interface (NAME, PARAMETERS, generate) and actually be able to
produce one valid problem. Catches missing exports, broken imports, and
tasks that silently return None for common seeds."""
from __future__ import annotations

import importlib

import pytest

import generate_benchmark as gb
from conftest import gen_one


@pytest.mark.parametrize("mod_name,task_name", gb.TASKS)
def test_task_exports(mod_name, task_name):
    mod = importlib.import_module(mod_name)
    assert hasattr(mod, "NAME"),       f"{mod_name} missing NAME"
    assert hasattr(mod, "PARAMETERS"), f"{mod_name} missing PARAMETERS"
    assert hasattr(mod, "generate"),   f"{mod_name} missing generate()"
    assert mod.NAME == task_name, f"{mod_name}.NAME={mod.NAME!r} != {task_name!r}"
    assert callable(mod.generate)
    assert isinstance(mod.PARAMETERS, dict)


@pytest.mark.parametrize("mod_name,task_name", gb.TASKS)
def test_task_generates_one_problem(mod_name, task_name):
    """Every registered task can produce a valid problem for at least one
    of its modes at a known seed. If no mode works, mark xfail — catches
    the case where a previously-working task silently breaks."""
    mod = importlib.import_module(mod_name)
    modes = mod.PARAMETERS.get("mode") or [None]
    # A small spread of seeds — any one passing is enough; tasks don't have
    # a valid problem at every seed by design.
    for seed in (42, 123, 7, 31415, 2718):
        for mode in modes:
            prob = gen_one(task_name, mode, seed)
            if prob is not None and not prob.error:
                # Problem looks well-formed
                assert prob.instruction, "empty instruction"
                assert prob.input_image.size  == (1024, 1024)
                assert prob.answer_image.size == (1024, 1024)
                return
    pytest.fail(f"{task_name} produced no valid problem across 5×{len(modes)} (seed, mode)")


def test_all_tinygrafixbench_modules_export_interface():
    """TGF modules expose TASKS (list of task names) and generate_task()."""
    import matplotlib.pyplot as plt
    graphs = ["bar_chart", "heatmap", "line_chart", "network", "scatter_plot"]
    try:
        for g in graphs:
            mod = importlib.import_module(f"tinygrafixbench.{g}")
            assert hasattr(mod, "TASKS"),         f"tinygrafixbench.{g} missing TASKS"
            assert hasattr(mod, "generate_task"), f"tinygrafixbench.{g} missing generate_task"
            assert isinstance(mod.TASKS, list) and len(mod.TASKS) == 4, \
                f"tinygrafixbench.{g}.TASKS should have 4 tasks, got {mod.TASKS!r}"
            # And actually run one
            _, _, instr = mod.generate_task(seed=42, task=mod.TASKS[0])
            assert isinstance(instr, str) and instr, f"{g}/{mod.TASKS[0]} empty instruction"
    finally:
        plt.close("all")
