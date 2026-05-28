# Extending PaintBench

PaintBench has three pluggable surfaces: **models**, **PaintBench
tasks**, and **TinyGrafixBench chart families**. This document walks
through the recipe for each. Architectural conventions (the determinism
contract, the `Problem` schema, the cache invariants) are documented in
[`CLAUDE.md`](../CLAUDE.md) — read that first.

## Adding a model

Models are subclasses of `BaseModel` in `src/inference.py`, registered
in the `_REGISTRY` dict at the bottom of the file.

### 1. Pick the right base shape

| Runtime | Pattern | Examples |
|---|---|---|
| Local diffusers pipeline | Subclass `BaseModel`, load with `diffusers.AutoPipelineFor*`, override `generate(image, instruction)` | `InstructPix2PixModel`, `Flux2DevModel`, `QwenImageEditModel` |
| Local research-code drop (no pip package) | Subclass `BaseModel`, vendor or auto-clone the upstream repo in `load_model()`, expose its inference call via `generate()` | `BAGELModel` (clones `bytedance-seed/BAGEL` on first use) |
| Remote API | Subclass `BaseModel`, do async API calls in `generate()`, drive concurrency via `asyncio.Semaphore` | `NanoBanana2Model`, `GptImage2Model` |

### 2. Implement the contract

```python
class MyModel(BaseModel):
    DEFAULT_MODEL_NAME = "my-org/my-model"

    def load_model(self):
        # Download / instantiate weights. Called once per process.
        self.pipe = ...

    def generate(self, image: Image.Image, instruction: str) -> Image.Image:
        # Single-image edit. Return a PIL Image. Optionally return a
        # (PIL Image, reasoning_text) tuple if the model emits a CoT
        # trace you want saved as a sibling `.txt` file.
        return self.pipe(image=image, prompt=instruction).images[0]
```

The thread-safety contract: `generate()` is called from a
`ThreadPoolExecutor` whenever `--workers N > 1`. If your model has a
mutable pipeline (e.g. diffusers' scheduler `_step_index`), wrap the
call body in a `with self._lock:` to serialise it. The base class
provides `self._lock` for free.

### 3. Register it

Append to `_REGISTRY` at the bottom of `src/inference.py`:

```python
_REGISTRY: dict[str, type[BaseModel]] = {
    ...,
    "my-model":  MyModel,
}
```

The registry key is what users pass via `make inference MODEL=my-model`.
Use lowercase-with-hyphens.

### 4. Test it

`tests/test_inference_registry.py` enumerates `_REGISTRY` and checks
that every key instantiates, has the expected interface, and surfaces
sensible defaults. Add your model name to `EXPECTED_MODELS` there.

### 5. Document it

Add a row to the Models table in [`../README.md`](../README.md) with
the HF model id and any notable requirements (VRAM, gated repo,
runtime install steps, etc.).

## Adding a PaintBench task

PaintBench tasks live in `src/tasks/<name>.py` and are registered in
the `TASKS` list at the top of `src/generate_benchmark.py`.

### 1. Write the generator

```python
# src/tasks/my_task.py
import random
from .base import Problem, _make_scene, _shape_at  # use shared helpers

NAME = "my_task"

PARAMETERS = {
    "mode": ["a", "b"],   # parameter grid; one (mode × visual_condition × seed) per problem
}

def generate(seed: int, bg_spec, W: int, H: int, obj_colors, **kwargs) -> Problem:
    rng = random.Random(seed)   # ⚠️ NEVER use module-level random
    mode = kwargs["mode"]

    scene = _make_scene(rng, bg_spec, W, H, obj_colors)
    # ... build input + answer images ...

    return Problem(
        input_image=input_img,
        answer_image=answer_img,
        instruction=f"...",
        metadata={"params": {"mode": mode}, ...},
    )
```

Critical rules (enforced by `tests/test_determinism.py`):

- All randomness must go through a seeded `random.Random` threaded
  down from the `seed` argument. **Never** touch the module-level
  `random`. Set comprehensions over strings must be wrapped in
  `sorted(...)` before any RNG sampling.
- The shared helpers in `src/tasks/base.py` (`_make_scene`,
  `_sample_control_point`, `_random_parallelogram`, ...) are
  pre-vetted for determinism — prefer them over re-rolling geometry.

### 2. Register it

In `src/generate_benchmark.py`:

```python
TASKS = [
    ...,
    ("tasks.my_task", "my_task"),
]

TASK_CATEGORIES = {
    ...,
    "my_task": "geometric",   # one of the existing 4 categories
}
```

### 3. Test it

`tests/test_task_registry.py` will hit the new task with several seeds
on next run. `tests/test_determinism.py` pins byte hashes for every
task — when you add a task, run the suite locally and let it tell you
the expected hash for your task's three pinned seeds, then commit
those hashes alongside the task implementation. If you later change
the generator's RNG path, update the hashes in the same commit.

### 4. Document it

Add a row to the PaintBench tasks table in [`../README.md`](../README.md).

## Adding a TinyGrafixBench chart family

TinyGrafixBench chart families live in `src/tinygrafixbench/<chart>.py`.
The structure is more rigid than PaintBench tasks because the
benchmark mandates **exactly 4 tasks per chart, one per category**
(Construction / Transformation / Removal / Recoloring).

### 1. Implement the generator module

Required exports:

```python
# src/tinygrafixbench/<chart>.py
TASKS = ["construction_task", "transformation_task", "removal_task", "recoloring_task"]

FIG_W_PX, FIG_H_PX, FIG_DPI = 1024, 768, 160   # fixed for byte-exact savefig

def build_state(seed):
    """Build base scene deterministically from seed. Returns a dict."""
    rng = make_rng(seed)
    ...
    return {"data": ..., "colors": ..., ...}

def task_<name>(base, rng):
    """Implement one task variant. Returns (input_state, answer_state, instruction)."""
    input_state = _copy_state(base)
    answer_state = _copy_state(base)
    # ... mutate answer_state ...
    return input_state, answer_state, "Instruction string..."

TASK_FNS = {
    "construction_task":  task_construction_task,
    ...
}

def generate_task(seed, task):
    """Required entry point. Returns (input_fig, answer_fig, instruction)."""
    base = build_state(seed)
    task_seed = int(seed) * 17 + TASKS.index(task)
    rng = make_rng(task_seed)
    input_state, answer_state, instruction = TASK_FNS[task](base, rng)
    return render_state(input_state), render_state(answer_state), instruction

def render_state(state):
    """Required helper. Returns a matplotlib Figure at the fixed dimensions."""
    ...
```

All randomness through `numpy.random.default_rng` (via the shared
`make_rng` helper in `utils.py`). Per-task seed is derived from the
seed × 17 + task-index to ensure each task in a problem gets
independent randomness while remaining seed-deterministic.

### 2. Register it

The generator module is auto-discovered via the `TASKS` list pattern —
no explicit registration needed. `generate_benchmark.py` picks up any
module under `src/tinygrafixbench/` that exports `TASKS` and
`generate_task`.

### 3. Test it

`tests/test_determinism.py` has a parameterised case per chart module
(`_TGF_CROSS_HASHSEED_CASES`). Add one entry for your new chart family
to keep coverage uniform.

### 4. Document it

Add a section in [`tinygrafixbench-tasks.md`](tinygrafixbench-tasks.md)
matching the existing structure: a 1-sentence chart description plus
a 4-row task table with descriptions and example instructions.

## Adding a benchmark variant

If you're building a sibling benchmark with the same overall structure
(programmatic generation + CIE76 eval) but a different domain — e.g. a
benchmark of 3D-rendered objects, or a UI-screenshot benchmark — the
recipe is to clone the TinyGrafixBench module pattern:

1. Add a top-level CLI flag in `src/generate_benchmark.py` (mirror
   `--paintbench` and `--tinygrafixbench`).
2. Place generators in their own subpackage (`src/<bench>/...`).
3. Eval / stats / report / visualize are already benchmark-agnostic
   and will pick up the new layout automatically as long as the
   on-disk structure
   `benchmarks/<bench>/<task>/<NNN>_{input,answer}.png + <NNN>.json`
   is preserved.
4. Update `src/stats.py`'s benchmark dispatcher (`process_paintbench`,
   `process_tinygrafixbench`) to add a `process_<bench>` function with
   the rollup hierarchy appropriate for your variant. The bootstrap
   helpers in `stats.py` are generic over the hierarchy shape.

This is more involved than adding a task — open a discussion before
starting if you'd like feedback on the structure.
