# CLAUDE.md

Orientation for AI coding agents (Claude Code, Cursor, Codex, …) working
in this repo. User-facing prose lives in `README.md`; raw `uv run python`
commands behind every `make` target live in `docs/run_cycle.md`. This file
covers what those don't: where the code fits together, what conventions
are load-bearing, and what to leave alone.

## What this repo is

**PaintBench** is a precise, deterministic visual-editing benchmark. Every
problem is an `(input_image, instruction, answer_image)` triplet generated
programmatically, so the answer is pixel-exact and the distribution of
valid answers is known by construction. Two siblings share one pipeline:

- **PaintBench** — 20 tasks x 8 visual conditions x 12 problems = 1920,
  plus a separate `preservation/` diagnostic split (excluded from scoring).
- **TinyGrafixBench** — 30 x 20 = 600 chart-edit problems on 5
  matplotlib chart families.

Same on-disk layout, same eval, same registry of models. Adding a
benchmark variant should slot in along the same pattern.

## The pipeline

Six stages, one CLI per stage under `src/`. Every stage has a `make`
target that wraps the CLI; the make target is the supported entry
point.

| Stage | CLI | Make target | Output |
|------|-----|-------------|--------|
| 1. Generate | `src/generate_benchmark.py` | `make generate-all` | `benchmarks/<bench>/<task>/<NNN>_{input,answer}.png + <NNN>.json` |
| 2. Inference | `src/inference.py` | `make inference MODEL=... BENCHMARK=...` | `model_outputs/<model>/<bench>/<task>/<NNNN>_output.png + <NNNN>.json` |
| 3. Eval | `src/eval.py` | `make eval` | `eval_outputs/.../<NNNN>_stats.json` + `eval_outputs/problem_stats.jsonl` |
| 4. Stats | `src/stats.py` | `make stats` | `eval_outputs/aggregate_stats.jsonl` (with 95% bootstrap CIs) |
| 5. Report | `src/report.py` | `make report` | `report.html` |
| 6. Visualize | `src/visualize.py` | `make viz` | Web UI on `:8765` (Generate / Benchmark / Eval tabs) |

When in doubt about how to drive a stage, run `make help` — every
target self-documents there, and the variable block prints below the
targets.

## Where to start reading

If you're new to the repo and need to make changes, read in this order:

1. `README.md` — task list, model list, run cycle.
2. `Makefile` — every supported workflow as a one-liner. The variable
   block at the top is the spec for what each knob does.
3. `src/tasks/base.py` — generation-side `Problem` dataclass + the
   shared geometry / color / control-point helpers every task uses.
4. `src/benchmark_source.py` — inference-side `Problem` and the
   `BenchmarkSource` adapter (`LocalBenchmarkSource` reads
   `benchmarks/<bench>/...`, `HfBenchmarkSource` loads via
   `datasets.load_dataset(repo, config, split, revision=<sha>)` so
   `--benchmark hf:<repo>@<sha>` works without a local clone).
5. `src/inference.py` — `BaseModel` interface + the `_REGISTRY` dict
   near the bottom that maps CLI keys (`flux2-dev`, `nano-banana-2`, …)
   to model classes. To add a model, subclass `BaseModel`, implement
   `load_model()` + `generate(image, instruction)`, and add it to
   `_REGISTRY`.
6. `src/tasks/<name>.py` — every PaintBench task module exports
   `NAME`, `PARAMETERS` (the parameter grid), and
   `generate(seed, bg_spec, W, H, obj_colors, **kwargs) -> Problem`.
   Register a new task by appending to `TASKS` at the top of
   `src/generate_benchmark.py`.
7. `src/tinygrafixbench/<chart>.py` — every chart module exports
   `TASKS` and `generate_task(seed, task) -> (input_fig, answer_fig, instruction)`.
   See `docs/tinygrafixbench-tasks.md` for the full task reference and
   a recipe for adding a new chart family.

## Architectural conventions

These are not stylistic preferences; the test suite enforces most of
them.

- **Determinism is load-bearing.** Any randomness in a task generator
  must go through a seeded `random.Random` (PaintBench) or
  `numpy.random.default_rng` (TinyGrafixBench) instance threaded down
  from the seed. **Never** touch the module-level `random` /
  `np.random`. `tests/test_determinism.py` re-runs every task with
  fixed seeds and compares pixel-byte hashes — silent RNG drift will
  fail it. The `Makefile`'s artifact-writing targets prepend
  `PYTHONHASHSEED=0` for belt-and-suspenders set/dict-iteration
  stability; keep that prefix if you add new artifact targets.
- **Two `Problem` shapes, one schema.** Generation yields the in-memory
  `Problem` from `src/tasks/base.py` (PIL images plus metadata).
  Inference consumes the lazy `Problem` from `src/benchmark_source.py`
  (paths or HF rows; images decoded on attribute access). They are
  related but distinct — don't import one where the other is expected.
- **Output paths are conventions, not implementation details.**
  `model_outputs/<model>/<benchmark>/<task>/<NNNN>_output.png` is the
  contract that ties inference, eval, and the visualizer together.
  Don't rename without updating `eval.py` and the visualizer in
  lockstep.
- **`eval_outputs/` is derived.** Fully reproducible from
  `benchmarks/` + `model_outputs/` via the cached eval pass. Treat it
  as disposable — only `model_outputs/` is worth preserving across
  machines.
- **Use `uv run python ...` or a `make` target — never bare `python`.**
  Dependencies are managed via `pyproject.toml` + `uv.lock`; there is
  no committed venv. Adding a dep means editing `pyproject.toml` (the
  right `[project.optional-dependencies]` extra: `inference` for GPU
  stacks, `api` for remote APIs, `data` for HF dataset publishing)
  and refreshing `uv.lock` with `uv sync --all-groups`.

## The cache + `OVERWRITE=1` idiom

Both inference and eval are **incremental by default**:

- **Inference** reuses any existing `<NNNN>_output.png` in `--out-dir`
  after a `PIL.Image.load()` decode check (so truncated writes from
  killed prior runs fall through and get redone). Cached problems
  return synthetic results marked `"skipped": true` and don't consume
  worker slots or API quota.
- **Eval** reuses any `<NNNN>_stats.json` sidecar whose mtime is >=
  every source PNG (input / answer / output) and the per-problem
  metadata JSON. The default also caches four diagnostic ΔE diff
  PNGs (`<NNNN>_diff_cie76_{0,2,5,10}.png`) consumed by the
  visualizer; pass `--no-save-images` (or `make eval-quick`) for
  stats-only fast runs.

When you change the underlying logic — inference prompt / sampling /
gateway routing, or eval CIE76 math / threshold list / normalization
— pass `OVERWRITE=1` to invalidate the relevant cache without manually
deleting trees:

```bash
make inference MODEL=flux2-dev BENCHMARK=PaintBench OVERWRITE=1
make eval OVERWRITE=1
```

This is preferred over `rm -rf` because the rerun then shows up in the
inference summary and the eval `=== Plan ===` block prints `(cached:
N/M)` so you can see at a glance what's being recomputed.

## Concurrency

Two independent knobs:

- **`JOBS=N`** — multiprocessing parallelism for `generate`,
  `eval`/`eval-quick`, and `stats`. Maps to `--jobs N` / `--workers N`.
  Default lets each script pick `os.cpu_count()`. **Set `JOBS=1` from
  sandboxed terminals** (notably Cursor's integrated terminal on
  macOS) where POSIX semaphore creation is blocked — see
  Troubleshooting.
- **`WORKERS=N`** — concurrent calls per task during inference. Sync
  models (diffusers, BAGEL, Hunyuan) use a `ThreadPoolExecutor` with
  per-instance `threading.Lock` serialisation around `generate()`;
  async models (Nano Banana, GPT-Image-2) use an `asyncio.Semaphore`.
  All paths are safe at any `WORKERS` value, but local GPU models are
  usually GPU-bound so `WORKERS>1` mainly helps remote APIs (4-8
  typical).

## Model-specific gotchas

- **BAGEL** (`--model bagel`) has no pip package — `BAGELModel.load_model()`
  clones the pinned upstream SHA from `_BAGEL_UPSTREAM_REF` in
  `src/inference.py` into `$HF_HUB_CACHE/paintbench/bagel-upstream/`
  on first use. `git` must be on `PATH`. For air-gapped nodes,
  pre-clone manually and override with the `BAGEL_REPO` env var or
  `--bagel-repo <path>`. `flash_attn` is a runtime requirement on the
  GPU node (not a `pyproject.toml` dep — install separately, with a
  prebuilt wheel matched to your CUDA / torch ABI; e.g.
  `pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl`).
- **Hunyuan** (`hunyuan-image-3` / `hunyuan-image-3-instruct`) needs
  **>= 8 x 80 GB VRAM**. Weights download to
  `<HF cache>/paintbench/HunyuanImage-3-Instruct{-Distil}/` by
  default; override with `--model-cache-dir` or `$HUNYUAN_MODEL_DIR`.
  The pipeline call passes `image_size=f"{H}x{W}"` matched to the
  input image — don't drop that arg or the model silently regenerates
  at a default size.
- **Nano Banana / GPT-Image-2** read API keys from `.env` at
  `inference.py` import time. See `.env.example` for the canonical
  template. Per-variant overrides (`NB1_MODEL_NAME` / `NB2_MODEL_NAME`
  / `GEMINI_BASE_URL` / `NB{1,2}_INCLUDE_THOUGHTS`) let one `.env`
  route both Nano Banana variants through a Gemini-compatible
  proxy/gateway with different model ids per registry entry.
  Precedence: **CLI flag > `<NB1|NB2>_<NAME>` env > `GEMINI_<NAME>` env > class default**.

## Adding a task

`README.md` has the surface-level recipe. Architecturally:

1. New file `src/tasks/<name>.py` exporting `NAME`, `PARAMETERS`,
   `generate(seed, bg_spec, W, H, obj_colors, **kwargs) -> Problem`.
   Reuse helpers in `src/tasks/base.py` rather than re-rolling
   geometry — the determinism contract assumes them.
2. Append `("tasks.<name>", "<name>")` to the `TASKS` list at the top
   of `src/generate_benchmark.py`, and add the task → category mapping
   in `TASK_CATEGORIES` below it.
3. `tests/test_task_registry.py` will hit the new task with several
   seeds on next run; `test_determinism.py` will pin its byte hashes.
   If you intentionally change a generator's RNG path later, update
   `test_determinism.py`'s expected hashes in the same commit.

## Don't commit / don't touch

- Anything in `.gitignore`: `benchmarks/`, `model_outputs/`,
  `eval_outputs/`, `data/`, `logs/`, `report.html`, `archive/`,
  `.venv/`, `.env`, `*.local.json`. These are generated artifacts or
  secrets; many are 100s of MB.
- `src/tinygrafixbench/fonts/DejaVuSans.ttf` — bundled deliberately so
  chart rendering is reproducible across machines without depending on
  system font availability. Do not swap or remove.
- The `Problem` JSON sidecar schema (`task`, `mode`, `visual_condition`,
  `instruction`, `problem_id`, `W`, `H`, …) is consumed by every
  downstream stage. Adding fields is fine; renaming is a coordinated
  change across `generate_benchmark.py`, `benchmark_source.py`,
  `eval.py`, `stats.py`, and the visualizer.

## Before committing

```bash
make lint     # ruff over src/ scripts/ tests/
make test     # pytest — covers determinism, registry, eval math,
              # cache invalidation, smoke pipeline
```

Both must pass. Write focused, atomic commits — explain the *why* in
the message, not a restatement of the diff.

## Troubleshooting

**`PermissionError: [Errno 1] Operation not permitted` from
`_multiprocessing.SemLock` during `make generate*` / `make eval` /
`make stats`.** Some macOS terminals (notably Cursor's integrated
terminal) sandbox POSIX semaphore creation, which kills
`ProcessPoolExecutor`. Bypass by disabling multiprocessing:

```bash
make generate-all JOBS=1
make eval         JOBS=1
make stats        JOBS=1
```

Doesn't affect Linux or non-sandboxed terminals (Terminal.app, iTerm2,
ssh sessions). CI runs Linux so this never bites there.

**`uv pip install --reinstall <name>` silently drifts off the
lockfile.** Bare `--reinstall <name>` pulls the latest matching version
from PyPI rather than the version pinned in `uv.lock`, which can break
CUDA/cuDNN compat for `torch`. The right forms are
`uv pip install --reinstall <name>==<lockfile-version>` (explicit) or
`uv sync --reinstall-package <name>` (re-resolves through the
lockfile).

## Further reading

- `README.md` — quick start, model table, full PaintBench task list,
  result-sync ergonomics.
- `CONTRIBUTING.md` — dev setup, PR conventions, release process.
- `docs/run_cycle.md` — the raw `uv run python` form behind each make
  target (use inside Slurm `--wrap` etc.).
- `docs/metric.md` — CIE76 / IoU / edit-accuracy explainer, plus the
  aggregation hierarchy and bootstrap-CI methodology.
- `docs/extending.md` — recipes for adding new models, PaintBench
  tasks, or TinyGrafixBench chart families.
- `docs/paintbench-tasks.md` — PaintBench task reference (4 categories
  x 5 tasks, plus the 8 visual conditions and the `preservation/`
  diagnostic).
- `docs/tinygrafixbench-tasks.md` — TinyGrafixBench task reference
  (5 charts x 4 tasks).
- `make help` — every supported target with defaults and examples.
