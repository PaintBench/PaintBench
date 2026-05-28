<div align="center">


<!-- TITLE -->
# PaintBench
> Deterministic Evaluation of Precise Visual Editing

<!-- BADGES -->

[![arXiv](https://img.shields.io/badge/cs.CV-arXiv:<ARXIV-ID>-b31b1b.svg?style&logo=arXiv)](https://arxiv.org/abs/<ARXIV-ID>)
[![PDF](https://img.shields.io/badge/📄_PDF-PaintBench-FDDEB3.svg)](https://arxiv.org/pdf/<ARXIV-ID>)
[![Project](https://img.shields.io/badge/🌎_Web-paintbench.github.io-blue.svg)](https://paintbench.github.io)
[![HF Dataset](https://img.shields.io/badge/HF-PaintBench-FED123.svg?style&logo=HuggingFace)](https://huggingface.co/datasets/PaintBench/PaintBench)

  <div style="font-family: charter;">
      <a href="https://kaixubuilds.github.io/" target="_blank">Kai Xu</a><sup>*</sup> &emsp;
      <a href="https://ellisbrown.github.io/" target="_blank">Ellis Brown</a><sup>*</sup> &emsp;
      <a href="https://shrikar17.github.io/" target="_blank">Shrikar Madhu</a> &emsp;
      <a href="https://cs.nyu.edu/~fergus" target="_blank">Rob Fergus</a> &emsp;
      <a href="https://hhexiy.ai/" target="_blank">He He</a> &emsp;
      <a href="https://www.sainingxie.com/" target="_blank">Saining Xie</a>
  </div>

  <br>
  <p><b>New York University</b></p>

  <sup>*Equal contribution.</sup> 
</div>

---

PaintBench evaluates generative image models on MS-Paint-style edits: recolor a
region, draw a border, move a shape, complete a grid pattern. Every problem is a
deterministic `(input_image, instruction, answer_image)` triplet generated from a
seed, so the answer is pixel-exact and the answer distribution is known by
construction — no human raters, no LLM judge, no ambiguity about what "correct"
means.

Two benchmarks share the same file layout and evaluation pipeline.

**PaintBench** — 20 tasks across 4 capability categories, evaluated under 8
visual conditions × 12 problems each (1,920 scored problems), plus a 96-problem
`preservation/` diagnostic split that probes input-fidelity floor.

| Category | Tasks |
|---|---|
| **Geometric transformation** | translation, rotation, reflection, scaling, shearing |
| **Structural manipulation** | construction, removal, copying, border, cropping |
| **Color change** | recolor, flood_fill, blending, gradient, point_operations |
| **Symbolic reasoning** | comparison, ordering, pattern, counting, legend |

**TinyGrafixBench** — chart-edit analog of the primitive ops: 5 matplotlib chart
families × 4 tasks × 30 seeds = 600 problems.

| Chart | Tasks |
|---|---|
| **Network** | add_node, swap_nodes, remove_node, recolor_node |
| **Bar chart** | add_bar, sort_bars, remove_bar, recolor_bar |
| **Scatter** | draw_best_fit_line, swap_axes, remove_outlier, recolor_class |
| **Heatmap** | add_cell, shift_heatmap, mask_cells, change_colormap |
| **Line chart** | draw_segments, normalize_series, filter_series, shade_interval |

Per-task references with modes, visual conditions, and example instructions live
in [`docs/paintbench-tasks.md`](docs/paintbench-tasks.md) and
[`docs/tinygrafixbench-tasks.md`](docs/tinygrafixbench-tasks.md).

## Quick start

```bash
make setup        # install core + dev deps (numpy, pillow, matplotlib, tqdm, ruff)
make generate-all # generate both benchmarks → benchmarks/
make help         # show every make target
```

> On sandboxed shells (e.g. Cursor's integrated terminal on macOS) where
> POSIX semaphore creation is blocked, pass `JOBS=1` to disable
> multiprocessing: `make generate-all JOBS=1`.

## Setup

Requires Python &ge; 3.12 and [uv](https://docs.astral.sh/uv/). Install `uv` with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install one of the dependency sets:

```bash
make setup            # core — generation, eval, stats, report, viz
make setup-inference  # + torch/diffusers/transformers (local GPU models)
make setup-api        # + google-genai / openai (Gemini, Nano Banana, GPT-Image)
make setup-data       # + datasets/huggingface-hub (read benchmarks from HF Hub)
```

For API-backed models, copy `.env.example` to `.env` and fill in your API
keys (`GEMINI_API_KEY`, `OPENAI_API_KEY`, …) before running inference.

## Full run cycle

```bash
make generate-all                                    # 1. generate benchmarks → benchmarks/
make inference MODEL=flux2-dev BENCHMARK=PaintBench  # 2. run a model on a benchmark
#   ...repeat for each (model, benchmark) pair, or: make inference-slurm-h100
make eval     # 3. per-problem CIE76 stats → eval_outputs/problem_stats.jsonl
make stats    # 4. aggregate rollups       → eval_outputs/aggregate_stats.jsonl
make report   # 5. HTML report             → report.html
make viz      # 6. (optional) interactive web visualizer at :8765
```

On a SLURM cluster, `make inference-slurm-h100` dispatches every
`(model × benchmark)` pair in one shot. The driver
([`scripts/submit_inference_h100.sh`](scripts/submit_inference_h100.sh))
is vendor-neutral, env-var-driven, and runs `uv run` directly on the compute
node. It supports typed `--gres=gpu:<type>:N` on modern clusters
(h100 / h200 / a100 / v100 / l40s, auto-detected from `SLURM_PARTITION` when
it matches a known GPU family). Requires `SLURM_PARTITION` + `SLURM_ACCOUNT`;
optional knobs (`SLURM_QOS`, `MODELS`, `BENCHMARKS`, …) are documented in the
script header.

[`docs/run_cycle.md`](docs/run_cycle.md) has the raw `uv run python` commands
behind each make target, for use inside a Slurm `--wrap`.

## Models (`src/inference.py`)

| CLI name | Runtime | Notes |
|----------|---------|-------|
| `instruct-pix2pix` | local (GPU) | `timbrooks/instruct-pix2pix` |
| `longcat-image-edit` | local (GPU) | `meituan-longcat/LongCat-Image-Edit`; CPU-offload by default |
| `qwen-image-edit` | local (GPU) | `Qwen/Qwen-Image-Edit-2511` |
| `flux2-dev` | local (GPU) | `black-forest-labs/FLUX.2-dev` |
| `flux1-kontext-dev` | local (GPU) | `black-forest-labs/FLUX.1-Kontext-dev`; BFL's dedicated instruction editor (gated repo — needs HF auth) |
| `flux2-klein-9b` | local (GPU) | `black-forest-labs/FLUX.2-klein-9B`; distilled 9B sibling of FLUX.2-dev (gated repo — needs HF auth) |
| `bagel` | local (GPU) | `ByteDance-Seed/BAGEL-7B-MoT`; needs the upstream [BAGEL repo](https://github.com/bytedance-seed/BAGEL) (research-code drop, no pip package, weights-only HF card). `BAGELModel.load_model()` auto-clones the pinned upstream SHA into `$HF_HUB_CACHE/paintbench/bagel-upstream/` on first use (`git` must be on PATH). For air-gapped nodes, pre-clone manually and point at it via `BAGEL_REPO` / `--bagel-repo`. `flash_attn` is a runtime requirement on the GPU node — see [`CLAUDE.md`](CLAUDE.md) for the wheel-install recipe |
| `nano-banana-1` | API | Gemini-compatible image-edit model; supports custom gateway overrides via env/CLI (e.g. an OpenAI-compatible passthrough — see `.env.example`) |
| `nano-banana-2` | API | Gemini-compatible image-edit model; same custom-gateway support as `nano-banana-1` |
| `gpt-image-2` | API | GPT Image Edit model; async API path |
| `hunyuan-image-3` | local (multi-GPU) | `tencent/HunyuanImage-3.0-Instruct-Distil`; 80B MoE (13B active), **&ge; 8&times;80 GB VRAM**. 8-step distilled sampling. Weights download to `<HF cache>/paintbench/HunyuanImage-3-Instruct-Distil/` by default (respects `HF_HUB_CACHE` / `HUGGINGFACE_HUB_CACHE` / `HF_HOME` / `XDG_CACHE_HOME`, in that order); override with `--model-cache-dir` or `$HUNYUAN_MODEL_DIR` |
| `hunyuan-image-3-instruct` | local (multi-GPU) | `tencent/HunyuanImage-3.0-Instruct`; non-distilled flagship, **&ge; 8&times;80 GB VRAM**. Same Hunyuan integration with full-checkpoint sampling; default cache at `<HF cache>/paintbench/HunyuanImage-3-Instruct/` |

## Layout

```
CLAUDE.md                     # orientation for AI coding agents
CONTRIBUTING.md               # dev setup, PR conventions, release process
CHANGELOG.md                  # release notes
pyproject.toml                # project metadata, deps, ruff config
Makefile                      # common operations (make help)
scripts/
  submit_inference_h100.sh    # Slurm submission (modern clusters: uv + typed --gres)
  gen_website_examples.py     # generate one (input, answer) pair per task for docs/web
docs/
  run_cycle.md                # raw `uv run python` commands behind each make target
  metric.md                   # CIE76 / IoU / edit-accuracy explainer + aggregation hierarchy
  extending.md                # recipes for adding models, tasks, chart families
  paintbench-tasks.md         # PaintBench task reference (4 categories x 5 tasks, modes, visual conditions)
  tinygrafixbench-tasks.md    # TinyGrafixBench task reference (5 charts x 4 tasks)
tests/                        # `make test` — determinism + registry + eval + smoke
src/
  generate_benchmark.py       # CLI: PaintBench / TinyGrafixBench
  inference.py                # Run models, save outputs, log hardware metrics
  eval.py                     # Pointwise CIE76 → per-problem stats
  stats.py                    # Aggregate problem_stats.jsonl → aggregate rollups (with 95% bootstrap CIs)
  report.py                   # HTML report from aggregate stats
  visualize.py                # Interactive web visualizer (generate / benchmark / eval tabs)
  benchmark_source.py         # LocalBenchmarkSource + HfBenchmarkSource adapters
  core/                       # Shared primitives
    background.py             #   Background generation (solid, striped)
    canvas.py                 #   Scene composition
    colors.py                 #   Color palette and sampling
    shapes.py                 #   Shape geometry and rendering
  tasks/                      # One module per PaintBench task + base.py
    base.py                   #   Shared utilities (Problem dataclass, etc.)
    translation.py            #   ... one file per task (20 tasks + base)
  tinygrafixbench/            # Chart-edit mini-benchmark
    bar_chart.py              #   4 tasks per chart type
    heatmap.py                #
    line_chart.py             #
    network.py                #
    scatter_plot.py           #
    utils.py                  #   Shared helpers (colors, fonts, themes)
    fonts/DejaVuSans.ttf      #   Bundled for reproducible rendering
benchmarks/                   # (gitignored) generated benchmark outputs
model_outputs/                # (gitignored) raw model outputs from inference
eval_outputs/                 # (gitignored) derived eval stats + diff image cache
data/                         # (gitignored) scratch space
logs/                         # (gitignored) Slurm logs
```

## PaintBench tasks

20 tasks organised into 4 capability categories:

| Category | Tasks |
|---|---|
| **Geometric transformation** | translation, rotation, reflection, scaling, shearing |
| **Structural manipulation** | construction, removal, copying, border, cropping |
| **Color change** | recolor, flood_fill, blending, gradient, point_operations |
| **Symbolic reasoning** | comparison, ordering, pattern, counting, legend |

Each task is evaluated across **8 visual conditions × 12 problems** (1,920
scored problems total), plus a 96-problem `preservation/` diagnostic split
that probes input-fidelity floor.

See [`docs/paintbench-tasks.md`](docs/paintbench-tasks.md) for the full
per-task reference (modes, visual conditions, parameter grids) and
[`docs/tinygrafixbench-tasks.md`](docs/tinygrafixbench-tasks.md) for the
TinyGrafixBench equivalent.

## Extending

See [`docs/extending.md`](docs/extending.md) for full recipes covering:

- Adding a new image-edit model to the registry
- Adding a new PaintBench task
- Adding a new TinyGrafixBench chart family
- Building a sibling benchmark on top of the same eval pipeline

For PaintBench tasks specifically, the surface is small: every task in
`src/tasks/<name>.py` exports `NAME`, `PARAMETERS`, and `generate(...) -> Problem`,
then gets registered in the `TASKS` list at the top of
`src/generate_benchmark.py`. All randomness must go through a seeded
`random.Random` (no global `random`) — the determinism contract is
enforced by `tests/test_determinism.py`. See [`CLAUDE.md`](CLAUDE.md)
for the architectural conventions.

The full TinyGrafixBench task reference lives in
[`docs/tinygrafixbench-tasks.md`](docs/tinygrafixbench-tasks.md).

## Citation

```bibtex
@article{paintbench2026,
  title   = {{PaintBench}: Deterministic Evaluation of Precise Visual Editing},
  author  = {Xu, Kai and Brown, Ellis and Madhu, Shrikar and
             Fergus, Rob and He, He and Xie, Saining},
  journal = {arXiv preprint arXiv:<ARXIV-ID>},
  year    = {2026}
}
```
