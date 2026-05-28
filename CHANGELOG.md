# Changelog

All notable changes to PaintBench are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-27

Initial public release accompanying the arXiv paper.

### Benchmarks
- **PaintBench** — 20 tasks across 4 capability categories (geometric
  transformation, structural manipulation, color change, symbolic reasoning),
  evaluated under 8 visual conditions × 12 problems = 1,920 scored problems,
  plus a 96-problem `preservation/` diagnostic split excluded from scoring.
- **TinyGrafixBench** — chart-edit analog of the primitive ops: 5 matplotlib
  chart families × 4 tasks × 30 seeds = 600 problems.

### Models
- 12 image-editing model registry entries spanning local diffusers
  (InstructPix2Pix, LongCat, Qwen-Image-Edit, FLUX.2-dev, FLUX.1-Kontext-dev,
  FLUX.2-Klein-9B, BAGEL, Hunyuan-Image-3 Distil and full), and remote APIs
  (Nano Banana 1/2, GPT-Image-2).
- `HfBenchmarkSource` (`src/benchmark_source.py`) enables
  `--benchmark hf:<repo>@<sha>` to load a published benchmark from
  Hugging Face Hub without a local clone, with revision pinning for
  reproducibility.

### Metric
- Pointwise CIE76 (Lab Euclidean) distance across 11 thresholds (0–10),
  yielding per-problem edit_accuracy, preservation_accuracy, and IoU on the
  edit / preservation mask split.
- Bootstrap-CI rollups via 2-level and 3-level hierarchical resampling matched
  to each aggregation level (B=10,000, seed=0). See [`docs/metric.md`](docs/metric.md).

### Reproducibility
- Deterministic generation across `PYTHONHASHSEED` values, enforced by
  `tests/test_determinism.py` (byte-hash pins) and `Makefile` prepending
  `PYTHONHASHSEED=0` to every artifact-writing target.
- `uv.lock` committed for reproducible installs across machines.
- Bundled `DejaVuSans.ttf` so chart rendering doesn't depend on system font
  availability.

[1.0.0]: https://github.com/PaintBench/PaintBench/releases/tag/v1.0.0
