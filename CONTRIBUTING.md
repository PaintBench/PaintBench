# Contributing to PaintBench

Thanks for your interest in PaintBench. This document covers what you
need to know to get a working dev setup, run the test suite, and submit
a useful PR.

## Dev setup

Requires Python >= 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/PaintBench/PaintBench.git
cd PaintBench
make setup    # uv sync --all-groups → core + dev deps in .venv/
```

For local-GPU model work, also install the inference extra:

```bash
make setup-inference   # + torch / diffusers / transformers
```

For API-backed models, copy `.env.example` to `.env` and fill in the
relevant keys.

## Verifying your change

Two commands must pass before any PR:

```bash
make lint     # ruff over src/, scripts/, tests/
make test     # pytest — ~30s, determinism + registry + eval + smoke
```

CI runs the same two commands on every push / pull request, so a green
local run means a green CI run.

If you change a task generator or any code that affects benchmark
output, also re-run a sample generation to check the diff visually:

```bash
make generate-all JOBS=1                    # ~6 min on a laptop
ls benchmarks/PaintBench/<your-task>/       # eyeball a few _input/_answer PNGs
```

On sandboxed shells (Cursor's macOS terminal etc.) where POSIX
semaphores are blocked, pass `JOBS=1` to avoid the `PermissionError`
from `ProcessPoolExecutor`. See [`CLAUDE.md`](CLAUDE.md) for details.

## Where things live

Read [`CLAUDE.md`](CLAUDE.md) first — it has the canonical map of
where each piece of functionality lives, plus the architectural
conventions that the test suite enforces.

For the public API:

- `src/inference.py` — model registry (`_REGISTRY`), `BaseModel`
  interface, all the model implementations
- `src/tasks/<name>.py` — PaintBench task generators (one per task)
- `src/tinygrafixbench/<chart>.py` — TinyGrafixBench chart generators
- `src/eval.py` — pointwise CIE76 metric (see [`docs/metric.md`](docs/metric.md))
- `src/stats.py` — aggregate rollups with bootstrap CIs
- `src/benchmark_source.py` — `LocalBenchmarkSource` / `HfBenchmarkSource` adapters

## Adding a model, task, or benchmark variant

See [`docs/extending.md`](docs/extending.md) for the step-by-step recipes.

## PR conventions

- **One thing per PR.** A new model, a metric tweak, and a doc fix
  should be three PRs. Easier to review, easier to revert.
- **Update the test suite** for any user-visible behaviour change.
  `tests/test_determinism.py` pins byte hashes for every task — if you
  change a generator's RNG path, update the expected hashes in the
  same commit.
- **Update [`CHANGELOG.md`](CHANGELOG.md)** under the `[Unreleased]`
  section.
- **Commit message:** short imperative subject (<=72 chars); body
  explains the *why*, not a restatement of the diff. We squash-merge,
  so the commit you send is the commit that lands.
- **Lint and test must pass** in CI before review. Open as a draft PR
  if the work is still in progress.

## Releasing

Maintainer-only:

1. Bump `version` in `pyproject.toml`.
2. Move `CHANGELOG.md`'s `[Unreleased]` block under a new versioned
   heading with the release date.
3. Tag the commit (`git tag v<version> && git push --tags`).
4. Cut a GitHub release pointing at the tag.

## Questions

For bug reports and feature requests, open an issue. For design
discussions, open a discussion. For security issues, please email
the maintainers privately (see paper for contact).
