.DEFAULT_GOAL := help

# ── Config ──────────────────────────────────────────────────────────────────────
MODEL      ?= instruct-pix2pix
BENCHMARK  ?= PaintBench
BENCHDIR   ?= benchmarks
RESULTSDIR ?= model_outputs
STATSDIR   ?= eval_outputs
# JOBS — parallelism knob for generate (--jobs), eval (--workers), and stats (--workers).
# Empty (default) lets each script pick os.cpu_count(). Set JOBS=1 to disable
# multiprocessing — needed when running from a sandboxed terminal (e.g. Cursor
# on macOS) where POSIX semaphore creation is blocked. Each *_ARG expands to
# the matching CLI flag only when JOBS is set.
JOBS         ?=
JOBS_ARG     = $(if $(strip $(JOBS)),--jobs $(JOBS),)
WORKERS_ARG  = $(if $(strip $(JOBS)),--workers $(JOBS),)

# WORKERS — concurrent calls per task for inference. Independent of JOBS:
# inference is I/O-bound and uses threads (no multiprocessing), so it isn't
# subject to the macOS sandbox restriction that motivated JOBS=1. Use 4–8 for
# remote API models; default empty = sequential (--workers 1).
WORKERS              ?=
INF_WORKERS_ARG      = $(if $(strip $(WORKERS)),--workers $(WORKERS),)

# Optional per-run Gemini overrides — empty by default, so inference falls
# back to .env / class defaults. Useful for running variants concurrently
# (e.g. an older Flash variant in a second terminal) without touching .env.
GEMINI_MODEL_NAME       ?=
GEMINI_BASE_URL         ?=
GEMINI_INCLUDE_THOUGHTS ?=
GEMINI_MODEL_ARG        = $(if $(strip $(GEMINI_MODEL_NAME)),--model-name $(GEMINI_MODEL_NAME),)
GEMINI_URL_ARG          = $(if $(strip $(GEMINI_BASE_URL)),--base-url $(GEMINI_BASE_URL),)
GEMINI_THOUGHTS_ARG     = $(if $(strip $(GEMINI_INCLUDE_THOUGHTS)),--include-thoughts $(GEMINI_INCLUDE_THOUGHTS),)

# RETRY_FAILED — path to a metrics_*.json from a prior run; inference
# re-runs only the problems that failed there. Output goes into the same
# tree (overwriting any stale PNGs) and writes a fresh metrics_*.json.
RETRY_FAILED         ?=
RETRY_FAILED_ARG     = $(if $(strip $(RETRY_FAILED)),--retry-failed $(RETRY_FAILED),)

# OVERWRITE — by default, both inference and eval are incremental:
#   inference: existing <NNNN>_output.png in --out-dir is reused after a
#              Pillow .load() decode check.
#   eval:      existing <NNNN>_stats.json sidecars in --results are reused
#              when their mtime ≥ each source PNG's mtime.
# Reruns after Ctrl-C / cancel / partial inference only redo the work
# that's missing or whose inputs changed. Set OVERWRITE=1 when you
# change inference logic (prompt, sampling, gateway routing, ...) or
# eval logic (CIE76 math, threshold list, normalization, ...) and want
# to invalidate the cache without manually deleting the output tree.
# Composes with RETRY_FAILED on the inference side.
OVERWRITE            ?=
OVERWRITE_ARG        = $(if $(strip $(OVERWRITE)),--overwrite,)

# MAX_PROBLEMS — cap problems-per-task. Useful for stratified sanity
# runs (MAX_PROBLEMS=1 → one problem per task-mode). Empty = all.
MAX_PROBLEMS         ?=
MAX_PROBLEMS_ARG     = $(if $(strip $(MAX_PROBLEMS)),--max-problems $(MAX_PROBLEMS),)

# TASKS — comma-separated task-mode filter (e.g. blending,rotation).
# Empty = all task-modes in the benchmark.
TASKS                ?=
TASKS_ARG            = $(if $(strip $(TASKS)),--tasks $(TASKS),)

# REPORT_EXCLUDE — comma-separated substrings; rows whose model name
# contains any of them are dropped from the rendered report.html.
# aggregate_stats.jsonl is untouched.
REPORT_EXCLUDE       ?=
REPORT_EXCLUDE_ARG   = $(if $(strip $(REPORT_EXCLUDE)),--exclude-models $(REPORT_EXCLUDE),)

# ── Setup ───────────────────────────────────────────────────────────────────────
# BAGEL: no separate setup target — src/inference.py auto-clones the pinned
# upstream into the HF cache on first load_model(). For offline / air-gapped
# nodes, pre-clone manually and point at it via BAGEL_REPO / --bagel-repo.
.PHONY: setup setup-inference setup-api setup-data

setup: ## Install core + dev dependencies
	uv sync --all-groups

setup-inference: ## Install core + local inference dependencies (needs GPU)
	uv sync --extra inference

setup-api: ## Install core + remote API inference dependencies (Gemini / Nano Banana)
	uv sync --extra api

setup-data: ## Install core + HF dataset reader (datasets, huggingface-hub) — enables --benchmark hf:<repo>
	uv sync --extra data

# ── Generate ────────────────────────────────────────────────────────────────────
.PHONY: generate generate-all generate-tinygrafixbench regenerate

# PYTHONHASHSEED=0 pins set / dict iteration order across Python processes.
# Required to keep benchmark generation byte-deterministic — task generators
# must never iterate sets of strings (or string-containing tuples) without
# wrapping in sorted(), but pinning here is belt-and-suspenders so a future
# regression can't silently change benchmark answers. Applied to every
# pipeline stage that produces a tracked artifact (benchmark images, eval
# sidecars, aggregate stats, HTML report) so any new dict/set iteration over
# strings introduced in those scripts can't silently shift bytes.
DETERMINISTIC_ENV := PYTHONHASHSEED=0
# Back-compat alias.
GEN_ENV := $(DETERMINISTIC_ENV)

generate: ## Generate PaintBench → benchmarks/
	$(GEN_ENV) uv run python src/generate_benchmark.py --paintbench --output $(BENCHDIR) $(JOBS_ARG)

generate-all: ## Generate PaintBench + TinyGrafixBench
	$(GEN_ENV) uv run python src/generate_benchmark.py --paintbench --tinygrafixbench --output $(BENCHDIR) $(JOBS_ARG)

generate-tinygrafixbench: ## Generate TinyGrafixBench only
	$(GEN_ENV) uv run python src/generate_benchmark.py --tinygrafixbench --output $(BENCHDIR) $(JOBS_ARG)

regenerate: ## Regenerate specific PaintBench problems  (PROBLEMS=task:id[,task:id...])
	@[ -n "$(strip $(PROBLEMS))" ] || (echo "Set PROBLEMS=task:id,task:id (e.g. PROBLEMS=translation:011)"; exit 1)
	$(GEN_ENV) uv run python src/generate_benchmark.py --regenerate $(PROBLEMS) --output $(BENCHDIR)/PaintBench

# ── Inference ───────────────────────────────────────────────────────────────────
.PHONY: inference inference-slurm-h100

inference: ## Run inference  (incremental by default; OVERWRITE=1 forces redo. MODEL=..., BENCHMARK=..., WORKERS=..., MAX_PROBLEMS=..., TASKS=..., RETRY_FAILED=..., GEMINI_MODEL_NAME=..., ...)
	uv run python src/inference.py \
		--model $(MODEL) \
		--benchmark $(BENCHDIR)/$(BENCHMARK) \
		--out-dir $(RESULTSDIR) \
		$(INF_WORKERS_ARG) $(MAX_PROBLEMS_ARG) $(TASKS_ARG) $(GEMINI_MODEL_ARG) $(GEMINI_URL_ARG) $(GEMINI_THOUGHTS_ARG) $(RETRY_FAILED_ARG) $(OVERWRITE_ARG)

inference-slurm-h100: ## Submit all (model × benchmark) pairs to a modern SLURM cluster (uv + typed --gres). Requires SLURM_PARTITION + SLURM_ACCOUNT — see script header for full env-var reference.
	bash scripts/submit_inference_h100.sh

# ── Evaluate / report ──────────────────────────────────────────────────────────
.PHONY: eval eval-quick stats report

eval: ## Evaluate model outputs (CIE76 per-pixel) → $(STATSDIR)/problem_stats.jsonl + cached ΔE diff PNGs (used by `make viz`); incremental, OVERWRITE=1 forces redo
	$(DETERMINISTIC_ENV) uv run python src/eval.py \
		--benchmarks $(BENCHDIR) \
		--model-outputs $(RESULTSDIR) \
		--eval-outputs $(STATSDIR) \
		$(WORKERS_ARG) $(OVERWRITE_ARG)

eval-quick: ## Stats-only eval (skip diagnostic ΔE diff PNG writes; ~1.4× faster on the canonical PaintBench grid, more on larger ones; `make viz` will then compute diff PNGs on first view); incremental, OVERWRITE=1 forces redo
	$(DETERMINISTIC_ENV) uv run python src/eval.py \
		--benchmarks $(BENCHDIR) \
		--model-outputs $(RESULTSDIR) \
		--eval-outputs $(STATSDIR) \
		--no-save-images $(WORKERS_ARG) $(OVERWRITE_ARG)

stats: ## Aggregate per-problem stats → $(STATSDIR)/aggregate_stats.jsonl (with 95% bootstrap CIs by default; ~30-60s on a 12-core box via per-model parallelism, ~6 min serial). Pass NO_CI=1 to skip the bootstrap step for fast iteration (~3s); pass JOBS=1 to disable parallelism (e.g. in Cursor's macOS sandbox). Matches `make generate` / `make eval` defaults (os.cpu_count()).
	$(DETERMINISTIC_ENV) uv run python src/stats.py \
		--input  $(STATSDIR)/problem_stats.jsonl \
		--output $(STATSDIR)/aggregate_stats.jsonl \
		$(if $(strip $(NO_CI)),--no-ci,) \
		$(WORKERS_ARG)

report: ## Build HTML report from aggregate stats → report.html
	$(DETERMINISTIC_ENV) uv run python src/report.py \
		--input  $(STATSDIR)/aggregate_stats.jsonl \
		--output report.html \
		$(REPORT_EXCLUDE_ARG)

# ── Visualization ──────────────────────────────────────────────────────────────
.PHONY: viz

viz: ## Launch interactive web visualizer (Generate / Benchmark / Eval tabs)
	uv run python src/visualize.py \
		--benchmarks    $(BENCHDIR) \
		--model-outputs $(RESULTSDIR) \
		--eval-outputs  $(STATSDIR)

# ── Code Quality ────────────────────────────────────────────────────────────────
.PHONY: lint format test

lint: ## Lint with ruff
	uv run ruff check src/ scripts/ tests/

format: ## Auto-format with ruff
	uv run ruff format src/

test: ## Run the pytest suite (determinism, registry, eval math, pipeline smoke)
	uv run pytest tests/ -v

# ── Help ────────────────────────────────────────────────────────────────────────
.PHONY: help

help: ## Show available targets
	@printf "\nUsage: make \033[36m<target>\033[0m [VAR=value ...]\n\n"
	@printf "Targets:\n"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@printf "\nVariables (override on command line):\n"
	@printf "  \033[36m%-22s\033[0m %s\n" "MODEL"      "Model name         (default: $(MODEL))"
	@printf "  \033[36m%-22s\033[0m %s\n" "BENCHMARK"  "Benchmark name     (default: $(BENCHMARK))"
	@printf "  \033[36m%-22s\033[0m %s\n" "BENCHDIR"   "Benchmark root     (default: $(BENCHDIR))"
	@printf "  \033[36m%-22s\033[0m %s\n" "RESULTSDIR" "Model outputs root (default: $(RESULTSDIR))"
	@printf "  \033[36m%-22s\033[0m %s\n" "STATSDIR"   "Eval stats root    (default: $(STATSDIR))"
	@printf "  \033[36m%-22s\033[0m %s\n" "JOBS"       "Parallelism for generate + eval + stats (set JOBS=1 if multiprocessing is blocked, e.g. in Cursor on macOS)"
	@printf "  \033[36m%-22s\033[0m %s\n" "WORKERS"    "Concurrent inference calls per task (4-8 typical for remote APIs)"
	@printf "  \033[36m%-22s\033[0m %s\n" "OVERWRITE"  "Set to 1 to invalidate inference output PNG cache + eval sidecar cache (default: incremental, reuse if present and fresh)"
	@printf "  \033[36m%-22s\033[0m %s\n" "MAX_PROBLEMS" "Cap problems per task-mode (1 = stratified one-per-mode sanity run)"
	@printf "  \033[36m%-22s\033[0m %s\n" "TASKS"      "Comma-separated task-mode filter (e.g. blending,rotation)"
	@printf "  \033[36m%-22s\033[0m %s\n" "RETRY_FAILED" "Path to prior metrics_*.json — re-runs only the problems that failed there"
	@printf "  \033[36m%-22s\033[0m %s\n" "REPORT_EXCLUDE" "Comma-separated model-name substrings dropped from report.html (default: empty)"
	@printf "\nExamples:\n"
	@printf "  make generate-all                                              # PaintBench + TinyGrafixBench\n"
	@printf "  make inference MODEL=flux2-dev BENCHMARK=PaintBench            # run one model (incremental — only redoes missing PNGs)\n"
	@printf "  make inference MODEL=nano-banana-1 BENCHMARK=PaintBench \\\\\n"
	@printf "                 WORKERS=8 OVERWRITE=1                            # force redo all (e.g. after inference code change)\n"
	@printf "  make eval stats report                                         # full evaluation pipeline\n"
	@printf "  make viz                                                       # open interactive visualizer at :8765\n"
	@printf "\n"
