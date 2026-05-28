# PaintBench — full run cycle

All commands run from the repo root. The `make` targets below wrap the raw
`uv run python` commands — use the raw form inside a Slurm `--wrap`, or
`scripts/submit_inference_h100.sh` for the whole grid.

## 1. Generate the benchmarks

```bash
make generate-all
# raw:
uv run python src/generate_benchmark.py --paintbench --tinygrafixbench --output benchmarks
```

Writes `benchmarks/PaintBench/` and `benchmarks/TinyGrafixBench/`.

## 2. Run inference (per model × benchmark)

Outputs land in `model_outputs/<model>/<benchmark>/`.

```bash
make inference MODEL=flux2-dev BENCHMARK=PaintBench

# raw:
uv run python src/inference.py \
    --model flux2-dev \
    --benchmark benchmarks/PaintBench \
    --out-dir model_outputs
```

On the cluster you can submit all GPU (model × benchmark) pairs in one shot
via the bundled submit script (modern clusters with typed
`--gres=gpu:<type>:N` for h100/h200/a100/v100/l40s, `uv run` directly).
Requires `SLURM_PARTITION` + `SLURM_ACCOUNT` in env or `.env`; everything
else has sensible defaults. See the script header for the full env-var
reference (`SLURM_QOS`, `MODELS`, `BENCHMARKS`, `EXTRA_INFERENCE_ARGS`, …).

```bash
SLURM_PARTITION=h200 SLURM_ACCOUNT=<acct> bash scripts/submit_inference_h100.sh
```

### Parallelism via task split

`src/inference.py --tasks <a,b,c>` runs the listed tasks sequentially on
one GPU. For wall-time wins on a multi-GPU cluster, submit one Slurm job
per task instead — they share `--out-dir model_outputs` safely because
the inference cache validates each existing `<idx>_output.png` with
`PIL.Image.load()` before deciding to skip, so cross-job dup-write
windows are small and never produce corruption.

Example: re-inferring 3 affected tasks (`removal`, `recolor`, `legend`)
across 6 models as 18 single-GPU jobs in parallel:

```bash
for MODEL in bagel qwen-image-edit flux1-kontext-dev flux2-dev flux2-klein-9b instruct-pix2pix; do
  for TASK in removal recolor legend; do
    sbatch \
        --partition=<your-partition> --account=<your-account> \
        --gres=gpu:1 --time=04:00:00 \
        --job-name="reinf-${MODEL}-${TASK}" \
        --output="logs/reinf-${MODEL}-${TASK}-%j.out" \
        --wrap "uv run python src/inference.py \
          --model ${MODEL} --benchmark benchmarks/PaintBench --tasks ${TASK} \
          --out-dir model_outputs --workers 4"
  done
done
```

~3× wall-time speedup per model vs. one job per model with `--tasks
removal,recolor,legend`. Zero code changes — all through the existing
CLI. `make inference` itself doesn't shard automatically (a model ×
benchmark pair maps to one process); do the sharding at the slurm-submit
layer.

## 3. Evaluate (pointwise CIE76)

```bash
make eval
# raw:
uv run python src/eval.py \
    --benchmarks benchmarks \
    --model-outputs model_outputs \
    --eval-outputs eval_outputs
```

Writes `eval_outputs/problem_stats.jsonl` AND caches per-problem ΔE diff
PNGs (4 thresholds: 0, 2, 5, 10) plus the normalized output (only when it
differs from the raw output) under `eval_outputs/<model>/<bench>/<task>/`.
The visualizer reads these caches in the Eval tab. For a stats-only fast
eval (no PNG writes — useful when iterating on metric numbers and `make
viz` will recompute on first view), run:

```bash
make eval-quick
```

## 4. Aggregate stats

```bash
make stats
# raw:
uv run python src/stats.py --input eval_outputs/problem_stats.jsonl --output eval_outputs/aggregate_stats.jsonl
```

## 5. HTML report

```bash
make report
# raw:
uv run python src/report.py --input eval_outputs/aggregate_stats.jsonl --output report.html
```

Open `report.html` in a browser.

## 6. Interactive visualizer (optional)

```bash
make viz
# raw:
uv run python src/visualize.py --benchmarks benchmarks --model-outputs model_outputs --eval-outputs eval_outputs
```

Then open http://localhost:8765 — tabs: Generate, Benchmark, Eval.

## (Optional) Load from a published HF dataset

Skip step 1 (local generation) by pointing `--benchmark` at a Hugging
Face dataset repo:

```bash
make setup-data    # one-time: installs `datasets` + `huggingface-hub`

uv run python src/inference.py \
    --model flux2-dev \
    --benchmark hf:PaintBench/PaintBench \
    --benchmark-config PaintBench \
    --out-dir model_outputs

# Pin a specific dataset revision (commit SHA or tag) for reproducibility:
uv run python src/inference.py \
    --model flux2-dev \
    --benchmark hf:PaintBench/PaintBench@v1.0.0 \
    --benchmark-config PaintBench
```

`--benchmark-config` is required when reading from HF — it selects
`PaintBench` vs `TinyGrafixBench` (published as separate configs on
the same dataset repo).
