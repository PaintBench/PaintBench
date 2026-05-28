#!/usr/bin/env bash
# Submit Slurm jobs (one per model × benchmark) on a modern SLURM cluster
# that uses typed --gres=gpu:<type>:N syntax (h100/h200/a100/v100/l40s).
#
# This script is fully env-var driven so it stays usable across clusters,
# accounts, and projects. All cluster-specific values (partition, account,
# QoS, GPU type, paths) come from env vars or a sourced .env — never
# hardcoded. The only assumptions are:
#
#   * the cluster runs SLURM
#   * `uv` is on PATH and the repo's pyproject.toml works under `uv sync`
#   * GPU nodes expose --gres=gpu:<type>:N (with <type> like "h100", "h200")
#
# Required env vars (script errors loudly if unset):
#
#   SLURM_PARTITION    e.g. h100, h200, gpu, dev
#   SLURM_ACCOUNT      cluster billing account
#
# Recommended (with sensible defaults):
#
#   SLURM_QOS          QoS name (omit for cluster default)
#   SLURM_GPU_TYPE     gres GPU type. Defaults to SLURM_PARTITION if it
#                      looks like an h100/h200/a100/v100 partition, else
#                      omits --gres type prefix (plain --gres=gpu:N).
#   SLURM_GPUS         GPUs per task                          (default 1)
#   SLURM_CPUS         CPUs per task                          (default 16)
#   SLURM_MEM          memory                                 (default 400G)
#   SLURM_TIME         walltime  HH:MM:SS or D-HH:MM:SS       (default 24:00:00)
#
# Job content:
#
#   MODELS             space-separated registry keys          (default: instruct-pix2pix)
#   BENCHMARKS         space-separated benchmark paths        (default: benchmarks/PaintBench)
#   WORKERS            --workers value for inference          (default 4)
#   EXTRA_INFERENCE_ARGS  appended to `python src/inference.py` (default: empty)
#                      e.g. "--max-problems 2 --tasks blending --overwrite"
#
# Layout:
#
#   WORKDIR            project root with src/ + pyproject.toml (default: $(pwd))
#   LOGDIR             where Slurm .out files land             (default: $WORKDIR/logs)
#
# Example (single smoke job, dev QoS):
#
#   SLURM_PARTITION=h200 SLURM_ACCOUNT=<acct> SLURM_QOS=<dev_qos> \
#   SLURM_TIME=01:00:00 \
#   MODELS=instruct-pix2pix \
#   EXTRA_INFERENCE_ARGS="--max-problems 2 --tasks blending --overwrite" \
#   bash scripts/submit_inference_h100.sh
#
# Example (overnight production matrix on H200):
#
#   SLURM_PARTITION=h200 SLURM_ACCOUNT=<acct> SLURM_QOS=<prod_qos> \
#   SLURM_TIME=24:00:00 \
#   MODELS="instruct-pix2pix qwen-image-edit flux2-dev" \
#   BENCHMARKS="benchmarks/PaintBench benchmarks/TinyGrafixBench" \
#   WORKERS=8 \
#   bash scripts/submit_inference_h100.sh
#
# Sourcing a .env: this script does NOT auto-source .env (Slurm submission
# is short-lived and explicit values are preferable for reproducibility).
# If you keep cluster knobs in .env, source it yourself first:
#   set -a; source .env; set +a; bash scripts/submit_inference_h100.sh

set -euo pipefail

# ── Required cluster identity ───────────────────────────────────────────────
PARTITION="${SLURM_PARTITION:?SLURM_PARTITION is required (e.g. h100, h200)}"
ACCOUNT="${SLURM_ACCOUNT:?SLURM_ACCOUNT is required}"
QOS="${SLURM_QOS:-}"

# ── Optional resource knobs ─────────────────────────────────────────────────
GPUS="${SLURM_GPUS:-1}"
CPUS="${SLURM_CPUS:-16}"
MEM="${SLURM_MEM:-400G}"
TIME="${SLURM_TIME:-24:00:00}"

# Default GPU type to the partition's GPU family if recognized.
# This makes --gres=gpu:h200:1 work transparently on clusters that require it.
# We extract just the family name (h100/h200/…) rather than the full partition
# string, so partitions with suffixes like "h200_shared" / "h100_dev" still
# resolve to a Slurm-valid gres type.
default_gpu_type=""
case "${PARTITION,,}" in
    h100*) default_gpu_type="h100" ;;
    h200*) default_gpu_type="h200" ;;
    a100*) default_gpu_type="a100" ;;
    v100*) default_gpu_type="v100" ;;
    l40s*) default_gpu_type="l40s" ;;
esac
GPU_TYPE="${SLURM_GPU_TYPE-$default_gpu_type}"

# ── Job content ─────────────────────────────────────────────────────────────
read -r -a MODELS_ARR <<< "${MODELS:-instruct-pix2pix}"
read -r -a BENCHMARKS_ARR <<< "${BENCHMARKS:-benchmarks/PaintBench}"
WORKERS="${WORKERS:-4}"
EXTRA_INFERENCE_ARGS="${EXTRA_INFERENCE_ARGS:-}"

# Guard against whitespace-only assignments in a sourced .env (e.g.
# `MODELS=" "` from a stale comma-list edit). Bash's :- defaults handle
# the literal-empty case (`MODELS=""` → "instruct-pix2pix"), but
# whitespace-only values bypass the default and `read -a` then yields
# an empty array — without this guard, that silently submits zero jobs.
[[ ${#MODELS_ARR[@]} -eq 0 ]] && { echo "ERROR: MODELS resolves to empty (check your .env / shell env)" >&2; exit 1; }
[[ ${#BENCHMARKS_ARR[@]} -eq 0 ]] && { echo "ERROR: BENCHMARKS resolves to empty (check your .env / shell env)" >&2; exit 1; }

# ── Paths ───────────────────────────────────────────────────────────────────
WORKDIR="${WORKDIR:-$(pwd)}"
LOGDIR="${LOGDIR:-${WORKDIR}/logs}"
mkdir -p "$LOGDIR"

# ── Compose --gres flag ─────────────────────────────────────────────────────
if [[ -n "$GPU_TYPE" ]]; then
    GRES="gpu:${GPU_TYPE}:${GPUS}"
else
    GRES="gpu:${GPUS}"
fi

submit() {
    local model="$1" benchmark="$2"
    local bench_name; bench_name=$(basename "$benchmark")

    local cmd="cd ${WORKDIR} && uv run python src/inference.py"
    cmd+=" --model ${model}"
    cmd+=" --benchmark ${benchmark}"
    cmd+=" --out-dir model_outputs"
    cmd+=" --workers ${WORKERS}"
    [[ -n "$EXTRA_INFERENCE_ARGS" ]] && cmd+=" ${EXTRA_INFERENCE_ARGS}"

    local sbatch_args=(
        --partition="$PARTITION"
        --account="$ACCOUNT"
        --nodes=1
        --ntasks-per-node=1
        --gres="$GRES"
        --cpus-per-task="$CPUS"
        --mem="$MEM"
        --time="$TIME"
        --job-name="inference-${model}-${bench_name}"
        --output="${LOGDIR}/inference-${model}-${bench_name}-%j.out"
    )
    [[ -n "$QOS" ]] && sbatch_args+=( --qos="$QOS" )

    echo "Submitting: ${model} × ${bench_name}"
    echo "  partition=${PARTITION}  account=${ACCOUNT}  qos=${QOS:-<default>}  gres=${GRES}  time=${TIME}"
    sbatch "${sbatch_args[@]}" --wrap "$cmd"
}

n_jobs=$(( ${#MODELS_ARR[@]} * ${#BENCHMARKS_ARR[@]} ))
echo "Submitting ${n_jobs} job(s):"
echo "  models     : ${MODELS_ARR[*]}"
echo "  benchmarks : ${BENCHMARKS_ARR[*]}"
echo "  workdir    : ${WORKDIR}"
echo "  logdir     : ${LOGDIR}"
echo ""

for model in "${MODELS_ARR[@]}"; do
    for benchmark in "${BENCHMARKS_ARR[@]}"; do
        submit "$model" "$benchmark"
    done
done

echo ""
echo "All ${n_jobs} jobs submitted."
echo "Monitor with:      squeue -u \$USER"
echo "Watch a log with:  tail -f ${LOGDIR}/inference-<model>-<benchmark>-<jobid>.out"
