#!/bin/bash
# Submit all four gate-sweep arrays with results + logs on scratch.
#
# Slurm only accepts submissions from the login nodes (every partition has
# AllocNodes=rhea,lov4), so run this from rhea or lov4 -- not from a compute
# node inside an interactive job, where sbatch fails with
# "Access/permission denied".
#
# Usage:
#   ssh rhea
#   cd /zfsauton2/home/wpotosna/neuralforecast/mica/training
#   bash slurm/submit_all.sh
#
# Optional: cap concurrent array tasks per job (default uncapped)
#   MAX_CONCURRENT=16 bash slurm/submit_all.sh

set -euo pipefail

# qos_general caps you at MaxSubmitJobsPU=50 and Slurm counts every array task
# individually, so submitting the full 0-389 / 0-239 / 0-314 arrays here fails with
# QOSMaxSubmitJobPerUserLimit. Use slurm/autosubmit.sh, which keeps ~45 tasks queued
# and refills as they finish. Set I_KNOW_ABOUT_THE_QOS_CAP=1 to run this anyway.
if [[ -z "${I_KNOW_ABOUT_THE_QOS_CAP:-}" ]]; then
    echo "ERROR: full-array submission exceeds the qos_general 50-job cap." >&2
    echo "       Use instead:  bash slurm/autosubmit.sh" >&2
    echo "       (override with I_KNOW_ABOUT_THE_QOS_CAP=1)" >&2
    exit 1
fi

SCRATCH=/zfsauton/scratch/wpotosna/neuralforecast_mica
export SAVE_DIR="${SAVE_DIR:-$SCRATCH/exp_results}"
export GIFT_EVAL_DIR="${GIFT_EVAL_DIR:-/zfsauton/scratch/wpotosna/GiftEval}"
LOG_DIR="${LOG_DIR:-$SCRATCH/logs}"
export LIGHTNING_ROOT="${LIGHTNING_ROOT:-$SCRATCH/lightning_runs}"
export NEURALFORECAST_RAY_STORAGE_PATH="${NEURALFORECAST_RAY_STORAGE_PATH:-$SCRATCH/ray_results}"

mkdir -p "$SAVE_DIR" "$LOG_DIR" "$LIGHTNING_ROOT" "$NEURALFORECAST_RAY_STORAGE_PATH" logs
test -f train_models.py  # must be submitted from mica/training

# job script -> its full array range (must match the #SBATCH --array in each)
JOBS=(
    "run_ciexcl_rerun 0-389"
    "run_new_datasets_full 0-239"
    "run_poolmean 0-314"
    "run_ecl_mlpquery 0-14"
)

for entry in "${JOBS[@]}"; do
    read -r script range <<<"$entry"
    array_arg=()
    if [[ -n "${MAX_CONCURRENT:-}" ]]; then
        array_arg=(--array="${range}%${MAX_CONCURRENT}")
    fi
    echo "Submitting ${script} (${range})"
    sbatch "${array_arg[@]}" \
        --output="${LOG_DIR}/%x_%A_%a.out" \
        --error="${LOG_DIR}/%x_%A_%a.err" \
        "slurm/${script}.sbatch"
done

echo
echo "SAVE_DIR : $SAVE_DIR"
echo "LOG_DIR  : $LOG_DIR"
echo "GIFT_EVAL: $GIFT_EVAL_DIR"
echo "TB logs : $LIGHTNING_ROOT"
echo "ray      : $NEURALFORECAST_RAY_STORAGE_PATH"
