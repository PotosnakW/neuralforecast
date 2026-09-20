#!/bin/bash
# Zero-shot timesfm3 baseline over every dataset in the MICA sweep, for direct
# comparison against chronos2.0_baseline and the trained gate variants.
#
# Runs from its own conda env (mica_zs_timesfm): timesfm3 pulls dependencies that
# conflict with envs/neuralforecast, so it is deliberately NOT installed there --
# that env is serving the 945-task gate sweep. neuralforecast and gift_eval are
# picked up via PYTHONPATH rather than installed, so no env owns a second copy
# of the repo and nothing is written into it.
#
# Zero-shot inference is deterministic, so this runs seed 1 only -- extra seeds
# would reproduce identical forecasts at full GPU cost.
#
# Usage:
#   cd mica/training && sbatch slurm/run_zeroshot_timesfm3.sbatch
#
#SBATCH --job-name=mica_zs_timesfm3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-20

set -euo pipefail

CONDA_BASE="/zfsauton2/home/wpotosna/miniconda3"
PYTHON="${CONDA_BASE}/envs/mica_timesfm3/bin/python"
EXPERIMENT_NAME=timesfm3_baseline

REPO=/zfsauton2/home/wpotosna/neuralforecast
export PYTHONPATH="${REPO}:/zfsauton2/home/wpotosna/gift-eval/src${PYTHONPATH:+:$PYTHONPATH}"

GIFT_EVAL_DIR=${GIFT_EVAL_DIR:-/zfsauton/scratch/wpotosna/GiftEval}
SAVE_DIR=${SAVE_DIR:-/zfsauton/scratch/wpotosna/neuralforecast_mica/exp_results}
LIGHTNING_ROOT=${LIGHTNING_ROOT:-/zfsauton/scratch/wpotosna/neuralforecast_mica/lightning_runs}
export GIFT_EVAL="${GIFT_EVAL_DIR}"
export TMPDIR=${TMPDIR:-/zfsauton/scratch/wpotosna/tmp}
export RAY_TMPDIR="${TMPDIR}"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/zfsauton/scratch/wpotosna/neuralforecast_mica/cache}
export HF_HOME=${HF_HOME:-${XDG_CACHE_HOME}/huggingface}
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}"

REPO_DIR=${SLURM_SUBMIT_DIR:?submit from mica/training/}
cd "${REPO_DIR}"
test -f zeroshot_models.py

DATASETS=(
    simglucose
    iowa_ihop_smex_windspeed
    iowa_plows_windspeed
    LOOP_SEATTLE/D
    M_DENSE/D
    M_DENSE/H
    covid_deaths
    ett1/D
    ett1/H
    ett1/W
    ett2/D
    ett2/H
    ett2/W
    jena_weather/D
    jena_weather/H
    solar/D
    solar/H
    solar/W
    electricity/D
    electricity/H
    electricity/W
)
DATASET_NAME=${DATASETS[$SLURM_ARRAY_TASK_ID]}
SEED=1

export LIGHTNING_DIR="${LIGHTNING_ROOT}/${SLURM_JOB_NAME:-local}_${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p logs "${LIGHTNING_DIR}"

echo "=========================================="
echo "Job ID      : ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
echo "Node        : ${SLURMD_NODENAME:-$(hostname)}"
echo "Experiment  : $EXPERIMENT_NAME"
echo "Dataset     : $DATASET_NAME"
echo "Seed        : $SEED  (zero-shot: deterministic)"
echo "Start       : $(date)"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"

srun "$PYTHON" -m zeroshot_models \
    --dataset_name "${DATASET_NAME}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --save_dir "${SAVE_DIR}" \
    --random_seed "${SEED}" \
    --num_samples 1 \
    --input_size_h_multiplier 2

echo "=========================================="
echo "Done : $(date)"
echo "=========================================="
