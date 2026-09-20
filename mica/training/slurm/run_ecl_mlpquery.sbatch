#!/bin/bash
# Ask 1: infini_mlpquerymixer_t5tiny (MLP-query gate) on the electricity (ECL)
# datasets, which are newly added to DATASET_NAMES and haven't been run
# before -- they needed the windows_batch_size=8 OOM fix for electricity's
# 321 series (see experiment_models.py, applied automatically off
# args.dataset_name). Each array task trains one (dataset, seed) pair on its
# own GPU, so the scheduler packs/parallelizes across every GPU you have
# access to instead of serializing everything in one long-running script.
#
# Usage:
#   cd mica/training && mkdir -p logs
#   sbatch slurm/run_ecl_mlpquery.sbatch
# To cap concurrent tasks at your node's GPU count:
#   sbatch --array=0-14%<N_GPUS> slurm/run_ecl_mlpquery.sbatch
#
#SBATCH --job-name=mica_ecl_mlpquery
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-14

set -euo pipefail

CONDA_BASE="/zfsauton2/home/wpotosna/miniconda3"
CONDA_ENV="neuralforecast"
# Absolute path, not `conda run -n neuralforecast python` -- conda run resolves
# an unqualified `python` via $PATH, which silently picks up another env's
# interpreter if one is hardcoded ahead of it there.
PYTHON="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"
GIFT_EVAL_DIR=${GIFT_EVAL_DIR:-/zfsauton/scratch/wpotosna/GiftEval}
SAVE_DIR=${SAVE_DIR:-/zfsauton/scratch/wpotosna/neuralforecast_mica/exp_results}
# Lightning TensorBoard logs (experiment_models.py reads LIGHTNING_DIR). Per array
# task, because TensorBoardLogger picks its version_N by scanning the directory --
# concurrent tasks sharing one root would race and interleave event files.
LIGHTNING_ROOT=${LIGHTNING_ROOT:-/zfsauton/scratch/wpotosna/neuralforecast_mica/lightning_runs}
export GIFT_EVAL="${GIFT_EVAL_DIR}"
# Keep ray's session dir, HF/matplotlib caches and any scratch files off $HOME
# (491G, 78% full) -- and off the node-local /tmp, which isn't shared and gets
# reaped. Normally inherited from ~/.bashrc via sbatch --export=ALL; pinned here
# so a submit shell that never sourced it still can't write into home or /tmp.
export TMPDIR=${TMPDIR:-/zfsauton/scratch/wpotosna/tmp}
export RAY_TMPDIR="${TMPDIR}"
# Ray tune trial results. Without this Ray writes to a hardcoded ~/ray_results
# on the home filesystem; honored via the storage_path patch in
# neuralforecast/common/_base_auto.py.
export NEURALFORECAST_RAY_STORAGE_PATH=${NEURALFORECAST_RAY_STORAGE_PATH:-/zfsauton/scratch/wpotosna/neuralforecast_mica/ray_results}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/zfsauton/scratch/wpotosna/neuralforecast_mica/cache}
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}"

REPO_DIR=${SLURM_SUBMIT_DIR:?submit from mica/training/}
cd "${REPO_DIR}"
test -f train_models.py

EXPERIMENT_NAME=infini_mlpquerymixer_t5tiny
DATASETS=(electricity/D electricity/H electricity/W)
SEEDS=(1 2 3 4 5)

n_seeds=${#SEEDS[@]}
dataset_idx=$(( SLURM_ARRAY_TASK_ID / n_seeds ))
seed_idx=$(( SLURM_ARRAY_TASK_ID % n_seeds ))
DATASET_NAME=${DATASETS[$dataset_idx]}
SEED=${SEEDS[$seed_idx]}

export LIGHTNING_DIR="${LIGHTNING_ROOT}/${SLURM_JOB_NAME:-local}_${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p logs "${LIGHTNING_DIR}"

echo "=========================================="
echo "Job ID      : ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
echo "Node        : ${SLURMD_NODENAME:-$(hostname)}"
echo "Experiment  : $EXPERIMENT_NAME"
echo "Dataset     : $DATASET_NAME"
echo "Seed        : $SEED"
echo "Start       : $(date)"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"

srun "$PYTHON" -m train_models \
    --dataset_name "${DATASET_NAME}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --save_dir "${SAVE_DIR}" \
    --random_seed "${SEED}" \
    --num_samples 1 \
    --input_size_h_multiplier 2

echo "=========================================="
echo "Done : $(date)"
echo "=========================================="
