#!/bin/bash
# Ask 3: rerun infini_channel_exclusion=True (ciexcl) after the leave-one-out
# memory-matrix aliasing fix in neuralforecast/common/_infini.py and
# _t5_infini.py (in-place `-=` against an expand()'d, stride-0 tensor was
# writing per-channel-different values into one aliased location -- silently
# wrong on old PyTorch, a RuntimeError on newer PyTorch). Only ciexcl was
# affected, so this uses the *_ciexcl-only experiment_name variants added to
# experiment_models.py, which train just the ciexcl arm instead of retraining
# the already-correct ciincl arm too.
#
# All 6 gate variants (shared beta, channelwise beta, layerwise beta,
# layerwise+channelwise beta, mlp, mlpquery), over the subset of datasets
# these were previously run on: simglucose, both iowa windspeed sets,
# M_DENSE, jena_weather, ett1, ett2. Each array task trains one
# (gate, dataset, seed) combo on its own GPU, so the scheduler
# packs/parallelizes across every GPU you have access to instead of
# serializing everything in one long-running script.
#
# Usage:
#   cd mica/training && mkdir -p logs
#   sbatch slurm/run_ciexcl_rerun.sbatch
# To cap concurrent tasks at your node's GPU count:
#   sbatch --array=0-389%<N_GPUS> slurm/run_ciexcl_rerun.sbatch
#
#SBATCH --job-name=mica_ciexcl_rerun
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-389

set -euo pipefail

CONDA_BASE="/zfsauton2/home/wpotosna/miniconda3"
CONDA_ENV="neuralforecast"
# Absolute path, not `conda run -n neuralforecast python` -- conda run resolves
# an unqualified `python` via $PATH, which silently picks up another env's
# interpreter if one is hardcoded ahead of it there.
PYTHON="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"
GIFT_EVAL_DIR=${GIFT_EVAL_DIR:-../GIFT_EVAL_DIR}
SAVE_DIR=${SAVE_DIR:-../exp_results}
export GIFT_EVAL="${GIFT_EVAL_DIR}"

REPO_DIR=${SLURM_SUBMIT_DIR:?submit from mica/training/}
cd "${REPO_DIR}"
test -f train_models.py

EXPERIMENTS=(
    infini_t5tiny_ciexcl
    infini_channelwise_t5tiny_ciexcl
    infini_layerwise_t5tiny_ciexcl
    infini_layerwise_channelwise_t5tiny_ciexcl
    infini_mlpmixer_t5tiny_ciexcl
    infini_mlpquerymixer_t5tiny_ciexcl
)
DATASETS=(
    simglucose
    iowa_ihop_smex_windspeed
    iowa_plows_windspeed
    M_DENSE/D
    M_DENSE/H
    jena_weather/D
    jena_weather/H
    ett1/D
    ett1/H
    ett1/W
    ett2/D
    ett2/H
    ett2/W
)
SEEDS=(1 2 3 4 5)

n_datasets=${#DATASETS[@]}
n_seeds=${#SEEDS[@]}
per_experiment=$(( n_datasets * n_seeds ))

exp_idx=$(( SLURM_ARRAY_TASK_ID / per_experiment ))
rem=$(( SLURM_ARRAY_TASK_ID % per_experiment ))
dataset_idx=$(( rem / n_seeds ))
seed_idx=$(( rem % n_seeds ))

EXPERIMENT_NAME=${EXPERIMENTS[$exp_idx]}
DATASET_NAME=${DATASETS[$dataset_idx]}
SEED=${SEEDS[$seed_idx]}

mkdir -p logs

echo "=========================================="
echo "Job ID      : ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
echo "Node        : ${SLURMD_NODENAME:-$(hostname)}"
echo "Experiment  : $EXPERIMENT_NAME"
echo "Dataset     : $DATASET_NAME"
echo "Seed        : $SEED"
echo "Start       : $(date)"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

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
