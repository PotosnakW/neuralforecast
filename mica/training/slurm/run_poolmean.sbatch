#!/bin/bash
# Ask 2: new mean-pool memory variants (A_global = (1/C) * sum_c V^(c),
# independent of the query -- the reviewer-requested pool_mean baseline in
# experiment_models.py) across every dataset in DATASET_NAMES, both gates:
#   - infini_poolmean_mlpquerymixer_t5tiny  (MLP-query gate)
#   - infini_poolmean_t5tiny                (betas gate)
# Each array task trains one (experiment, dataset, seed) combo on its own
# GPU, so the scheduler packs/parallelizes across every GPU you have access
# to instead of serializing everything in one long-running script.
#
# Usage:
#   cd mica/training && mkdir -p logs
#   sbatch slurm/run_poolmean.sbatch
# To cap concurrent tasks at your node's GPU count:
#   sbatch --array=0-209%<N_GPUS> slurm/run_poolmean.sbatch
#
#SBATCH --job-name=mica_poolmean
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-209

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

# Mirrors DATASET_NAMES in launch_experiments.py -- keep in sync if that list changes.
EXPERIMENTS=(infini_poolmean_mlpquerymixer_t5tiny infini_poolmean_t5tiny)
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
