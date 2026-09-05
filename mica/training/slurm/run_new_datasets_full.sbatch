#!/bin/bash
# All 6 gate variants (shared beta, channelwise beta, layerwise beta,
# layerwise+channelwise beta, mlp, mlpquery), trained fresh -- both
# infini_channel_exclusion=True (ciexcl) and =False (ciincl) arms -- on the
# 8 dataset-variants in DATASET_NAMES that have never been run for the gate
# sweep before: LOOP_SEATTLE/D, covid_deaths, solar/{D,H,W},
# electricity/{D,H,W}. Uses the full (non-"_ciexcl") experiment names, each
# of which trains both arms in one call, so ciexcl and ciincl are covered in
# a single pass instead of double-training ciexcl once here and once via
# run_ciexcl_rerun.sbatch.
#
# NOTE: infini_mlpquerymixer_t5tiny on electricity/{D,H,W} is trained
# separately by run_ecl_mlpquery.sbatch (submitted as its own experiment),
# so that one (gate, dataset) slice is skipped below as a no-op to avoid
# training it twice.
#
# Each array task trains one (gate, dataset, seed) combo on its own GPU, so
# the scheduler packs/parallelizes across every GPU you have access to
# instead of serializing everything in one long-running script.
#
# Usage:
#   cd mica/training && mkdir -p logs
#   sbatch slurm/run_new_datasets_full.sbatch
# To cap concurrent tasks at your node's GPU count:
#   sbatch --array=0-239%<N_GPUS> slurm/run_new_datasets_full.sbatch
#
#SBATCH --job-name=mica_new_datasets_full
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-239

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
    infini_t5tiny
    infini_channelwise_t5tiny
    infini_layerwise_t5tiny
    infini_layerwise_channelwise_t5tiny
    infini_mlpmixer_t5tiny
    infini_mlpquerymixer_t5tiny
)
DATASETS=(
    LOOP_SEATTLE/D
    covid_deaths
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

# Handled separately by run_ecl_mlpquery.sbatch -- skip to avoid double-training.
if [[ "$EXPERIMENT_NAME" == "infini_mlpquerymixer_t5tiny" && "$DATASET_NAME" == electricity/* ]]; then
    echo "Skipping ${EXPERIMENT_NAME} / ${DATASET_NAME} / seed ${SEED} -- covered by run_ecl_mlpquery.sbatch"
    exit 0
fi

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
