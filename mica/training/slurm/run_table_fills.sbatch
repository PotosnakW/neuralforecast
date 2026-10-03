#!/bin/bash
# Runs the paper tables still need, on the H200s.
#
# 1) Electricity baselines, all at windows_batch_size=8 (experiment_models.py's
#    default for electricity) so every model shares one batch size on
#    electricity -- the January baselines had run it at 64 and the September
#    MICA/poolmean runs at 8. Full H/D/W x 5 seeds for each trained baseline
#    (replaces the January electricity runs), plus Chronos-2 (zero-shot, seed 1):
#      vanilla_t5tiny, itransformer_baseline, crossformer_baseline,
#      timerxl_baseline, timemixer_baseline, multivariateMLP_baseline,
#      tsmixer_baseline                                       105 tasks
#      chronos2.0_baseline                                      3 tasks
# 2) Longer-context ablation (input_size_h_multiplier=3), MICA ciincl arm only
#    (MICA_CIINCL_ONLY; the table reads *_ciincl and the March runs had only it):
#      infini_mlpquerymixer_t5tiny: missing seeds on the original six datasets,
#        plus all 5 seeds on solar/{D,H,W}, LOOP_SEATTLE/D, ett2/{D,W},
#        electricity/{H,D,W}                                   51 tasks
#      vanilla_t5tiny: all 5 seeds on solar/H, LOOP_SEATTLE/D, ett2/{D,W},
#        electricity/{H,D,W}                                   35 tasks
#    Listed after the main-table runs (tasks 0-107) so those finish first;
#    within the ablation, longest first -- the hourly mixer runs
#    (electricity/H, solar/H) are the ones at risk of the 48h limit.
#
# Results go to their own scratch dir, then get synced into the shared
# icml_exp_results: chronos2.0_baseline -> chronos2.0_baseline_wpotosna;
# ishm3 runs of infini_mlpquerymixer_t5tiny / vanilla_t5tiny ->
# infini_mlpquerymixer_t5tiny_ishm3 / vanilla_t5tiny_ishm3.
#
# Usage -- MUST run on a login node (rhea/lov4):
#   cd /zfsauton2/home/wpotosna/neuralforecast/mica/training
#   bash slurm/submit_table_fills.sh   # in tmux; qos_radio_freq caps you at 50
#                                      # queued jobs, so a plain sbatch is rejected
#   sbatch --array=5,9 slurm/run_table_fills.sbatch      # rerun some tasks
#
#SBATCH --job-name=mica_table_fills
#SBATCH --output=/zfsauton/scratch/wpotosna/neuralforecast_mica/logs/%x_%A_%a.out
#SBATCH --error=/zfsauton/scratch/wpotosna/neuralforecast_mica/logs/%x_%A_%a.err
#SBATCH --partition=project
#SBATCH --qos=qos_radio_freq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --array=0-193

set -euo pipefail

TASKS=()
add() {  # add EXPERIMENT DATASET ISHM SEED...
    local exp=$1 ds=$2 ishm=$3; shift 3
    for s in "$@"; do TASKS+=("$exp $ds $ishm $s"); done
}
for ds in electricity/H electricity/D electricity/W; do
    add chronos2.0_baseline "$ds" 2 1
done
for exp in vanilla_t5tiny itransformer_baseline crossformer_baseline timerxl_baseline \
           timemixer_baseline multivariateMLP_baseline tsmixer_baseline; do
    for ds in electricity/H electricity/D electricity/W; do
        add "$exp" "$ds" 2 1 2 3 4 5
    done
done
MQ=infini_mlpquerymixer_t5tiny
for ds in electricity/H solar/H electricity/D LOOP_SEATTLE/D solar/D \
          electricity/W solar/W ett2/D ett2/W; do
    add $MQ "$ds" 3 1 2 3 4 5
done
add $MQ covid_deaths   3 5
add $MQ jena_weather/H 3 1 5
add $MQ jena_weather/D 3 2
add $MQ ett1/D         3 4
add $MQ ett1/W         3 3
for ds in electricity/H solar/H electricity/D LOOP_SEATTLE/D electricity/W ett2/D ett2/W; do
    add vanilla_t5tiny "$ds" 3 1 2 3 4 5
done
# keep #SBATCH --array=0-(N-1) in sync with this list
[[ ${#TASKS[@]} -eq 194 ]] || { echo "TASKS has ${#TASKS[@]} entries; update --array"; exit 1; }

read -r EXPERIMENT_NAME DATASET_NAME ISHM SEED <<<"${TASKS[$SLURM_ARRAY_TASK_ID]}"
if [[ "$ISHM" == 3 ]]; then export MICA_CIINCL_ONLY=1; fi

CONDA_BASE="/zfsauton2/home/wpotosna/miniconda3"
PYTHON="${CONDA_BASE}/envs/neuralforecast/bin/python"
SCRATCH=/zfsauton/scratch/wpotosna/neuralforecast_mica
SAVE_DIR=${SAVE_DIR:-${SCRATCH}/exp_results_table_fills}
LIGHTNING_ROOT=${LIGHTNING_ROOT:-${SCRATCH}/lightning_runs}
export GIFT_EVAL=${GIFT_EVAL_DIR:-/zfsauton/scratch/wpotosna/GiftEval}
export TMPDIR=${TMPDIR:-/zfsauton/scratch/wpotosna/tmp}
export RAY_TMPDIR="${TMPDIR}"
export NEURALFORECAST_RAY_STORAGE_PATH=${NEURALFORECAST_RAY_STORAGE_PATH:-${SCRATCH}/ray_results}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${SCRATCH}/cache}
export HF_HOME=${HF_HOME:-${XDG_CACHE_HOME}/huggingface}
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${SAVE_DIR}"

REPO_DIR=${SLURM_SUBMIT_DIR:?submit from mica/training/}
cd "${REPO_DIR}"
test -f train_models.py

export LIGHTNING_DIR="${LIGHTNING_ROOT}/${SLURM_JOB_NAME:-local}_${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${LIGHTNING_DIR}"

echo "=========================================="
echo "Job ID      : ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
echo "Node        : ${SLURMD_NODENAME:-$(hostname)}"
echo "Experiment  : $EXPERIMENT_NAME"
echo "Dataset     : $DATASET_NAME"
echo "Seed        : $SEED"
echo "ISHM        : $ISHM"
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
    --input_size_h_multiplier "${ISHM}"

echo "=========================================="
echo "Done : $(date)"
echo "=========================================="
