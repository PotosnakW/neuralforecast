#!/bin/bash
# Second round of runs the paper tables still need, on the H200s.
#
# 1) Longer-context ablation (input_size_h_multiplier=3), MICA ciincl arm only
#    (MICA_CIINCL_ONLY), on the 6 original datasets neither the March runs nor
#    run_table_fills.sbatch covered:
#      simglucose, iowa_ihop_smex_windspeed, iowa_plows_windspeed, M_DENSE/H,
#      ett1/H, ett2/H  x  {infini_mlpquerymixer_t5tiny, vanilla_t5tiny}
#      x 5 seeds                                                  60 tasks
# 2) Weighted-channels ablation on electricity/{H,D,W} (the only datasets it
#    was missing), windows_batch_size=8 like every other electricity run:
#      infini_mlpquerymixer_t5tiny_{static,dynamic}_weights x 5 seeds  30 tasks
#
# Ordered with the runs of unknown length first (5-min datasets at 3x context).
#
# Results go to their own scratch dir, then get synced into the shared
# icml_exp_results: ishm3 runs -> infini_mlpquerymixer_t5tiny_ishm3 /
# vanilla_t5tiny_ishm3; weights runs keep their experiment name.
#
# Usage -- MUST run on a login node (rhea/lov4), in tmux:
#   cd /zfsauton2/home/wpotosna/neuralforecast/mica/training
#   SBATCH_FILE=slurm/run_table_fills2.sbatch \
#     STATE=/zfsauton/scratch/wpotosna/neuralforecast_mica/logs/table_fills2_next_id \
#     bash slurm/submit_table_fills.sh
#   sbatch --array=5,9 slurm/run_table_fills2.sbatch     # rerun some tasks
#
#SBATCH --job-name=mica_table_fills2
#SBATCH --output=/zfsauton/scratch/wpotosna/neuralforecast_mica/logs/%x_%A_%a.out
#SBATCH --error=/zfsauton/scratch/wpotosna/neuralforecast_mica/logs/%x_%A_%a.err
#SBATCH --partition=project
#SBATCH --qos=qos_radio_freq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --array=0-89

set -euo pipefail

TASKS=()
add() {  # add EXPERIMENT DATASET ISHM SEED...
    local exp=$1 ds=$2 ishm=$3; shift 3
    for s in "$@"; do TASKS+=("$exp $ds $ishm $s"); done
}
for ds in simglucose iowa_ihop_smex_windspeed iowa_plows_windspeed M_DENSE/H ett1/H ett2/H; do
    add infini_mlpquerymixer_t5tiny "$ds" 3 1 2 3 4 5
done
for ds in electricity/H electricity/D electricity/W; do
    for exp in infini_mlpquerymixer_t5tiny_static_weights infini_mlpquerymixer_t5tiny_dynamic_weights; do
        add "$exp" "$ds" 2 1 2 3 4 5
    done
done
for ds in simglucose iowa_ihop_smex_windspeed iowa_plows_windspeed M_DENSE/H ett1/H ett2/H; do
    add vanilla_t5tiny "$ds" 3 1 2 3 4 5
done
# keep #SBATCH --array=0-(N-1) in sync with this list
[[ ${#TASKS[@]} -eq 90 ]] || { echo "TASKS has ${#TASKS[@]} entries; update --array"; exit 1; }


read -r EXPERIMENT_NAME DATASET_NAME ISHM SEED <<<"${TASKS[$SLURM_ARRAY_TASK_ID]}"
if [[ "$ISHM" == 3 ]]; then export MICA_CIINCL_ONLY=1; fi

CONDA_BASE="/zfsauton2/home/wpotosna/miniconda3"
PYTHON="${CONDA_BASE}/envs/neuralforecast/bin/python"
SCRATCH=/zfsauton/scratch/wpotosna/neuralforecast_mica
SAVE_DIR=${SAVE_DIR:-${SCRATCH}/exp_results_table_fills2}
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
