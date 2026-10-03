#!/bin/bash
# One array task per experiment in build_results.py (`python build_results.py --list`).
# CPU only; electricity_H forecasts are ~1.6 GB per seed, hence the memory.
#
# Usage:
#   cd mica/tables && sbatch build_results.sbatch
#   sbatch --array=5 build_results.sbatch          # rebuild one experiment
#
#SBATCH --job-name=mica_results
#SBATCH --output=/zfsauton/scratch/wpotosna/neuralforecast_mica/logs/%x_%A_%a.out
#SBATCH --error=/zfsauton/scratch/wpotosna/neuralforecast_mica/logs/%x_%A_%a.err
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-25

set -euo pipefail
PYTHON=/zfsauton2/home/wpotosna/miniconda3/envs/neuralforecast/bin/python
export GIFT_EVAL=${GIFT_EVAL_DIR:-/zfsauton/scratch/wpotosna/GiftEval}
export TMPDIR=${TMPDIR:-/zfsauton/scratch/wpotosna/tmp}
export PYTHONPATH="/zfsauton2/home/wpotosna/neuralforecast:/zfsauton2/home/wpotosna/gift-eval/src${PYTHONPATH:+:$PYTHONPATH}"

cd "${SLURM_SUBMIT_DIR}"
"$PYTHON" build_results.py --experiment "${SLURM_ARRAY_TASK_ID}"
