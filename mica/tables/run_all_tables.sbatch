#!/bin/bash
# Computational study tables on a real GPU.
#
# Do NOT run run_all_tables.sh on rhea/lov4: those are login nodes with no GPU,
# so torch falls back to CPU and the sweep crawls. C=600 and L=8192 also need
# far more than the 11GB on the older cards, hence the a6000 constraint.
#
# Usage:
#   cd mica/tables && sbatch run_all_tables.sbatch
#
#SBATCH --job-name=mica_tables
#SBATCH --output=/zfsauton/scratch/wpotosna/neuralforecast_mica/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/wpotosna/neuralforecast_mica/logs/%x_%j.err
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a6000:1

set -uo pipefail

export TMPDIR=${TMPDIR:-/zfsauton/scratch/wpotosna/tmp}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/zfsauton/scratch/wpotosna/neuralforecast_mica/cache}
export HF_HOME=${HF_HOME:-${XDG_CACHE_HOME}/huggingface}
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}"

cd "${SLURM_SUBMIT_DIR:?submit from mica/tables/}"
test -f run_all_tables.sh

echo "Node: ${SLURMD_NODENAME:-$(hostname)}  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

bash run_all_tables.sh

echo "Done: $(date)"
