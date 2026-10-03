#!/bin/bash
# Submit tasks at risk of OOM on a 48GB A6000 to the 141GB H200s (project
# partition, gpu2). Same job name as the regular array, so pending_tasks.py sees
# them as in flight and autosubmit.sh never double-submits them; they also don't
# count against autosubmit's qos_general slots.
#
# At risk = mixer gates (mlp / mlp-query) on the windows_batch_size=64 datasets
# with >250 series: LOOP_SEATTLE/D (323) and covid_deaths (266). The mlpmixer
# runs on LOOP_SEATTLE OOMed on A6000 when trial 1 of the Optuna search
# (mlpmixer_hidden_size=512) filled all 47GB; covid_deaths mlpmixer sits at
# 32-42GB. Electricity has 321+ series but already runs windows_batch_size=8.
#
# Submitted so far:
#   run_new_datasets_full  160-164,200-204   LOOP_SEATTLE mlpmixer, mlpquerymixer
#   run_new_datasets_full  205-209           covid_deaths mlpquerymixer
#   run_poolmean           15-19,30-34       LOOP_SEATTLE, covid_deaths poolmean_mlpquerymixer
#   run_new_datasets_full  165-169           covid_deaths mlpmixer (hit 24h on A6000 at 8-9/20 trials)
#
# Usage -- MUST run on a login node (rhea/lov4). IDS is required, so a bare
# re-run can't double-submit tasks that are already on the H200s:
#   cd /zfsauton2/home/wpotosna/neuralforecast/mica/training
#   SCRIPT=run_poolmean IDS=15-19,30-34 bash slurm/submit_h200.sh

set -euo pipefail
test -f train_models.py || { echo "ERROR: run from mica/training/"; exit 1; }

SCRIPT="${SCRIPT:-run_new_datasets_full}"
IDS="${IDS:?set IDS, e.g. IDS=205-209}"
LOG_DIR="${LOG_DIR:-/zfsauton/scratch/wpotosna/neuralforecast_mica/logs}"

sbatch --array="${IDS}" \
       --partition=project --qos=qos_radio_freq --gres=gpu:h200:1 \
       --time=48:00:00 \
       --output="${LOG_DIR}/%x_%A_%a.out" \
       --error="${LOG_DIR}/%x_%A_%a.err" \
       "slurm/${SCRIPT}.sbatch"
