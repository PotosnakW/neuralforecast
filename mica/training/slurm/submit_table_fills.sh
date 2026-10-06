#!/bin/bash
# Feeds run_table_fills.sbatch's 194 tasks to the H200s in order.
#
# qos_radio_freq caps you at MaxSubmitJobsPU=50 (array tasks count
# individually), so `sbatch --array=0-193` is rejected outright. This keeps
# ~TARGET tasks queued+running and submits the next ids as slots free up.
#
# Restart-safe: the next id to submit is kept in STATE, so Ctrl-C and re-run
# picks up where it left off. Delete STATE to start over.
#
# Usage -- MUST run on a login node (rhea/lov4), in tmux so it survives logout:
#   ssh rhea
#   cd /zfsauton2/home/wpotosna/neuralforecast/mica/training
#   tmux new -s mica_fills
#   bash slurm/submit_table_fills.sh
#
# Tunables: TARGET=48  POLL_SECONDS=300  DRY_RUN=1  SBATCH_FILE=...  STATE=...

set -uo pipefail
test -f train_models.py || { echo "ERROR: run from mica/training/"; exit 1; }

SBATCH_FILE="${SBATCH_FILE:-slurm/run_table_fills.sbatch}"
LAST=$(grep -oP '^#SBATCH --array=0-\K\d+' "$SBATCH_FILE")
TARGET="${TARGET:-48}"
POLL_SECONDS="${POLL_SECONDS:-300}"
DRY_RUN="${DRY_RUN:-0}"
QOS=qos_radio_freq
STATE="${STATE:-/zfsauton/scratch/wpotosna/neuralforecast_mica/logs/table_fills_next_id}"

next=$(cat "$STATE" 2>/dev/null || echo 0)
echo "submitting tasks ${next}-${LAST} of ${SBATCH_FILE}, keeping ~${TARGET} on ${QOS}"

while [[ $next -le $LAST ]]; do
    have=$(squeue -u "$USER" -h -r -q "$QOS" 2>/dev/null | wc -l)
    room=$(( TARGET - have ))
    if [[ $room -gt 0 ]]; then
        end=$(( next + room - 1 ))
        [[ $end -gt $LAST ]] && end=$LAST
        echo "[$(date '+%F %T')] queue ${have}/${TARGET}; submitting ${next}-${end}"
        if [[ "$DRY_RUN" == 1 ]]; then
            echo "    DRY: sbatch --array=${next}-${end} ${SBATCH_FILE}"
            next=$(( end + 1 ))
        elif sbatch --array="${next}-${end}" "$SBATCH_FILE"; then
            next=$(( end + 1 ))
            echo "$next" > "$STATE"
        else
            echo "    submit failed (will retry next poll)"
        fi
    fi
    [[ $next -le $LAST ]] && sleep "$POLL_SECONDS"
done
echo "[$(date '+%F %T')] all tasks 0-${LAST} submitted."
