#!/bin/bash
# Keeps the Slurm queue topped up with gate-sweep tasks until all 945 combos
# are done.
#
# Why this exists: qos_general caps you at MaxSubmitJobsPU=50, and Slurm counts
# every array task individually -- so `sbatch --array=0-389` is rejected
# outright. The concurrency cap is 8 GPUs anyway, so holding ~45 tasks in the
# queue is already ~6x oversubscribed and loses no throughput.
#
# Safe to stop (Ctrl-C) and re-run at any time: pending_tasks.py re-derives what
# is outstanding from results on disk plus what is currently in the queue, so
# nothing is submitted twice and finished combos are never redone.
#
# Usage -- MUST run on a login node (rhea/lov4); every partition has
# AllocNodes=rhea,lov4, so sbatch fails from a compute node.
#
#   ssh rhea
#   cd /zfsauton2/home/wpotosna/neuralforecast/mica/training
#   tmux new -s mica            # so it survives logout
#   bash slurm/autosubmit.sh
#   # detach with Ctrl-B then D ; reattach later with: tmux attach -t mica
#
# Tunables:
#   TARGET_QUEUED=45   how many tasks to keep queued (must stay under the 50 cap)
#   POLL_SECONDS=300   how often to top up
#   DRY_RUN=1          print the sbatch commands instead of running them

set -uo pipefail

SCRATCH=/zfsauton/scratch/wpotosna/neuralforecast_mica
export SAVE_DIR="${SAVE_DIR:-$SCRATCH/exp_results}"
export GIFT_EVAL_DIR="${GIFT_EVAL_DIR:-/zfsauton/scratch/wpotosna/GiftEval}"
export LIGHTNING_ROOT="${LIGHTNING_ROOT:-$SCRATCH/lightning_runs}"
export NEURALFORECAST_RAY_STORAGE_PATH="${NEURALFORECAST_RAY_STORAGE_PATH:-$SCRATCH/ray_results}"
LOG_DIR="${LOG_DIR:-$SCRATCH/logs}"

TARGET_QUEUED="${TARGET_QUEUED:-45}"
POLL_SECONDS="${POLL_SECONDS:-300}"
QOS="${QOS:-qos_general}"
DRY_RUN="${DRY_RUN:-0}"

test -f train_models.py || { echo "ERROR: run from mica/training/"; exit 1; }
mkdir -p "$SAVE_DIR" "$LOG_DIR" "$LIGHTNING_ROOT" "$NEURALFORECAST_RAY_STORAGE_PATH" logs

# n tasks currently submitted under the capped QOS (array tasks counted individually)
queued_count() {
    squeue -u "$USER" -h -r -q "$QOS" 2>/dev/null | wc -l
}

echo "autosubmit: keeping ~${TARGET_QUEUED} tasks queued on ${QOS}, polling every ${POLL_SECONDS}s"
echo "            results -> $SAVE_DIR"
echo "            logs    -> $LOG_DIR"
echo

while true; do
    mapfile -t PENDING < <(python3 slurm/pending_tasks.py)
    if [[ ${#PENDING[@]} -eq 0 ]]; then
        if [[ $(queued_count) -eq 0 ]]; then
            echo "[$(date '+%F %T')] all combos complete, and queue is empty. Done."
            exit 0
        fi
        echo "[$(date '+%F %T')] nothing left to submit; waiting on $(queued_count) in-flight task(s)"
        sleep "$POLL_SECONDS"; continue
    fi

    have=$(queued_count)
    room=$(( TARGET_QUEUED - have ))
    if [[ $room -le 0 ]]; then
        echo "[$(date '+%F %T')] queue at ${have}/${TARGET_QUEUED}; ${#PENDING[@]} still outstanding"
        sleep "$POLL_SECONDS"; continue
    fi

    # take the next `room` pending tasks, grouped per script into one --array list
    declare -A CHUNK=()
    taken=0
    for row in "${PENDING[@]}"; do
        [[ $taken -ge $room ]] && break
        stem=${row%% *}; tid=${row##* }
        CHUNK[$stem]="${CHUNK[$stem]:+${CHUNK[$stem]},}${tid}"
        taken=$(( taken + 1 ))
    done

    for stem in "${!CHUNK[@]}"; do
        ids="${CHUNK[$stem]}"
        n=$(awk -F, '{print NF}' <<<"$ids")
        echo "[$(date '+%F %T')] submitting ${stem}: ${n} task(s) [${ids:0:70}$([[ ${#ids} -gt 70 ]] && echo ...)]"
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "    DRY: sbatch --array=${ids} --output=${LOG_DIR}/%x_%A_%a.out --error=${LOG_DIR}/%x_%A_%a.err slurm/${stem}.sbatch"
        else
            sbatch --array="${ids}" \
                   --output="${LOG_DIR}/%x_%A_%a.out" \
                   --error="${LOG_DIR}/%x_%A_%a.err" \
                   "slurm/${stem}.sbatch" || echo "    submit failed for ${stem} (will retry next poll)"
        fi
    done
    unset CHUNK
    sleep "$POLL_SECONDS"
done
