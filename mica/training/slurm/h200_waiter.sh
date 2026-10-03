#!/bin/bash
# Moves OOM-prone poolmean tasks from general onto the H200s as qos_radio_freq
# submit slots free up. qos_radio_freq caps queued+running jobs per user, so
# sbatch is simply rejected while you're at the cap -- this retries until it
# isn't. Each task's general copy stays queued until its H200 copy is accepted,
# so pending_tasks.py always sees it in flight and autosubmit never resubmits it.
# A general copy that starts running before an H200 slot frees is left alone.
# No duplicates: the H200 copy is submitted held and only released once the
# general copy is confirmed gone; if it isn't, the held copy is cancelled.
#
# Usage -- MUST run on a login node (rhea/lov4), e.g. in its own tmux window:
#   cd /zfsauton2/home/wpotosna/neuralforecast/mica/training
#   GEN_JOB=69901 TASKS="80 82 83 95 96 97 98" bash slurm/h200_waiter.sh

set -uo pipefail
test -f train_models.py || { echo "ERROR: run from mica/training/"; exit 1; }

SCRIPT="${SCRIPT:-run_poolmean}"
GEN_JOB="${GEN_JOB:?set GEN_JOB, the general array job id holding the tasks}"
read -ra TASKS <<<"${TASKS:?set TASKS, e.g. TASKS=\"80 82 83\"}"
POLL_SECONDS="${POLL_SECONDS:-600}"
LOG_DIR="${LOG_DIR:-/zfsauton/scratch/wpotosna/neuralforecast_mica/logs}"

# Keep the general copies off the 24GB A5000 node while they wait.
for t in "${TASKS[@]}"; do
    scontrol update JobId="${GEN_JOB}_${t}" ExcNodeList=gpu28 2>/dev/null \
        || echo "note: couldn't exclude gpu28 for ${GEN_JOB}_${t}"
done

while [[ ${#TASKS[@]} -gt 0 ]]; do
    left=()
    for t in "${TASKS[@]}"; do
        state=$(squeue -h -r -j "${GEN_JOB}_${t}" -o %T 2>/dev/null)
        if [[ "$state" != "PENDING" ]]; then
            echo "[$(date '+%F %T')] task ${t}: general copy is '${state:-gone}', dropping"
            continue
        fi
        # Submitted held, so it can't start until the general copy is confirmed gone.
        hid=$(sbatch --parsable --hold --array="${t}" \
                     --partition=project --qos=qos_radio_freq --gres=gpu:h200:1 \
                     --time=48:00:00 \
                     --output="${LOG_DIR}/%x_%A_%a.out" \
                     --error="${LOG_DIR}/%x_%A_%a.err" \
                     "slurm/${SCRIPT}.sbatch" 2>/dev/null) || { left+=("$t"); continue; }
        hid=${hid%%;*}
        # --state=PENDING: never kills a general copy that just started running.
        scancel --state=PENDING "${GEN_JOB}_${t}"
        gen_alive=""
        for _ in 1 2 3 4 5 6; do
            gen_alive=$(squeue -h -r -j "${GEN_JOB}_${t}" -t PENDING,RUNNING -o %T 2>/dev/null)
            [[ -z "$gen_alive" ]] && break
            sleep 5
        done
        if [[ -n "$gen_alive" ]]; then
            scancel "$hid"
            echo "[$(date '+%F %T')] task ${t}: general copy still ${gen_alive}; cancelled held H200 job ${hid}, dropping"
            continue
        fi
        if scontrol release "$hid"; then
            echo "[$(date '+%F %T')] task ${t}: on H200 as ${hid}, cancelled ${GEN_JOB}_${t}"
        else
            echo "[$(date '+%F %T')] task ${t}: WARNING general copy cancelled but H200 job ${hid} still held -- run: scontrol release ${hid}"
        fi
    done
    TASKS=("${left[@]}")
    [[ ${#TASKS[@]} -gt 0 ]] && sleep "$POLL_SECONDS"
done
echo "[$(date '+%F %T')] all tasks moved or dropped. Done."
