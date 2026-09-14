#!/bin/bash
# Preflight for the mica sbatch arrays. Run from mica/training on the SUBMIT NODE:
#   cd mica/training && bash slurm/preflight.sh
# Exits non-zero if anything that would break all array tasks is wrong.
#
# CONDA_BASE/CONDA_ENV are read out of run_poolmean.sbatch so this can't drift
# from what the jobs actually use. Override GIFT_EVAL_DIR/SAVE_DIR the same way
# you would when submitting.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE=$(sed -n 's/^CONDA_BASE="\(.*\)"$/\1/p' "$HERE/run_poolmean.sbatch")
CONDA_ENV=$(sed -n 's/^CONDA_ENV="\(.*\)"$/\1/p' "$HERE/run_poolmean.sbatch")
PYTHON="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"
PIP="${CONDA_BASE}/envs/${CONDA_ENV}/bin/pip"
GIFT_EVAL_DIR=${GIFT_EVAL_DIR:-../GIFT_EVAL_DIR}
SAVE_DIR=${SAVE_DIR:-../exp_results}
REPO="$(cd "$HERE/../../.." && pwd)"

fail=0; warned=0
ck(){ if eval "$2" >/dev/null 2>&1; then printf "  \033[32mok  \033[0m %s\n" "$1"
      else printf "  \033[31mFAIL\033[0m %s\n" "$1"; fail=1; fi; }
warn(){ if eval "$2" >/dev/null 2>&1; then printf "  \033[32mok  \033[0m %s\n" "$1"
      else printf "  \033[33mwarn\033[0m %s\n" "$1"; warned=1; fi; }

echo "== paths (conda from run_poolmean.sbatch: $CONDA_BASE, env: $CONDA_ENV) =="
ck   "CONDA_BASE exists"                        "[ -d '$CONDA_BASE' ]"
ck   "interpreter exists"                       "[ -x '$PYTHON' ]"
ck   "cwd is mica/training"                     "[ -f train_models.py ]"
ck   "logs/ exists (slurm opens it pre-script)" "[ -d logs ]"
ck   "GIFT_EVAL_DIR exists: $GIFT_EVAL_DIR"     "[ -d '$GIFT_EVAL_DIR' ]"
ck   "SAVE_DIR exists: $SAVE_DIR"               "[ -d '$SAVE_DIR' ]"
ck   "local csv datasets present"               "[ -f ../datasets/simglucose_90_days.csv ]"

echo "== code version (repo: $REPO) =="
echo "       HEAD: $(git -C "$REPO" log --oneline -1 2>/dev/null)"
ck   "working tree has pool_mean in _infini.py"    "grep -q pool_mean '$REPO/neuralforecast/common/_infini.py'"
ck   "working tree has pool_mean in _t5_infini.py" "grep -q pool_mean '$REPO/neuralforecast/common/_t5_infini.py'"
warn "no uncommitted changes to tracked files"     "[ -z \"\$(git -C '$REPO' status --porcelain --untracked-files=no)\" ]"
warn "HEAD is pushed to a remote"                  "[ -n \"\$(git -C '$REPO' branch -r --contains HEAD 2>/dev/null)\" ]"

echo "== environment =="
ck "import train_models"                   "'$PYTHON' -c 'import train_models'"
# The silent-failure check: a non-editable site-packages copy would train OLD code
# and still exit 0 on every task.
ck "neuralforecast resolves into this repo" \
   "'$PYTHON' -c \"import neuralforecast,os,sys; sys.exit(0 if os.path.realpath(neuralforecast.__file__).startswith(os.path.realpath('$REPO')) else 1)\""
for m in _infini _t5_infini; do
  ck "imported $m.py has pool_mean" \
     "grep -q pool_mean \"\$('$PYTHON' -c 'import neuralforecast.common.$m as m; print(m.__file__)' 2>/dev/null)\""
done
ck   "torch sees a CUDA device"  "'$PYTHON' -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'"
warn "pip check clean"           "'$PIP' check"

echo "== slurm =="
ck   "sbatch on PATH"              "command -v sbatch"
ck   "partition 'general' exists"  "sinfo -h -p general -o %P | grep -q general"
warn "qos 'qos_general' exists"    "sacctmgr -nP show qos format=name | grep -qx qos_general"
warn "gpu gres offered on general" "sinfo -h -p general -o %G | grep -q gpu"

echo
if [ $fail -ne 0 ]; then echo "PREFLIGHT FAILED -- fix the above before sbatch"; exit 1; fi
[ $warned -ne 0 ] && echo "PREFLIGHT PASSED (with warnings)" || echo "PREFLIGHT PASSED -- safe to sbatch"
exit 0
