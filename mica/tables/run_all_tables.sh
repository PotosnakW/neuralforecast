#!/bin/bash
# Produce every computational-study table: C=7, C=600, and the context-length
# ablation -- across all three conda envs, then merge into one CSV per config.
#
# Three envs are needed because no single one can build every model: envs/toto2
# ships a newer `transformers` that cannot construct MICA's T5/MOMENT models,
# and toto2 / timesfm have incompatible torch pins. Each env contributes the
# rows it can (--only), and the per-config CSVs are concatenated at the end.
#
# Usage:
#   cd mica/tables && bash run_all_tables.sh
#   CONTEXTS="96 512" bash run_all_tables.sh     # subset while iterating
#   OUT_DIR=/some/path bash run_all_tables.sh

set -uo pipefail

CONDA=/zfsauton2/home/wpotosna/miniconda3/envs
SCRIPT=computational_parameter_study_baselines.py
OUT_DIR="${OUT_DIR:-/zfsauton/scratch/wpotosna/neuralforecast_mica/table_results}"
H="${H:-48}"
WBS="${WBS:-1}"
DEVICE="${DEVICE:-cuda:0}"

export PYTHONPATH="/zfsauton2/home/wpotosna/neuralforecast:/zfsauton2/home/wpotosna/gift-eval/src${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/zfsauton/scratch/wpotosna/neuralforecast_mica/cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export TMPDIR="${TMPDIR:-/zfsauton/scratch/wpotosna/tmp}"

# context lengths for the L ablation (8*8=64 .. 8192)
CONTEXTS="${CONTEXTS:-16 32 64 96 128 256 312 512 8192}"
# n_series sweep for plots/parameter_scaling.ipynb, which reads one file per
# channel count. Set MODE=nsweep to produce those instead of the three tables.
NSERIES="${NSERIES:-7 15 20 50 100 150 200 300 400 500 600}"
MODE="${MODE:-tables}"

mkdir -p "$OUT_DIR"

# env : --only filter : tag
RUNS=(
    "neuralforecast::mica"
    "toto2:Toto-2:toto2"
    "mica_timesfm3:TimesFM-3:timesfm3"
)

run_one() {  # $1=n_series  $2=input_size
    local n=$1 L=$2
    echo "=============================================================="
    echo "  n_series=$n  input_size=$L  h=$H"
    echo "=============================================================="
    for entry in "${RUNS[@]}"; do
        IFS=: read -r env only tag <<<"$entry"
        local py="$CONDA/$env/bin/python"
        [[ -x "$py" ]] || { echo "  !! missing env $env, skipping"; continue; }
        local onlyarg=()
        [[ -n "$only" ]] && onlyarg=(--only "$only")
        echo "  -> env=$env ${only:+(only $only)}"
        "$py" "$SCRIPT" --n_series "$n" --input_size "$L" --h "$H" \
              --windows_batch_size "$WBS" --device "$DEVICE" \
              --out_dir "$OUT_DIR" --tag "$tag" "${onlyarg[@]}" \
            2>&1 | grep -E "^  \[models\]|^wrote|Error" | sed 's/^/     /'
    done
    # merge this config's per-env CSVs
    "$CONDA/neuralforecast/bin/python" - "$OUT_DIR" "$n" "$L" <<'PY'
import sys, glob, os, pandas as pd
out_dir, n, L = sys.argv[1], sys.argv[2], sys.argv[3]
parts = sorted(glob.glob(os.path.join(out_dir, f'flops_baseline_table_n{n}_is{L}_*.csv')))
if not parts:
    print("     (nothing to merge)"); raise SystemExit
df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
dest = os.path.join(out_dir, f'flops_baseline_table_n{n}_is{L}_MERGED.csv')
df.to_csv(dest, index=False)
print(f"     merged {len(parts)} files -> {dest}  ({len(df)} rows)")
PY
}

if [[ "$MODE" == "nsweep" ]]; then
    # parameter_scaling.ipynb reads flops_baseline_table_n{N}.csv (no _is suffix),
    # so after merging each config we copy the merged file to that name.
    echo "### n_series sweep for parameter_scaling.ipynb (input_size=${PS_L:-96}) ###"
    for N in $NSERIES; do
        run_one "$N" "${PS_L:-96}"
        _merged="$OUT_DIR/flops_baseline_table_n${N}_is${PS_L:-96}_MERGED.csv"
        if [[ -f "$_merged" ]]; then
            cp "$_merged" "$OUT_DIR/flops_baseline_table_n${N}.csv"
            echo "     -> $OUT_DIR/flops_baseline_table_n${N}.csv"
        fi
    done
    echo
    echo "All outputs in $OUT_DIR"
    exit 0
fi

echo "### Table 1: C=7 ###"
run_one 7 96

echo "### Table 2: C=600 ###"
run_one 600 96

echo "### Table 3: context-length ablation (C=7) ###"
for L in $CONTEXTS; do run_one 7 "$L"; done

echo
echo "All outputs in $OUT_DIR"
ls -1 "$OUT_DIR"/*MERGED.csv 2>/dev/null | sed 's/^/  /'
