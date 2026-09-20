#!/usr/bin/env python3
"""Work out which array task IDs still need to run.

Parses EXPERIMENTS/DATASETS/SEEDS straight out of each .sbatch so the index
math can never drift from what the job scripts actually do. A task counts as
done when its results directory holds a forecasts.csv -- train_models.py
writes that last, after fcst.save(), so its presence means the combo finished.

Prints one "<script-stem> <task-id>" per line for everything still outstanding.
"""
import os
import re
import subprocess
import sys
from glob import glob
from pathlib import Path

SLURM_DIR = Path(__file__).resolve().parent
SAVE_DIR = os.environ.get(
    "SAVE_DIR", "/zfsauton/scratch/wpotosna/neuralforecast_mica/exp_results"
)
# Order matters: autosubmit.sh fills its queue slots from the top of this list,
# so whatever comes first gets scheduled first. The zero-shot baselines lead
# because they are only 42 inference-only tasks -- putting them behind the ~890
# training tasks would delay the baselines by days for no gain.
SCRIPTS = [
    "run_zeroshot_toto2",
    "run_zeroshot_toto2_313m",
    "run_zeroshot_timesfm3",
    "run_ciexcl_rerun",
    "run_new_datasets_full",
    "run_poolmean",
    "run_ecl_mlpquery",
]


def parse(stem):
    """Recover (experiments, datasets, seeds, array_max, job_name) from a sbatch."""
    src = (SLURM_DIR / f"{stem}.sbatch").read_text()

    def arr(name):
        m = re.search(name + r"=\(([^)]*)\)", src, re.S)
        return m.group(1).split() if m else None

    exps = arr("EXPERIMENTS") or [re.search(r"EXPERIMENT_NAME=(\S+)", src).group(1)]
    datasets = arr("DATASETS")
    # The zero-shot scripts have a scalar `SEED=1` rather than a SEEDS array,
    # because zero-shot inference is deterministic and extra seeds would just
    # recompute identical forecasts.
    seeds = arr("SEEDS") or [re.search(r"^SEED=(\S+)", src, re.M).group(1)]
    amax = int(re.search(r"#SBATCH --array=0-(\d+)", src).group(1))
    jobname = re.search(r"#SBATCH --job-name=(\S+)", src).group(1)
    return exps, datasets, seeds, amax, jobname


def combo_for(stem, exps, datasets, seeds, tid):
    per = len(datasets) * len(seeds)
    exp_i, rem = divmod(tid, per) if len(exps) > 1 else (0, tid)
    ds_i, seed_i = divmod(rem, len(seeds))
    return exps[exp_i], datasets[ds_i], seeds[seed_i]


def is_done(exp, dataset, seed):
    # mirrors train_models.py: {save_dir}/{exp}/{dataset with / -> _}/rs{seed}_ishm{m}_h{h}
    pat = f"{SAVE_DIR}/{exp}/{dataset.replace('/', '_')}/rs{seed}_ishm*_h*/forecasts.csv"
    return bool(glob(pat))


def in_flight():
    """{job_name: {array task ids currently queued or running}}"""
    out = {}
    try:
        r = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", "wpotosna"), "-h", "-r",
             "-o", "%j %K"],
            capture_output=True, text=True, timeout=60,
        )
    except Exception:
        return out
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        name, tid = parts
        if tid.isdigit():
            out.setdefault(name, set()).add(int(tid))
    return out


def main():
    flying = in_flight()
    rows = []
    for stem in SCRIPTS:
        exps, datasets, seeds, amax, jobname = parse(stem)
        busy = flying.get(jobname, set())
        for tid in range(amax + 1):
            if tid in busy:
                continue
            exp, ds, seed = combo_for(stem, exps, datasets, seeds, tid)
            # run_new_datasets_full deliberately no-ops this slice; run_ecl_mlpquery
            # owns it. It never writes results, so never treat it as outstanding.
            if stem == "run_new_datasets_full" and \
               exp == "infini_mlpquerymixer_t5tiny" and ds.startswith("electricity/"):
                continue
            if is_done(exp, ds, seed):
                continue
            rows.append((stem, tid))
    for stem, tid in rows:
        print(stem, tid)
    if "--summary" in sys.argv:
        from collections import Counter
        c = Counter(s for s, _ in rows)
        for stem in SCRIPTS:
            print(f"  {stem:28s} {c.get(stem, 0):4d} outstanding", file=sys.stderr)
        print(f"  {'TOTAL':28s} {len(rows):4d}", file=sys.stderr)


if __name__ == "__main__":
    main()
