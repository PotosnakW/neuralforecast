import os
import itertools
import subprocess
from multiprocessing import Process
from datetime import datetime


DATASET_NAMES = [
    'simglucose',
    'iowa_ihop_smex_windspeed',
    'iowa_plows_windspeed',
    'LOOP_SEATTLE/D',
    'M_DENSE/D',
    'M_DENSE/H',
    'covid_deaths',
    'ett1/D',
    'ett1/H',
    'ett1/W',
    'ett2/D',
    'ett2/H',
    'ett2/W',
    'jena_weather/D',
    'jena_weather/H',
    'solar/D',
    'solar/H',
    'solar/W',
    'electricity/D',
    'electricity/H',
    'electricity/W',
]
EXPERIMENT_NAMES = [
    'vanilla_t5tiny',
    'infini_mlpquerymixer_t5tiny',
    'infini_poolmean_mlpquerymixer_t5tiny',
    'itransformer_baseline',
    'crossformer_baseline',
    'timerxl_baseline',
    'tsmixer_baseline',
    'timemixer_baseline',
    'multivariateMLP_baseline',
    'chronos2.0_baseline',
    'infini_mlpquerymixer_t5tiny_static_weights',
    'infini_mlpquerymixer_t5tiny_dynamic_weights',
]
GPU_INDICES = [0, 1, 2, 3]
RANDOM_SEEDS = [1, 2, 3, 4, 5]
FILE_NAME = 'train_models'
SAVE_DIR = '../exp_results'  # Add your path here
GIFT_EVAL_DIR = '../GIFT_EVAL_DIR' # Add your path here
CONDA_DIR = '~/miniconda3'  # Add your path here


def run_experiment(gpu_id, experiment_name, dataset_name, random_seed):
    """Run a single experiment, blocking until it completes."""
    experiment_cmd = (
        f"source {CONDA_DIR}/bin/activate && "
        f"conda activate neuralforecast && "
        f"export GIFT_EVAL={GIFT_EVAL_DIR} && "
        f"CUDA_VISIBLE_DEVICES={gpu_id} python -m {FILE_NAME} "
        f"    --dataset_name {dataset_name} "
        f"    --experiment_name {experiment_name} "
        f"    --save_dir {SAVE_DIR} "
        f"    --random_seed {random_seed} "
        f"    --num_samples 1 "
        f"    --input_size_h_multiplier 2"
    )

    start = datetime.now()
    print(f"[GPU {gpu_id}] START  {experiment_name} | {dataset_name} | seed={random_seed}  ({start.strftime('%H:%M:%S')})", flush=True)

    result = subprocess.run(
        experiment_cmd,
        shell=True,
        executable='/bin/bash',
    )

    end = datetime.now()
    elapsed = end - start
    status = "DONE  " if result.returncode == 0 else "FAILED"
    print(f"[GPU {gpu_id}] {status} {experiment_name} | {dataset_name} | seed={random_seed}  "
          f"(elapsed: {str(elapsed).split('.')[0]})", flush=True)


def worker_process(gpu_id, experiment_queue):
    """Worker process for a single GPU - runs one experiment at a time."""
    for experiment_name, dataset_name, random_seed in experiment_queue:
        run_experiment(gpu_id, experiment_name, dataset_name, random_seed)


def print_experiment_plan(experiments_per_gpu):
    """Print detailed experiment plan."""
    print("\n" + "="*80)
    print("EXPERIMENT PLAN")
    print("="*80)
    print(f"Experiments:   {EXPERIMENT_NAMES}")
    print(f"Save Dir:      {SAVE_DIR}")
    print(f"Start Time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    total = sum(len(exps) for exps in experiments_per_gpu.values())
    print(f"\nTotal Experiments : {total}")
    print(f"GPUs              : {GPU_INDICES}")
    print(f"Experiment Names  : {len(EXPERIMENT_NAMES)}")
    print(f"Datasets          : {len(DATASET_NAMES)}")
    print(f"Random Seeds      : {RANDOM_SEEDS}")

    print("\n" + "-"*80)
    print("GPU DISTRIBUTION:")
    print("-"*80)

    for gpu_id, exps in experiments_per_gpu.items():
        print(f"\nGPU {gpu_id}: {len(exps)} experiments")
        print("  " + "-"*76)
        for idx, (experiment_name, dataset_name, random_seed) in enumerate(exps, 1):
            print(f"  {idx:3d}. {experiment_name:35s} | {dataset_name:25s} | seed={random_seed}")

    print("\n" + "="*80)


def get_user_confirmation():
    """Get user confirmation to proceed."""
    while True:
        response = input("Do you want to launch these experiments? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            print("\nLaunching experiments...")
            return True
        elif response in ['n', 'no']:
            print("\nExperiments cancelled.")
            return False
        else:
            print("Please enter 'y' or 'n'")


def main():
    # Outer loop: experiment names. Inner loop: datasets x seeds.
    experiments = [
        (experiment_name, dataset_name, random_seed)
        for experiment_name in EXPERIMENT_NAMES
        for dataset_name, random_seed in itertools.product(DATASET_NAMES, RANDOM_SEEDS)
    ]

    # Round-robin distribution across GPUs
    experiments_per_gpu = {gpu_id: [] for gpu_id in GPU_INDICES}
    for idx, exp in enumerate(experiments):
        gpu_id = GPU_INDICES[idx % len(GPU_INDICES)]
        experiments_per_gpu[gpu_id].append(exp)

    print_experiment_plan(experiments_per_gpu)

    if not get_user_confirmation():
        return

    print("\n" + "="*80)
    print("LAUNCHING PROCESSES...")
    print("="*80 + "\n")

    processes = []
    for gpu_id, exp_queue in experiments_per_gpu.items():
        print(f"Starting worker process for GPU {gpu_id} ({len(exp_queue)} experiments queued)...")
        p = Process(target=worker_process, args=(gpu_id, exp_queue))
        p.start()
        processes.append(p)

    print(f"\nAll {len(processes)} GPU workers started!")
    print("-"*80)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("INTERRUPTED - Waiting for current experiments to finish...")
        print("="*80)
        for p in processes:
            p.join()

    print("\nAll experiments complete!")


if __name__ == '__main__':
    main()