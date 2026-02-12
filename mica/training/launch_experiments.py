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
    'LOOP_SEATTLE/H',
    'M_DENSE/D',
    'M_DENSE/H',
    'covid_deaths',
    'ett1/15T',
    'ett1/D',
    'ett1/H',
    'ett1/W',
    'ett2/15T',
    'ett2/D',
    'ett2/H',
    'ett2/W',
    'jena_weather/D',
    'jena_weather/H',
    'solar/D',
    'solar/H',
    'solar/W',
]
GPU_INDICES = [0, 1, 2, 3]  # Specify which GPUs to use
RANDOM_SEEDS = [1, 2, 3, 4, 5]
FILE_NAME = 'train_models' # Must use 'train_models_pca for experiment_name in ['vanilla_pca_t5tiny']
EXPERIMENT_NAME = 'vanilla_t5tiny' # CHANGE TO SPECIFY EXPERIMENT
SAVE_DIR = '../exp_results' # CHANGE TO YOUR WORKING DIRECTORY
GIFT_EVAL_DIR = "../GiftEval" # CHANGE TO YOUR GIFT EVAL DIRECTORY
CONDA_DIR = f'/home/miniconda3' #TODO CHANGE TO YOUR CONDA DIRECTORY

def run_experiment(gpu_id, dataset_name, random_seed):
    """Run a single experiment"""
    dataset_clean = dataset_name.replace('/', '_').replace(' ', '_')
    session_name = f"gpu{gpu_id}_{dataset_clean}_seed{random_seed}_{EXPERIMENT_NAME}"

    experiment_cmd = f"""
    source {CONDA_DIR}/bin/activate && \
    conda activate neuralforecast && \
    export GIFT_EVAL={GIFT_EVAL_DIR} && \
    CUDA_VISIBLE_DEVICES={gpu_id} python -m {FILE_NAME} \
        --dataset_name {dataset_name} \
        --experiment_name {EXPERIMENT_NAME} \
        --save_dir {SAVE_DIR} \
        --random_seed {random_seed} \
        --num_samples 1 \
        --input_size_h_multiplier 2
    """

    cmd = f"""
        tmux new-session -d -s {session_name} bash -c '
        {experiment_cmd}
        echo "Experiment completed. Press any key to close this tmux session."
        read -n 1
        '
        """
    
    print(f"[GPU {gpu_id}] Starting: {dataset_name} (seed={random_seed})")
    print(f"tmux session: {session_name}")
    
    result = subprocess.run(cmd, shell=True, executable='/bin/bash', 
                          capture_output=True, text=True)
            
def worker_process(gpu_id, experiment_queue):
    """Worker process for a single GPU"""
    import time
    for dataset_name, random_seed in experiment_queue:
        run_experiment(gpu_id, dataset_name, random_seed)
        time.sleep(2)  # Brief pause between experiments

def print_experiment_plan(experiments_per_gpu):
    """Print detailed experiment plan"""
    print("\n" + "="*80)
    print("EXPERIMENT PLAN")
    print("="*80)
    print(f"Experiment Name: {EXPERIMENT_NAME}")
    print(f"Save Directory: {SAVE_DIR}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    total_experiments = sum(len(exps) for exps in experiments_per_gpu.values())
    print(f"\nTotal Experiments: {total_experiments}")
    print(f"GPU Indices: {GPU_INDICES}")
    print(f"Number of GPUs: {len(GPU_INDICES)}")
    print(f"Datasets: {len(DATASET_NAMES)}")
    print(f"Random Seeds: {RANDOM_SEEDS}")
    
    print("\n" + "-"*80)
    print("GPU DISTRIBUTION:")
    print("-"*80)
    
    for gpu_id, exps in experiments_per_gpu.items():
        print(f"\nGPU {gpu_id}: {len(exps)} experiments")
        print("  " + "-"*76)
        for idx, (dataset_name, random_seed) in enumerate(exps, 1):
            print(f"  {idx:2d}. {dataset_name:30s} | seed={random_seed}")
    
    print("\n" + "="*80)

def get_user_confirmation():
    """Get user confirmation to proceed"""
    print("\n" + "="*80)
    while True:
        response = input("Do you want to launch these experiments? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            print("\n✓ Launching experiments...")
            return True
        elif response in ['n', 'no']:
            print("\n✗ Experiments cancelled.")
            return False
        else:
            print("Please enter 'y' or 'n'")

def main():    
    # Create all experiments
    experiments = list(itertools.product(DATASET_NAMES, RANDOM_SEEDS))
    
    # Divide across specified GPUs
    experiments_per_gpu = {gpu_id: [] for gpu_id in GPU_INDICES}
    for idx, exp in enumerate(experiments):
        gpu_id = GPU_INDICES[idx % len(GPU_INDICES)]
        experiments_per_gpu[gpu_id].append(exp)
    
    # Print detailed plan
    print_experiment_plan(experiments_per_gpu)
    
    # Get confirmation
    if not get_user_confirmation():
        return
    
    # Launch processes
    print("\n" + "="*80)
    print("LAUNCHING PROCESSES...")
    print("="*80 + "\n")
    
    processes = []
    for gpu_id, exp_queue in experiments_per_gpu.items():
        print(f"Starting worker process for GPU {gpu_id}...")
        p = Process(target=worker_process, args=(gpu_id, exp_queue))
        p.start()
        processes.append(p)
    
    print(f"\n✓ All {len(processes)} GPU workers started!")
    print("-"*80)
    
    # Wait for all to complete
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("INTERRUPTED - Waiting for current experiments to finish...")
        print("="*80)
        for p in processes:
            p.join()

if __name__ == '__main__':
    main()
