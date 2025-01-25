import os

device=5

save_dir = 'ADD HERE'
file_name = 'eval_models'
experiment_mode = 'aggregate' #'component'
experiments = [
    'pe_ablation',
    'contextlen_ablation',
    'size_ablation',
    'architecture_ablation', 
    'scaler_ablation',
    'loss_ablation',
    'tokenlen_ablation',
    'proj_ablation',
    'tokenization_ablation',
    'decomp_ablation',
    'nont5models',
    ]
random_seeds = [1, 5, 10]


for experiment in experiments:
    for random_seed in random_seeds:
    
        # ---------------------------- synthetic sinusoid data --------------------------------
        os.system(f'CUDA_VISIBLE_DEVICES={device} python -m {file_name} \
            --dataset_name synthetic_sinusoid_composition \
            --experiment_name {experiment} \
            --experiment_mode {experiment_mode} \
            --n_series 100 \
            --n_compositions 2 \
            --save_dir {save_dir} \
            --seq_len 1200 \
            --h 192 \
            --random_seed {random_seed} \
            --eval_mode aggregate \
                  ')
    
        #---------------------------- ETTm2 data --------------------------------
        os.system(f'CUDA_VISIBLE_DEVICES={device} python -m {file_name} \
            --dataset_name ettm2_100_series \
            --experiment_name {experiment} \
            --experiment_mode {experiment_mode} \
            --n_series 100 \
            --n_compositions 100 \
            --save_dir {save_dir} \
            --seq_len 1056 \
            --h 48 \
            --random_seed {random_seed} \
            --eval_mode aggregate \
                      ')
    
        # ---------------------------- ECL data --------------------------------
        os.system(f'CUDA_VISIBLE_DEVICES={device} python -m {file_name} \
            --dataset_name ecl_100_series \
            --experiment_name {experiment} \
            --experiment_mode {experiment_mode} \
            --n_series 100 \
            --n_compositions 100 \
            --save_dir {save_dir} \
            --seq_len 1056 \
            --h 48 \
            --random_seed {random_seed} \
            --eval_mode aggregate \
                ')
    
        # ---------------------------- Solar data --------------------------------
        os.system(f'CUDA_VISIBLE_DEVICES={device} python -m {file_name} \
            --dataset_name solar1h_100_series \
            --experiment_name {experiment} \
            --experiment_mode {experiment_mode} \
            --n_series 100 \
            --n_compositions 100 \
            --save_dir {save_dir} \
            --seq_len 1056 \
            --h 48 \
            --random_seed {random_seed} \
            --eval_mode aggregate \
                ')

        # ---------------------------- Subseasonal data --------------------------------
        os.system(f'CUDA_VISIBLE_DEVICES={device} python -m {file_name} \
            --dataset_name subseasonal_100_series \
            --experiment_name {experiment} \
            --experiment_mode {experiment_mode} \
            --n_series 100 \
            --n_compositions 100 \
            --save_dir {save_dir} \
            --seq_len 1056 \
            --h 48 \
            --random_seed {random_seed} \
            --eval_mode aggregate \
               ')
        
        # ---------------------------- Loop Seattle data --------------------------------
        os.system(f'CUDA_VISIBLE_DEVICES={device} python -m {file_name} \
            --dataset_name loopseattle_100_series \
            --experiment_name {experiment} \
            --experiment_mode {experiment_mode} \
            --n_series 100 \
            --n_compositions 100 \
            --save_dir {save_dir} \
            --seq_len 1056 \
            --h 48 \
            --random_seed {random_seed} \
            --eval_mode aggregate \
                ')
