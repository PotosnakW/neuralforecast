import os

device=6

file_name = 'train_models'

# Edit these to run specific experiments
save_dir = 'ADD HERE'
experiment = 'architecture_ablation'
experiment_mode = 'aggregate'


random_seeds = [1, 5, 10]
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
        --val_size 192 \
        --freq min \
        --random_seed {random_seed} \
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
        --val_size 48 \
        --freq 15min \
        --random_seed {random_seed} \
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
        --val_size 48 \
        --freq h \
        --random_seed {random_seed} \
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
        --val_size 48 \
        --freq h \
        --random_seed {random_seed} \
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
        --val_size 48 \
        --freq W \
        --random_seed {random_seed} \
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
        --val_size 48 \
        --freq h \
        --random_seed {random_seed} \
            ')
