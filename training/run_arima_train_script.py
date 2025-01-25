import os

device=4
save_dir = '/home/extra_scratch/wpotosna/results_randomseed1'
file_name = 'train_stats_models' 
experiment = 'arima' #'ets'
experiment_mode = 'aggregate'

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
        ')
