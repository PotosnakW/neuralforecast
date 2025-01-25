import os

device=4
save_dir = 'ADD HERE'
file_name = 'topkfourier_script'
experiment = 'topkfourier'

for topk in [1, 2, 3, 4, 5]:

    #---------------------------- ETTm2 data --------------------------------
    os.system(f'CUDA_VISIBLE_DEVICES={device} python -m {file_name} \
        --dataset_name synthetic_sinusoid_composition \
        --experiment_name {experiment} \
        --n_series 100 \
        --n_compositions 2 \
        --ood_ratio 0 \
        --save_dir {save_dir} \
        --seq_len 1200 \
        --freq min \
        --topk {topk} \
                  ')

    #---------------------------- ETTm2 data --------------------------------
    os.system(f'CUDA_VISIBLE_DEVICES={device} python -m {file_name} \
        --dataset_name ettm2_100_series \
        --experiment_name {experiment} \
        --n_series 100 \
        --n_compositions 100 \
        --ood_ratio 0 \
        --save_dir {save_dir} \
        --seq_len 1056 \
        --freq 15min \
        --topk {topk} \
                    ')

    # ---------------------------- ECL data --------------------------------
    os.system(f'CUDA_VISIBLE_DEVICES={device} python -m {file_name} \
        --dataset_name ecl_100_series \
        --experiment_name {experiment} \
        --n_series 100 \
        --n_compositions 100 \
        --ood_ratio 0 \
        --save_dir {save_dir} \
        --seq_len 1056 \
        --freq h \
        --topk {topk} \
                ')

    # ---------------------------- Solar data --------------------------------
    os.system(f'CUDA_VISIBLE_DEVICES={device} python -m {file_name} \
        --dataset_name solar1h_100_series \
        --experiment_name {experiment} \
        --n_series 100 \
        --n_compositions 100 \
        --ood_ratio 0 \
        --save_dir {save_dir} \
        --seq_len 1056 \
        --freq h \
        --topk {topk} \
                ')

    # ---------------------------- Subseasonal data --------------------------------
    os.system(f'CUDA_VISIBLE_DEVICES={device} python -m {file_name} \
        --dataset_name subseasonal_100_series \
        --experiment_name {experiment} \
        --n_series 100 \
        --n_compositions 100 \
        --ood_ratio 0 \
        --save_dir {save_dir} \
        --seq_len 1056 \
        --freq W \
        --topk {topk} \
            ')

    # ---------------------------- Loop Seattle data --------------------------------
    os.system(f'CUDA_VISIBLE_DEVICES={device} python -m {file_name} \
        --dataset_name loopseattle_100_series \
        --experiment_name {experiment} \
        --n_series 100 \
        --n_compositions 100 \
        --ood_ratio 0 \
        --save_dir {save_dir} \
        --seq_len 1056 \
        --freq h \
        --topk {topk} \
            ')
