import os

device = 1

for trial in range(8):
    
    os.system(f'CUDA_VISIBLE_DEVICES={device} python -m train_statsforecast_models --experiment_id ETS_{trial}')

    os.system(f'CUDA_VISIBLE_DEVICES={device} python -m train_models_baseline --num_samples 20 --experiment_id baseline_{trial}')
    
    os.system(f'CUDA_VISIBLE_DEVICES={device} python -m train_models_sum_total --num_samples 20 --experiment_id sum_total_{trial}')
    
    os.system(f'CUDA_VISIBLE_DEVICES={device} python -m train_models_treat --num_samples 20 --experiment_id treat_{trial}')