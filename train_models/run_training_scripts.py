import os

device = 2
num_samples = 20

for trial in range(8):
    
    os.system(f'CUDA_VISIBLE_DEVICES={device} python -m train_statsforecast_models --experiment_id ETS_{trial}')

    os.system(f'CUDA_VISIBLE_DEVICES={device} python -m train_models_baseline --num_samples {num_samples} --experiment_id baseline_{trial}')
    
    os.system(f'CUDA_VISIBLE_DEVICES={device} python -m train_models_baseline_transformers --num_samples {num_samples} --experiment_id baseline_{trial}')
    
    os.system(f'CUDA_VISIBLE_DEVICES={device} python -m train_models_sum_total --num_samples {num_samples} --experiment_id sum_total_{trial}')
    
    os.system(f'CUDA_VISIBLE_DEVICES={device} python -m train_models_treat --num_samples {num_samples} --experiment_id treat_{trial}')