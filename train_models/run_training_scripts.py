import os

device = 4
num_samples = 20

os.system(f'CUDA_VISIBLE_DEVICES={device} python -m train_statsforecast_models \
    --results_dir ../results/ \
    --horizon 6 \
    --input_size 120 \
    --experiment_id baseline_0 \
    ')

for trial in range(8):
    os.system(f'CUDA_VISIBLE_DEVICES={device} python -m train_models_baseline \
        --results_dir ../results/ \
        --horizon 6 \
        --input_size 120 \
        --num_samples {num_samples} \
        --experiment_id baseline_{trial} \
        ')

    os.system(f'CUDA_VISIBLE_DEVICES={device} python -m train_models_baseline_transformers \
        --results_dir ../results_univariate_transformers/ \
        --horizon 6 \
        --input_size 120 \
        --num_samples {num_samples} \
        --experiment_id baseline_{trial} \
        ')

    os.system(f'CUDA_VISIBLE_DEVICES={device} python -m train_models_baseline_rnn \
        --results_dir ../results_rnn/ \
        --horizon 6 \
        --input_size 120 \
        --num_samples 3 \
        --experiment_id baseline_{trial} \
        ')
    