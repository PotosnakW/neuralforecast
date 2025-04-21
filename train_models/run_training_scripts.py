import os

device = 4
num_samples = 20

os.system(f'CUDA_VISIBLE_DEVICES={device} python -m train_statsforecast_models \
    --results_dir ../results/ets_model \
    --horizon 6 \
    --input_size 120 \
    --experiment_id 0 \
    ')

for trial in range(8):
    os.system(f'CUDA_VISIBLE_DEVICES={device} python -m train_models_baseline \
        --results_dir ../results/multivariate_models \
        --horizon 6 \
        --input_size 120 \
        --num_samples {num_samples} \
        --experiment_id {trial} \
        ')

    os.system(f'CUDA_VISIBLE_DEVICES={device} python -m train_models_baseline_transformers \
        --results_dir ../results/univariate_models \
        --horizon 6 \
        --input_size 120 \
        --num_samples {num_samples} \
        --experiment_id {trial} \
        ')

    os.system(f'CUDA_VISIBLE_DEVICES={device} python -m train_models_baseline_rnn \
        --results_dir ../results/rnn_model \
        --horizon 6 \
        --input_size 120 \
        --num_samples 3 \
        --experiment_id {trial} \
        ')

    os.system(f'CUDA_VISIBLE_DEVICES={device} python -m train_models_pk \
        --results_dir ../results/multivariate_models \
        --horizon 6 \
        --input_size 120 \
        --num_samples {num_samples} \
        --experiment_id {trial} \
        ')

    os.system(f'CUDA_VISIBLE_DEVICES={device} python -m train_models_sumtotal \
        --results_dir ../results/multivariate_models \
        --horizon 6 \
        --input_size 120 \
        --num_samples {num_samples} \
        --experiment_id {trial} \
        ')
    