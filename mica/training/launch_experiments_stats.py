import subprocess


DATASET_NAMES = [
    'simglucose',
    'iowa_ihop_smex_windspeed',
    'iowa_plows_windspeed',
    'M_DENSE/D',
    'M_DENSE/H',
    'jena_weather/D',
    'jena_weather/H',
    'hierarchical_sales/D',
    'hierarchical_sales/W',
    'ett1/D',
    'ett1/H',
    'ett1/W',
    'ett2/D',
    'ett2/H',
    'ett2/W',
    'covid_deaths',
    'LOOP_SEATTLE/D',
    'solar/D',
    'solar/H',
    'solar/W',
]

FILE_NAME = 'train_models_stats' 
EXPERIMENT_NAME = 'statsforecast'
SAVE_DIR = '../icml_exp_results' # CHANGE TO YOUR WORKING DIRECTORY

for dataset_name in DATASET_NAMES:
    print(dataset_name)
    cmd = f"""
        CUDA_VISIBLE_DEVICES=0 python -m {FILE_NAME} \
            --dataset_name {dataset_name} \
            --experiment_name {EXPERIMENT_NAME} \
            --save_dir {SAVE_DIR} \
            --random_seed 1 \
            --num_samples 1 \
            --input_size_h_multiplier 2
        """
    
    print('launch experiment')
    result = subprocess.run(cmd, shell=True, executable='/bin/bash', 
                          capture_output=True, text=True)
    
    