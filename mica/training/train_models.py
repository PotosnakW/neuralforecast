import numpy as np
import pandas as pd
import argparse
import os

from neuralforecast import NeuralForecast

from experiment_datasets import *
from experiment_models import *


def main(args):
    df, h, val_size, test_size, freq = get_datasets(args)
    df.ds = pd.to_datetime(df.ds, format='%Y-%m-%d %H:%M:%S')

    args.h = h
    args.freq = freq
    args.n_series = len(df.unique_id.unique())
    args.input_size = int(h * args.input_size_h_multiplier)

    dataset_file_name = args.dataset_name.replace('/', '_')
    args.save_path = f'{args.save_dir}/{args.experiment_name}/{dataset_file_name}/rs{args.random_seed}_ishm{args.input_size_h_multiplier}_h{args.h}'
    os.makedirs(args.save_path, exist_ok=True)
    
    models = get_models(args)
    fcst = NeuralForecast(
        freq=freq,
        models=models,
    )

    fcst_df = fcst.cross_validation(
        df=df, 
        val_size=val_size,
        test_size=test_size,
        step_size=1,
        n_windows=None,
    )
    fcst.save(
        path=args.save_path,
        model_index=None,
        overwrite=True,
        save_dataset=False
    )
    fcst_df.to_csv(args.save_path+'/forecasts.csv', index=False)

def parse_args():
    desc = "Example of hyperparameter tuning"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset_name', type=str, help='control of hyperopt sample')
    parser.add_argument('--experiment_name', type=str, help='control which models are trained')
    parser.add_argument('--save_dir', type=str, help='directory where new results folder will be created')
    parser.add_argument('--random_seed', type=int, help='random seed for experiments')
    parser.add_argument('--num_samples', type=int, help='number of hyperopt samples')
    parser.add_argument('--input_size_h_multiplier', type=int, help='multiplier for horizon-based input size')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    if args is None:
        exit()

    if args.experiment_name in ['vanilla_pca_t5tiny', 'vanilla_ica_t5tiny']:
        raise Exception("'Must use train_models_pca.py for experiments 'vanilla_pca_t5tiny' or 'vanilla_ica_t5tiny'.")
    
    main(args)
