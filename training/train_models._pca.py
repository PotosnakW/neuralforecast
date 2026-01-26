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

    df_transformed, transformer, train_mean, train_std, column_order = decorrelate_data(
        df=df, 
        val_size=val_size, 
        test_size=test_size, 
        method=args.decorrelate_method
    )

    args.h = h
    args.freq = freq
    args.n_series = len(df.unique_id.unique())
    args.input_size = int(h * args.input_size_h_multiplier)

    dataset_file_name = args.dataset_name.replace('/', '_')
    args.save_path = f'{args.save_dir}/{args.experiment_name}/{dataset_file_name}/rs{args.random_seed}_ishm{args.input_size_h_multiplier}_h{args.h}'
    os.makedirs(args.save_path, exist_ok=True)
    
    models = get_models(args)
    fcst = NeuralForecast(freq=freq, models=models)
    
    fcst_df = fcst.cross_validation(
        df=df_transformed, 
        val_size=val_size,
        test_size=test_size,
        step_size=1,
        n_windows=None
    )
    fcst.save(
        path=args.save_path,
        model_index=None,
        overwrite=True,
        save_dataset=False
    )
    
    assert fcst_df.groupby('unique_id').size().nunique() == 1
    
    fcst_df['row'] = fcst_df.groupby('unique_id').cumcount()
    reserved_cols = {'unique_id', 'ds', 'cutoff', 'y', 'y_original', 'row'}
    model_columns = [col for col in fcst_df.columns if col not in reserved_cols]
    
    for model_name in model_columns:
        fcst_df_pivot = fcst_df.pivot(index='row', columns='unique_id', values=model_name)
        fcst_df_pivot = fcst_df_pivot[column_order]
        
        # Inverse transform
        fcst_df_standardized = transformer.inverse_transform(fcst_df_pivot.values)
        fcst_df_original = fcst_df_standardized * train_std.values + train_mean.values
        
        # Map back
        for i, orig_id in enumerate(column_order):
            mask = fcst_df['unique_id'] == orig_id
            fcst_df.loc[mask, model_name] = fcst_df_original[:, i]
    
    fcst_df.drop(columns=['row', 'y'], inplace=True)

    fcst_df = fcst_df.merge(
        df_transformed[['unique_id', 'ds', 'y_original']],
        on=['unique_id', 'ds'],
        how='left'
    )
    fcst_df.rename(columns={'y_original': 'y'}, inplace=True)
    
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

    if args.experiment_name not in ['vanilla_pca_t5tiny', 'vanilla_ica_t5tiny']:
        raise Exception("'experiment_name' must be 'vanilla_pca_t5tiny' or 'vanilla_ica_t5tiny'.")

    if args.experiment_name == 'vanilla_pca_t5tiny': 
        args.decorrelate_method = 'pca'
    elif args.experiment_name == 'vanilla_ica_t5tiny':
        args.decorrelate_method = 'ica'

    main(args)
