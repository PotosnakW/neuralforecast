import pandas as pd
import numpy as np
import os
import argparse

from neuralforecast.losses.numpy import mae, rmse
from experiment_datasets import *


def _evaluate_dataset(Y_hat_df, model_name, av_mask=None):
    results_df = Y_hat_df.copy()
    
    if av_mask is not None:
        results_df['cutoff'] = pd.to_datetime(results_df['cutoff'])
        av_mask['cutoff'] = pd.to_datetime(av_mask['cutoff'])

        # Convert unique_id to same type (string) in both dataframes
        results_df['unique_id'] = results_df['unique_id'].astype(str)
        av_mask['unique_id'] = av_mask['unique_id'].astype(str)
        
        # Filter values with at least 1 available mask in input window
        results_df = results_df.merge(
            av_mask[['unique_id', 'cutoff', 'sum_av_mask']], 
            on=['unique_id', 'cutoff'], 
            how='left'
        )
        results_df = results_df[results_df['sum_av_mask'] > 0].reset_index(drop=True)
    
    if 'available_mask' in results_df.columns:
        # Filter ffill values of y
        results_df = results_df[results_df['available_mask'] == 1]
    
    mae_result = mae(results_df['y'], results_df[model_name])
    rmse_result = rmse(results_df['y'], results_df[model_name])
    
    return mae_result, rmse_result

def _generate_av_mask(df, horizon):
    data = df.copy()
    av_mask = (data.groupby('unique_id', group_keys=False)
                   .apply(lambda x: x.assign(
                       sum_av_mask=x['available_mask'].rolling(
                           window=2*horizon, # Fixed input size
                           min_periods=1
                       ).sum()),
                    )
                   .reset_index(drop=True)
                   .rename(columns={'ds': 'cutoff'}))
    
    return av_mask

def get_results_df(results_dir, dataset_names, experiment_name, random_seeds):
    all_datasets_df = pd.DataFrame()
    for dataset_name in dataset_names:
        class Args:
            pass
        args = Args()
        args.dataset_name = dataset_name

        if dataset_name == 'iowa_ihop_smex_windspeed':
            df = pd.read_csv('../datasets/preprocessed_iowa_ihop_smex02_dataset.csv')
            df.ds = pd.to_datetime(df.ds, format='%Y-%m-%d %H:%M:%S')
            df = (df.groupby('unique_id')
                .resample('5min', on='ds')
                .agg({'y': 'mean', 'available_mask': 'max'})  # or 'min', 'first', etc.
                .reset_index())
            h = 24
    
        elif dataset_name == 'iowa_plows_windspeed':
            df = pd.read_csv('../datasets/preprocessed_iowa_plows_dataset.csv')
            df.ds = pd.to_datetime(df.ds, format='%Y-%m-%d %H:%M:%S')
            df = (df.groupby('unique_id')
                .resample('5min', on='ds')
                .agg({'y': 'mean', 'available_mask': 'max'})  # or 'min', 'first', etc.
                .reset_index())
            h = 24
    
        else:      
            try:
                df, h, _, _, _ = get_datasets(args)
            except Exception as e:
                print(f'Error loading dataset {dataset_name}: {e}')
                continue
        
        av_mask = None
        if 'available_mask' in df.columns:
            # Check if available_mask has any zeros (i.e., not all ones)
            if not (df['available_mask'] == 1).all():
                print(f'Generating mask for {dataset_name}')
                av_mask = _generate_av_mask(df=df, horizon=h)
            else:
                print(f'Skipping mask generation for {dataset_name} (all values are 1)')
    
        # Check if results directory exists
        dataset_name_underscore = dataset_name.replace('/', '_')
        results_path = f'{results_dir}/{experiment_name}/{dataset_name_underscore}'
        if not os.path.exists(results_path):
            print(f'Warning: Path does not exist: {results_path}')
            continue
            
        rss_df = pd.DataFrame()    
        for fi in random_seeds:
            print(dataset_name, fi)
            forecast_path = f'{results_path}/rs{fi}_ishm2_h{h}/forecasts.csv'
            
            if experiment_name == 'statsforecast':
                # Try primary path first
                try:
                    results_df = pd.read_csv(forecast_path)
                except FileNotFoundError:
                    fallback_path = f'{results_dir}/statsforecast_refit_false/{dataset_name_underscore}'
                    forecast_path = f'{fallback_path}/rs{fi}_ishm2_h{h}/forecasts.csv'
                    try:
                        results_df = pd.read_csv(forecast_path)
                    except FileNotFoundError:
                        print(f'Warning: {forecast_path} not found in both locations')
                        continue
            else:
                try:
                    results_df = pd.read_csv(forecast_path)
                except FileNotFoundError:
                    print(f'Warning: {forecast_path} not found')
                    continue
            
            model_columns = [col for col in results_df.columns 
                           if col not in ['ds', 'cutoff', 'unique_id', 'y', 'available_mask']]
            
            if not model_columns:
                print(f'Warning: No model columns')
                continue
            
            rs_df = pd.DataFrame()
            for column in model_columns:
                try:
                    mae_result, rmse_result = _evaluate_dataset(
                        Y_hat_df=results_df, 
                        model_name=column,
                        av_mask=av_mask,
                    )
                    
                    column_metrics_df = pd.DataFrame(
                        [mae_result, rmse_result], 
                        index=[f'{column}_mae', f'{column}_rmse'], 
                        columns=[fi]
                    )
                    rs_df = pd.concat([rs_df, column_metrics_df], axis=0)
                except Exception as e:
                    print(f'Error evaluating {column}: {e}')
                    continue
            
            if not rs_df.empty:
                rss_df = pd.concat([rss_df, rs_df], axis=1)
        
        # Skip if no results were collected
        if rss_df.empty:
            print(f'Warning: No results collected for {dataset_name}')
            continue
        
        # Calculate mean and std
        rf_mean = pd.DataFrame(rss_df.mean(axis=1)).T.add_suffix('_mean')
        rf_sd = pd.DataFrame(rss_df.std(axis=1)).T.add_suffix('_sd')
        rf_metrics = pd.concat([rf_mean, rf_sd], axis=1)
        rf_metrics.index = [dataset_name_underscore]
        
        all_datasets_df = pd.concat([all_datasets_df, rf_metrics], axis=0)
    
    # Save results
    if not all_datasets_df.empty:
        all_datasets_df.reset_index(inplace=True, drop=False)
        all_datasets_df.rename(columns={'index': 'dataset'}, inplace=True)
        output_path = f'{results_dir}/{experiment_name}/results.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        all_datasets_df.to_csv(output_path, index=False)
        print('Results saved')
    else:
        print('Warning: No results to save')
    
    return all_datasets_df


results_dir = '../exp_results'

dataset_names = [
    'simglucose',
    'iowa_ihop_smex_windspeed',
    'iowa_plows_windspeed',
    'M_DENSE/H',
    'M_DENSE/D',
    'jena_weather/H',
    'jena_weather/D',
    'ett1/H',
    'ett1/D',
    'ett1/W',
    'ett2/H',
    'ett2/D',
    'ett2/W',
    'covid_deaths',
    'LOOP_SEATTLE/D',
    'electricity/H',
    'electricity/D',
    'electricity/W',
    'solar/H',
    'solar/D',
    'solar/W',
]

experiment_names = [
    'vanilla_t5tiny',
    'vanilla_pca_t5tiny',
    'infini_mlpmixer_t5tiny',
    'infini_mlpquerymixer_t5tiny',
    'infini_t5tiny',
    'infini_channelwise_t5tiny',
    'infini_layerwise_t5tiny',
    'infini_layerwise_channelwise_t5tiny',
    'multivariateMLP_baseline',
    'itransformer_baseline',
    'crossformer_baseline',
    'timerxl_baseline',
    'tsmixer_baseline',
    'statsforecast',
    'chronos2.0_baseline',
]


def parse_args():
    desc = "evaluation"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--experiment_name', type=str, help='experiment name')
    parser.add_argument('--GIFT_EVAL_path', type=str, help='GIFTEVAL repo path')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    if args is None:
        exit()

    if args.experiment_name not in experiment_names:
        raise Exception('experiment name not included.')
    
    os.environ['GIFT_EVAL'] = args.GIFT_EVAL_path
    
    if args.experiment_name in ['statsforecast', 'chronos2.0_baseline']:
        random_seeds = [1]
    else:
        random_seeds = [1, 2, 3, 4, 5]

    get_results_df(
        results_dir=results_dir, 
        dataset_names=dataset_names, 
        experiment_name=args.experiment_name,
        random_seeds=random_seeds,
    )
