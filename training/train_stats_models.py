import numpy as np
import pandas as pd
import argparse
import os
import sys

from statsforecast import StatsForecast
from statsforecast.models import ARIMA, AutoETS 

from experiment_datasets import *


def main(args):

    component_df, aggregate_df = get_datasets(args)

    train_df = aggregate_df.groupby('unique_id').head(args.seq_len-args.h)
    test_df = aggregate_df.groupby('unique_id').tail(args.h)

    if (args.dataset_name == 'ecl_100_series')|(args.dataset_name == 'solar1h_100_series')|(args.dataset_name == 'loopseattle_100_series'):
        season_length = 24
    if args.dataset_name == 'ettm2_100_series':
        season_length = 96
    if args.dataset_name == 'subseasonal_100_series':
        season_length = 52
    if args.dataset_name == 'synthetic_sinusoid_composition':
        season_length = 48

    if args.experiment_name = 'arima':

        all_fcsts = pd.DataFrame()
        for id_ in train_df.unique_id.unique():
            print(id_)
            train_df_id = train_df[train_df.unique_id==id_].copy()
            test_df_id = test_df[test_df.unique_id==id_].copy()
    
            sf = ARIMA(order=(0, 0, 2), season_length=season_length) # better model
            #sf = ARIMA(order=(1, 0, 1), season_length=12)
            model = sf.fit(y=train_df_id.y.values)
    
            y_hat_dict = model.predict(h=args.h, level=None)
            
            fcsts_df = pd.DataFrame(y_hat_dict['mean'], columns=['ARIMA'])
            fcsts_df['ds'] = test_df_id.ds.values
            fcsts_df['unique_id'] = test_df_id.unique_id.values
            fcsts_df['y'] = test_df_id.y.values
        
            all_fcsts = pd.concat([all_fcsts, fcsts_df], axis=0)

    elif args.experiment_name == 'ets':
        sf = StatsForecast(models=[AutoETS(season_length=season_length, model='ZNA')], freq=args.freq)
        model = sf.fit(df=train_df)
        all_fcsts = model.predict(h=args.h)
        all_fcsts['y'] = test_df['y'].values

    os.makedirs(args.save_path, exist_ok=True)
    all_fcsts.to_csv(args.save_path+'/aggregate_model_aggregate_dataset_fcsts.csv', index=False)

def parse_args():
    desc = "Example of hyperparameter tuning"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset_name', type=str, help='control of hyperopt sample')
    parser.add_argument('--experiment_name', type=str, help='control which models are trained')
    parser.add_argument('--experiment_mode', type=str, help='control which datasets are used to train models')
    parser.add_argument('--n_series', type=int, help='control number of timeseries examples')
    parser.add_argument('--n_compositions', type=int, help='control number of composite basis function signals')
    parser.add_argument('--save_dir', type=str, help='directory where new results folder will be created')
    parser.add_argument('--seq_len', type=int, help='dataset sequence length')
    parser.add_argument('--h', type=int, help='forecast horizon')
    parser.add_argument('--val_size', type=int, help='forecast horizon')
    parser.add_argument('--freq', type=str, help='frequency of the data')
    
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    if args is None:
        exit()

    args.save_path = f'{args.save_dir}/{args.experiment_name}/{args.dataset_name}_{args.n_compositions}bases/{args.experiment_mode}/{args.n_series}_samples'

    if args.experiment_name not in ['arima', 'ets']:
        raise Exception('Experiment not included.')

    main(args)