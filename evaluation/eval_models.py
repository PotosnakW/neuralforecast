import numpy as np
import pandas as pd
import argparse
import os
import sys
import yaml

from neuralforecast.core import NeuralForecast

sys.path.append('../preprocessing')
sys.path.append('../training')
from experiment_datasets import *
from experiment_models import *


def main(args):
    component_df, aggregate_df = get_datasets(args)
    
    if args.eval_mode == 'aggregate':
        input_df = aggregate_df
    elif args.eval_mode == 'component':
        input_df = component_df

    input_df.ds = pd.to_datetime(input_df.ds, format='%Y-%m-%d %H:%M:%S')
    test_df = input_df.groupby('unique_id').tail(args.h)
    input_df = input_df.groupby('unique_id').head(args.seq_len-args.h)

    nf = NeuralForecast.load(path=args.save_path)
    preds = nf.predict(df=input_df)

    preds.reset_index(drop=False, inplace=True)
    eval_df = pd.concat([preds.set_index(['unique_id', 'ds']), 
                         test_df.set_index(['unique_id', 'ds'])], 
                        axis=1)
    
    eval_df.reset_index().to_csv(args.save_path+f'/{args.experiment_mode}_model_{args.eval_mode}_dataset_fcsts.csv', index=False)

def parse_args():
    desc = "Example of hyperparameter tuning"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset_name', type=str, help='dataset used to train the model')
    parser.add_argument('--experiment_name', type=str, help='control which models are trained')
    parser.add_argument('--experiment_mode', type=str, help='control which datasets are used to train models')
    parser.add_argument('--n_series', type=int, help='control number of timeseries examples')
    parser.add_argument('--n_compositions', type=int, help='control number of composite basis function signals')
    parser.add_argument('--save_dir', type=str, help='directory where new results folder will be created')
    parser.add_argument('--seq_len', type=int, help='dataset sequence length')
    parser.add_argument('--h', type=int, help='forecast horizon')
    parser.add_argument('--random_seed', type=int, help='random seed for experiments')
    parser.add_argument('--eval_mode', type=str, help='control which dataset to evaluate the model')
    
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    if args is None:
        exit()

    args.save_path = f'{args.save_dir}/results_randomseed{str(random_seed)}/{args.experiment_name}/{args.dataset_name}_{args.n_compositions}bases/{args.experiment_mode}/{args.n_series}_series'

    main(args)

