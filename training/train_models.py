import numpy as np
import pandas as pd
import argparse
import os
import sys
import fsspec
import pickle

from neuralforecast import NeuralForecast

from experiment_datasets import *
from experiment_models import *


def main(args):

    component_df, aggregate_df = get_datasets(args)
    
    component_df.ds = pd.to_datetime(component_df.ds, format='%Y-%m-%d %H:%M:%S')
    aggregate_df.ds = pd.to_datetime(aggregate_df.ds, format='%Y-%m-%d %H:%M:%S')

    if args.experiment_mode == 'component':
        train_df = component_df.groupby('unique_id').head(args.seq_len-args.h)

    elif args.experiment_mode == 'aggregate':
        train_df = aggregate_df.groupby('unique_id').head(args.seq_len-args.h)

    models = get_models(args)
    nf = NeuralForecast(models=models, freq=args.freq)
    nf.fit(df=train_df, val_size=args.val_size)

    nf.save(path=args.save_path, 
                model_index=None, 
                overwrite=True, 
                save_dataset=False,
              )

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
    parser.add_argument('--random_seed', type=int, help='random seed for experiments')

    
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    if args is None:
        exit()

    args.save_path = f'{args.save_dir}/results_randomseed{args.random_seed}/{args.experiment_name}/{args.dataset_name}_{args.n_compositions}bases/{args.experiment_mode}/{args.n_series}_series'
    args.ckpt_path = args.save_path+'/ckpts'

    if args.experiment_name not in ['tokenlen_ablation',
                                    'contextlen_ablation',
                                    'tokenization_ablation',
                                    'pe_ablation',
                                    'loss_ablation',
                                    'attn_ablation',
                                    'proj_ablation',
                                    'scaler_ablation',
                                    'decomp_ablation',
                                    'size_ablation',
                                    'nont5models',
                                    ]:
        raise Exception('Experiment not included.')
    
    main(args)

