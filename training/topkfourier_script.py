import numpy as np
import pandas as pd
import argparse
import os
import sys

from experiment_datasets import *

def _create_component_df(aggregate_df, seq_len, top_k, freq):

    ds = pd.date_range(start=aggregate_df.ds.min(),
                       periods=seq_len, 
                       freq=freq,
                      )
    
    n_series = aggregate_df.shape[0] // seq_len
    signals = aggregate_df.y.values.reshape(n_series, seq_len)
    basis_functions = get_fft_bases(signals, top_k)

    num_bases = basis_functions.shape[0]
    names = [f'composite{i}' for i in range(1, num_bases+1)]
    names = np.concatenate([np.repeat(i, seq_len) for i in names])
    
    df = pd.DataFrame(basis_functions.flatten(),
                     index=names,
                     columns=['topkfourier']
                     )
    df.index.name = 'unique_id'
    df['ds'] = np.concatenate([ds for _ in range(num_bases)])
    df['y'] = aggregate_df.y.values
    df.reset_index(drop=False, inplace=True)

    return df

def get_fft_bases(x, top_k):
    seq_len = x.shape[-1]
    ts = 1.0 / seq_len
    t = np.arange(1e-5, 1, ts)

    # Compute the DFT of the input signals
    dft = np.fft.fft(x, n=seq_len, axis=-1, norm='ortho')
    ks = np.arange(seq_len)

    # Compute magnitudes and sort by magnitude
    dft_magnitudes = np.abs(dft)  # Compute the absolute value
    sorted_indices = np.argsort(dft_magnitudes, axis=-1)[:, ::-1]  # Sort in descending order
    top_kvals = sorted_indices[:, :top_k]

    top_dft = np.take_along_axis(dft, top_kvals, axis=1)
    top_dft = np.expand_dims(top_dft, axis=-1).repeat(seq_len, axis=-1)

    top_kvals = np.expand_dims(top_kvals, axis=-1).repeat(seq_len, axis=-1)

    # Compute the basis functions
    t = np.expand_dims(t, axis=0)  # Match dimensions with x
    bs = np.exp(-2j * np.pi * top_kvals * t)  # [num_signals, top_k, seq_len]
    bs = np.flip(bs, axis=-1)  # Flip along the sequence dimension # [num_signals, top_k, seq_len]

    basis_functions = (top_dft*bs) / np.sqrt(seq_len)  # Normalize # [num_signals, top_k, seq_len]

    #correct for shift of 1
    basis_functions = np.pad(basis_functions, ((0, 0), (0, 0), (1, 0)), mode='edge')
    basis_functions = basis_functions[:, :, :seq_len]
        
    basis_functions = np.sum(basis_functions, axis=1)

    return basis_functions


def main(args):

    component_df, aggregate_df = get_datasets(args)

    top_k_bases_df = _create_component_df(aggregate_df, 
                                          seq_len=args.seq_len, 
                                          top_k=args.topk, 
                                          freq=args.freq
                                         )

    os.makedirs(args.save_path, exist_ok=True)
    top_k_bases_df.to_csv(args.save_path+'/aggregate_model_aggregate_dataset_fcsts.csv', index=False)

def parse_args():
    desc = "Example of hyperparameter tuning"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset_name', type=str, help='control of hyperopt sample')
    parser.add_argument('--experiment_name', type=str, help='control which models are trained')
    parser.add_argument('--n_series', type=int, help='control number of timeseries examples')
    parser.add_argument('--n_compositions', type=int, help='control number of composite basis function signals')
    parser.add_argument('--save_dir', type=str, help='directory where new results folder will be created')
    parser.add_argument('--seq_len', type=int, help='dataset sequence length')
    parser.add_argument('--freq', type=str, help='frequency of the data')
    parser.add_argument('--topk', type=int, help='top k fourier bases')
    
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    if args is None:
        exit()

    args.save_path = f'{args.save_dir}/{args.experiment_name}_top{args.topk}/{args.dataset_name}_{args.n_compositions}bases/aggregate/{args.n_series}_samples'

    if args.experiment_name != 'topkfourier':
        raise Exception('Experiment not included.')

    main(args)
