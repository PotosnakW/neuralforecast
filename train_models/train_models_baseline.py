import os
import pickle
import time
import argparse
import pandas as pd
import numpy as np
import sys

from experiment_space_baseline import *
from data_parameters import get_data_parameters

from ray.tune.search.hyperopt import HyperOptSearch

from neuralforecast.auto import AutoNHITS, AutoNBEATSx, AutoMLP, AutoRNN, AutoDLinear, AutoLSTM, AutoTCN, AutoTFT
from neuralforecast.core import NeuralForecast
from neuralforecast.losses.pytorch import HuberLoss

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def main(args):

    #----------------------------------------------- Load Data -----------------------------------------------#
    data_dir, static_dir, val_size, test_size, freq, horizons, input_size, exog = get_data_parameters(args)

    Y_df = pd.read_csv(data_dir)
    if Y_df.ds.dtype != '<M8[ns]':
        Y_df.ds = pd.to_datetime(Y_df.ds, format='%Y-%m-%d %H:%M:%S')

    if static_dir is not None:
        static_df = pd.read_csv(static_dir)
        
    args.exog = exog
    args.n_series = len(Y_df.unique_id.unique())
    args.freq = freq
    args.input_size = input_size

    #----------------------------------------------- Training -----------------------------------------------#
    # Fit and predict
    for horizon in horizons:
        args.horizon = horizon
        print(50*'-', dataset, 50*'-')
        print(50*'-', horizon, 50*'-')
        start = time.time()
        
        results_dir = f'../results/{args.dataset}_{args.horizon}/baseline_models/trial_{args.experiment_id}'
        os.makedirs(results_dir, exist_ok = True)

        nhits_config = get_nhits_experiment_space(args)
        mlp_config = get_mlp_experiment_space(args)
        rnn_config = get_rnn_experiment_space(args)
        lstm_config = get_lstm_experiment_space(args)
        tcn_config = get_tcn_experiment_space(args)
        dlinear_config = get_dlinear_experiment_space(args)
        tft_config = get_tft_experiment_space(args)
        nbeats_config = get_nbeats_experiment_space(args)

        fcst = NeuralForecast(freq=freq,
                              models=[
                                      AutoNHITS(h=args.horizon, 
                                                config=nhits_config,
                                                loss=HuberLoss(),
                                                search_alg=HyperOptSearch(),
                                                num_samples=args.num_samples),
                                      AutoMLP(h=args.horizon, 
                                              config=mlp_config,
                                              loss=HuberLoss(),
                                              search_alg=HyperOptSearch(),
                                              num_samples=args.num_samples),
                                       AutoRNN(h=args.horizon,
                                               config=rnn_config,
                                               loss=HuberLoss(),
                                               search_alg=HyperOptSearch(),
                                               num_samples=args.num_samples),
                                       AutoLSTM(h=args.horizon, 
                                                config=lstm_config,
                                                loss=HuberLoss(),
                                                search_alg=HyperOptSearch(),
                                                num_samples=args.num_samples),
                                       AutoTCN(h=args.horizon, 
                                               config=tcn_config,
                                               loss=HuberLoss(),
                                               search_alg=HyperOptSearch(),
                                               num_samples=args.num_samples),
                                       AutoDLinear(h=args.horizon, 
                                               config=dlinear_config,
                                               loss=HuberLoss(),
                                               search_alg=HyperOptSearch(),
                                               num_samples=args.num_samples),
                                       AutoTFT(h=args.horizon, 
                                               config=tft_config,
                                               n_series=args.n_series,
                                               loss=HuberLoss(),
                                               search_alg=HyperOptSearch(),
                                               num_samples=args.num_samples),
                                       AutoNBEATSx(h=args.horizon, 
                                               config=nbeats_config,
                                               loss=HuberLoss(),
                                               search_alg=HyperOptSearch(),
                                               num_samples=args.num_samples),
                                    ],)

        fcst_df = fcst.cross_validation(df=Y_df, 
                                        static_df=static_df,
                                        val_size=val_size,
                                        test_size=test_size,
                                        step_size=1,
                                        n_windows=None,
                                        )
        fcst_df.to_csv(results_dir+f'/forecasts.csv', index=False)

        fcst.save(path=results_dir,
                  model_index=None,
                  overwrite=True,
                  save_dataset=False)
        
        print('Time: ', time.time() - start)
        
def parse_args():
    desc = "Example of hyperparameter tuning"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--num_samples', type=int, help='control of hyperopt sample')
    parser.add_argument('--experiment_id', default=None, required=False, type=str, help='string to identify experiment')
    
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    if args is None:
        exit()

    datasets = ['ohiot1dm',
                'ohiot1dm_exog',
                'simglucose',
                'simglucose_exog',
               ]
    # datasets = ['ohiot1dm_exog_#540',
    #             'ohiot1dm_exog_#544',
    #             'ohiot1dm_exog_#552',
    #             'ohiot1dm_exog_#559',
    #             'ohiot1dm_exog_#563',
    #             'ohiot1dm_exog_#567',
    #             'ohiot1dm_exog_#570',
    #             'ohiot1dm_exog_#575',
    #             'ohiot1dm_exog_#584',
    #             'ohiot1dm_exog_#588',
    #             'ohiot1dm_exog_#591',
    #             'ohiot1dm_exog_#596'
    #           ]
    
    for dataset in datasets:
        args.dataset = dataset
    
        main(args)
