import os
import pickle
import time
import argparse
import pandas as pd
import numpy as np
import sys

from experiment_space_sum_total import *
from data_parameters import get_data_parameters

from ray.tune.search.hyperopt import HyperOptSearch

from neuralforecast.auto import AutoNHITS_TREAT, AutoNBEATSx_TREAT, AutoTFT
from neuralforecast.core import NeuralForecast
from neuralforecast.losses.pytorch import MSE, HuberLoss

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
        
        results_dir = f'../results/{args.dataset}_{args.horizon}/sum_total_models/trial_{args.experiment_id}'
        os.makedirs(results_dir, exist_ok = True)
        
        nhits_sumtotal_config = get_nhits_sumtotal_experiment_space(args)
        nbeatsx_sumtotal_config1 = get_nbeatsx_sumtotal_experiment_space1(args)
        nbeatsx_sumtotal_config2 = get_nbeatsx_sumtotal_experiment_space2(args)
        nbeatsx_sumtotal_config3 = get_nbeatsx_sumtotal_experiment_space3(args)
        tft_sumtotal_config = get_tft_sumtotal_experiment_space(args)
            
        fcst = NeuralForecast(freq=freq,
                              models=[
                                    AutoNHITS_TREAT(h=args.horizon, 
                                                config=nhits_sumtotal_config,
                                                n_series=args.n_series,
                                                loss=HuberLoss(),
                                                search_alg=HyperOptSearch(),
                                                num_samples=args.num_samples),
                                    AutoTFT(h=args.horizon, 
                                                config=tft_sumtotal_config,
                                                n_series=args.n_series,
                                                loss=HuberLoss(),
                                                search_alg=HyperOptSearch(),
                                                num_samples=args.num_samples),
                                    AutoNBEATSx_TREAT(h=args.horizon, 
                                                config=nbeatsx_sumtotal_config3,
                                                n_series=args.n_series,
                                                loss=HuberLoss(),
                                                search_alg=HyperOptSearch(),
                                                num_samples=args.num_samples,
                                                alias='autonbeatsxtreat_cts'),
                                    ],)

        fcst_df = fcst.cross_validation(df=Y_df, 
                                        static_df=static_df,
                                        val_size=val_size,
                                        test_size=test_size, 
                                        step_size=1,
                                        n_windows=None
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
    parser.add_argument('--experiment_id', required=True, type=str, help='string to identify experiment')
    
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    if args is None:
        exit()

    datasets = ['ohiot1dm_exog',
                'simglucose_exog',
               ]
    
    for dataset in datasets:
        args.dataset = dataset

        main(args)
