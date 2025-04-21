import os
import pickle
import time
import argparse
import pandas as pd
import numpy as np
import sys

from statsforecast import StatsForecast
from statsforecast.models import AutoETS

def get_data_parameters(args):
    exog = {}
    
    if args.dataset == 'ohiot1dm':
        data_dir = '../datasets/ohiot1dm_exog_9_day_test.csv'
        static_dir = '../datasets/ohiot1dm_static.csv'
        val_size = 2691
        test_size = 2691
        freq = '5min'
        exog['stat_exog_list'] = ['559', '563', '570', '575', '588', 
                                  '591', '540', '544', '552', '567',
                                  '584', 'insulin_type_novalog', 'female',
                                  'age_20_40', 'age_40_60', 'pump_model_630G']
        exog['hist_exog_list'] = None
        exog['futr_exog_list'] = None 
            
    if args.dataset == 'simglucose':
        data_dir = '../datasets/simglucose_exog_9_day_test.csv'
        static_dir = './datasets/simglucose_static.csv'
        val_size = 2592
        test_size = 2592
        freq = '5min'
        exog['stat_exog_list'] = ['adolescent#001', 'adolescent#002', 'adolescent#003', 'adolescent#004', 'adolescent#005',
                                  'adolescent#006', 'adolescent#007', 'adolescent#008', 'adolescent#009', 'adolescent#010', 
                                  'adult#001', 'adult#002', 'adult#003', 'adult#004', 'adult#005',
                                  'adult#006', 'adult#007', 'adult#008', 'adult#009', 'adult#010',
                                  'child#001', 'child#002', 'child#003', 'child#004', 'child#005',
                                  'child#006', 'child#007', 'child#008', 'child#009', 
                                  'Age', 'BW', 'adolescent', 'adult']
        exog['hist_exog_list'] = None
        exog['futr_exog_list'] = None

    return data_dir, static_dir, val_size, test_size, freq, exog

def main(args):

    #----------------------------------------------- Load Data -----------------------------------------------#
    data_dir, static_dir, val_size, test_size, freq, exog = get_data_parameters(args)
    args.exog = exog
    
    Y_df = pd.read_csv(data_dir)
    if Y_df.ds.dtype != '<M8[ns]':
        Y_df.ds = pd.to_datetime(Y_df.ds, format='%Y-%m-%d %H:%M:%S')
        
    if static_dir is not None:
        static_df = pd.read_csv(static_dir)
        
    args.exog = exog
    args.freq = freq

    #----------------------------------------------- Training -----------------------------------------------#
    # Fit and predict
    print(50*'-', args.dataset, 50*'-')
    print(50*'-', args.horizon, 50*'-')
    print(50*'-', args.input_size, 50*'-')
    start = time.time()

    results_dir = f'{args.results_dir}/{args.dataset}_{args.horizon}/baseline_models/trial_{args.experiment_id}'
    os.makedirs(results_dir, exist_ok = True)
        
    fcst = StatsForecast(freq = freq,
                         models = [AutoETS(season_length = int(pd.Timedelta('1D')/pd.Timedelta(args.freq)),)]
                        )

    fcst_df = fcst.cross_validation(h = args.horizon,
                                    df=Y_df, 
                                    step_size=1, 
                                    n_windows=2685,
                                    refit=False
                                    )

    fcst_df.to_csv(results_dir+f'/forecasts.csv', index=False)
    print('Time: ', time.time() - start)

def parse_args():
    desc = "Example of hyperparameter tuning"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--results_dir', type=str, help='results_dir')
    parser.add_argument('--horizon', type=int, help='forecast horizon')
    parser.add_argument('--input_size', type=int, help='input size')
    parser.add_argument('--experiment_id', default=None, required=False, type=str, help='string to identify experiment')
    
    return parser.parse_args()
    
if __name__ == '__main__':

    args = parse_args()
    if args is None:
        exit()

    datasets = ['ohiot1dm', 
                'simglucose'] 
    
    for dataset in datasets:
        args.dataset = dataset
        main(args)
