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
        data_dir = './data/ohiot1dm_exog_9_day_test.csv'
        static_dir = './data/ohiot1dm_static.csv'
        val_size = 2691
        test_size = 2691
        freq = '5min'
        horizons = [6]
        input_sizes = [120] 
        exog['stat_exog_list'] = ['559', '563', '570', '575', '588', 
                                  '591', '540', '544', '552', '567',
                                  '584', 'insulin_type_novalog', 'female',
                                  'age_20_40', 'age_40_60', 'pump_model_630G']
        exog['hist_exog_list'] = None
        exog['futr_exog_list'] = None 
            
    if args.dataset == 'simglucose':
        data_dir = './data/simglucose_exog_9_day_test.csv'
        static_dir = './data/simglucose_static.csv'
        val_size = 2592
        test_size = 2592
        freq = '5min'
        horizons = [6] 
        input_sizes = 120
        exog['stat_exog_list'] = ['adolescent#001', 'adolescent#002', 'adolescent#003', 'adolescent#004', 'adolescent#005',
                                  'adolescent#006', 'adolescent#007', 'adolescent#008', 'adolescent#009', 'adolescent#010', 
                                  'adult#001', 'adult#002', 'adult#003', 'adult#004', 'adult#005',
                                  'adult#006', 'adult#007', 'adult#008', 'adult#009', 'adult#010',
                                  'child#001', 'child#002', 'child#003', 'child#004', 'child#005',
                                  'child#006', 'child#007', 'child#008', 'child#009', 
                                  'Age', 'BW', 'adolescent', 'adult']
        exog['hist_exog_list'] = None
        exog['futr_exog_list'] = None

    return data_dir, static_dir, val_size, test_size, freq, horizons, input_sizes, exog


def main(args):

    #----------------------------------------------- Load Data -----------------------------------------------#
    data_dir, static_dir, val_size, test_size, freq, horizons, input_size, exog = get_data_parameters(args)
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
    for horizon in horizons:
        args.horizon = horizon
        print(50*'-', dataset, 50*'-')
        print(50*'-', horizon, 50*'-')
        start = time.time()
        
        output_dir = args.results_dir+f'/{args.dataset}_{args.horizon}/'
        os.makedirs(output_dir, exist_ok = True)
        
        fcst = StatsForecast(freq = freq,
                             models = [AutoETS(season_length = int(pd.Timedelta('1D')/pd.Timedelta(args.freq)),
                                                )]
                            )

        fcst_df = fcst.cross_validation(h = args.horizon,
                                        df=Y_df, 
                                        step_size=1, 
                                        n_windows=2685,
                                        refit=False
                                       )

        filename = output_dir+'/stats_model.pkl'
        pickle.dump(fcst, open(filename, 'wb'))

        fcst_df.to_csv(output_dir+f'/forecasts_{args.experiment_id}.csv', index=False)
        print('Time: ', time.time() - start)

def parse_args():
    desc = "Example of hyperparameter tuning"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--experiment_id', default=None, required=False, type=str, help='string to identify experiment')
    
    return parser.parse_args()
    
if __name__ == '__main__':

    args = parse_args()
    if args is None:
        exit()

    datasets = [#'ohiot1dm', 
                'simglucose'] 

    args.results_dir = '../results/ETS/'
    
    for dataset in datasets:
        args.dataset = dataset
        main(args)
