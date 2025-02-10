from ray import tune
import torch
import re

def get_nhits_treat_experiment_space(args):
    space = {'max_steps': tune.choice([2000]),  
              'val_check_steps': 100,
                'input_size': args.input_size,
                'stat_exog_list': tune.choice([args.exog['stat_exog_list']]),
                'hist_exog_list': tune.choice([args.exog['hist_exog_list']]),
                'futr_exog_list': tune.choice([args.exog['futr_exog_list']]),
                'stack_types': ['identity', 'concentrator', 'identity'], # concentrator should be in middle
                'mlp_units': tune.choice([ 3*[[1024,1024]] ]),
                'n_pool_kernel_size': [1,1,1],
                'n_freq_downsample': [1,1,1],
                'dropout_prob_theta': tune.choice([0.0]),
                'batch_size': 4,
                'windows_batch_size': 256,
                'scaler_type':  None,
                'early_stop_patience_steps': 5,
                'learning_rate': tune.loguniform(1e-4, 1e-2),
                'random_seed': tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                #Treat parameters:
                'n_series': args.n_series,
                'concentrator_type': 'log_normal',
                'init_ka1': tune.choice([1.7, 1.8, 1.9]),
                'init_ka2': tune.choice([1.0, 1.1, 1.2]),
                'init_ka3':tune.choice([1.7, 1.8, 1.9]),
                'freq': int(re.sub("[^0-9]","", args.freq))
           }
    return space

def get_nbeatsx_treat_experiment_space3(args):
    space = {'input_size': args.input_size,
            'max_steps': tune.choice([2000]),
            'val_check_steps': 100,
            'stat_exog_list': tune.choice([args.exog['stat_exog_list']]),
            'hist_exog_list': tune.choice([args.exog['hist_exog_list']]),
            'futr_exog_list': tune.choice([args.exog['futr_exog_list']]),
            'stack_types': ['concentrator', 'trend', 'seasonality'],
            'n_blocks': 3*[1],
            'mlp_units': tune.choice([ 3*[[1024,1024]] ]),
            'dropout_prob_theta': tune.choice([0.0]),
            'learning_rate': tune.loguniform(1e-4, 1e-1),
            'early_stop_patience_steps': 5,
            'batch_size': 4,
            'scaler_type': 'minmax_treatment',
            'random_seed': tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            #Treat parameters:
            'n_series': args.n_series,
            'concentrator_type': 'log_normal',
            'init_ka1': tune.choice([1.7, 1.8, 1.9]),
            'init_ka2': tune.choice([1.0, 1.1, 1.2]),
            'init_ka3':tune.choice([1.7, 1.8, 1.9]),
            'freq': int(re.sub("[^0-9]","", args.freq))
            }

    return space

def get_tft_treat_experiment_space(args):
    space = {'input_size': args.input_size,
            'max_steps': tune.choice([2000]),
            'val_check_steps': 100,
            'stat_exog_list': tune.choice([args.exog['stat_exog_list']]),
            'hist_exog_list': tune.choice([args.exog['hist_exog_list']]),
            'futr_exog_list': tune.choice([args.exog['futr_exog_list']]),
            'hidden_size': tune.choice([128, 256, 512]),
            'n_head': tune.choice([4, 8, 16]),
            'dropout': tune.choice([0.0]),
            'attn_dropout': tune.choice([0.0]),
            'learning_rate': tune.loguniform(1e-4, 1e-1),
            'early_stop_patience_steps': 5,
            'batch_size': 4,
            'scaler_type': 'minmax_treatment',
            'random_seed': tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            #Treat parameters:
            'use_concentrator': True,
            'n_series': args.n_series,
            'concentrator_type': 'log_normal',
            'init_ka1': tune.choice([1.7, 1.8, 1.9]),
            'init_ka2': tune.choice([1.0, 1.1, 1.2]),
            'init_ka3':tune.choice([1.7, 1.8, 1.9]),
            'freq': int(re.sub("[^0-9]","", args.freq))
            }

    return space