from ray import tune
import torch
import re

def get_nhits_treat_experiment_space(args):
    space= {'max_steps': tune.choice([2000]),  
             'val_check_steps': 100,
                'input_size': args.input_size,
                'stat_exog_list': tune.choice([args.exog['stat_exog_list']]),
                'hist_exog_list': tune.choice([args.exog['hist_exog_list']]),
                'futr_exog_list': tune.choice([args.exog['futr_exog_list']]),
                'stack_types': ['identity', 'identity', 'concentrator'],
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
                # Treat parameters:
                'n_series': args.n_series,
                'concentrator_type': 'sum_total',
                #'treatment_var_name': tune.choice(['basal_insulin']),
                'freq': int(re.sub("[^0-9]","", args.freq))
                }
            
    return space
