from ray import tune
import torch
import re

def get_nhits_experiment_space(args):
    space = {'max_steps': tune.choice([2000]),
                'val_check_steps': 100,
                'input_size': 120,
                'input_size': args.input_size,
                'stat_exog_list': tune.choice([args.exog['stat_exog_list']]),
                'hist_exog_list': tune.choice([args.exog['hist_exog_list']]),
                'futr_exog_list': tune.choice([args.exog['futr_exog_list']]),
                'stack_types': ['identity', 'identity', 'identity'],
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
            }
    return space

def get_mlp_experiment_space(args):
    space = {'input_size': args.input_size,
            'max_steps': tune.choice([2000]),
            'val_check_steps': 100,
            'stat_exog_list': tune.choice([args.exog['stat_exog_list']]),
            'hist_exog_list': tune.choice([args.exog['hist_exog_list']]),
            'futr_exog_list': tune.choice([args.exog['futr_exog_list']]),
            'num_layers': tune.choice([3]),
            'hidden_size': tune.choice([128, 256]),
            'learning_rate': tune.loguniform(1e-4, 1e-1),
            'early_stop_patience_steps': 5,
            'batch_size': 4,
            'windows_batch_size': tune.choice([256]),
            'scaler_type': 'minmax_treatment',
            'random_seed': tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            }
    
    return space

def get_rnn_experiment_space(args):
    space = {'input_size': args.input_size,
            'max_steps': tune.choice([2000]),
            'val_check_steps': 100,
            'stat_exog_list': tune.choice([args.exog['stat_exog_list']]),
            'hist_exog_list': tune.choice([args.exog['hist_exog_list']]),
            'futr_exog_list': tune.choice([args.exog['futr_exog_list']]),
            'encoder_n_layers': tune.choice([3]),
            'encoder_hidden_size': tune.choice([128, 256]),
            'encoder_activation': tune.choice(['relu']),
            'encoder_bias': tune.choice([True]),
            'encoder_dropout': 0.0,
            'context_size': tune.choice([10]),
            'decoder_hidden_size': tune.choice([128, 256]),
            'decoder_layers': tune.choice([3]),
            'learning_rate': tune.loguniform(1e-4, 1e-1),
            'early_stop_patience_steps': 5,
            'batch_size': 4,
            'scaler_type': 'minmax_treatment',
            'random_seed': tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            }

    return space
    
def get_lstm_experiment_space(args):
    space = {'input_size': args.input_size,
            'max_steps': tune.choice([2000]),
            'val_check_steps': 100,
            'stat_exog_list': tune.choice([args.exog['stat_exog_list']]),
            'hist_exog_list': tune.choice([args.exog['hist_exog_list']]),
            'futr_exog_list': tune.choice([args.exog['futr_exog_list']]),
            'encoder_n_layers': tune.choice([3]),
            'encoder_hidden_size': tune.choice([128, 256]),
            'encoder_bias': tune.choice([True]),
            'encoder_dropout': 0.0,
            'context_size': tune.choice([10]),
            'decoder_hidden_size': tune.choice([128, 256]),
            'decoder_layers': tune.choice([3]),
            'learning_rate': tune.loguniform(1e-4, 1e-1),
            'early_stop_patience_steps': 5,
            'batch_size': 4,
            'scaler_type': 'minmax_treatment',
            'random_seed': tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            }

    return space
    
def get_tcn_experiment_space(args):
    space = {'input_size': args.input_size,
            'max_steps': tune.choice([2000]),
            'val_check_steps': 100,
            'stat_exog_list': tune.choice([args.exog['stat_exog_list']]),
            'hist_exog_list': tune.choice([args.exog['hist_exog_list']]),
            'futr_exog_list': tune.choice([args.exog['futr_exog_list']]),
            'kernel_size': tune.choice([3]),
            'dilations': tune.choice([[2, 4, 8]]),
            'encoder_hidden_size': tune.choice([128, 256]),
            'encoder_activation': tune.choice(['ReLU']),
            'context_size': tune.choice([10]),
            'decoder_hidden_size': tune.choice([128, 256]),
            'decoder_layers': tune.choice([3]),
            'learning_rate': tune.loguniform(1e-4, 1e-1),
            'early_stop_patience_steps': 5,
            'batch_size': 4,
            'scaler_type': 'minmax_treatment',
            'random_seed': tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            }

    return space

