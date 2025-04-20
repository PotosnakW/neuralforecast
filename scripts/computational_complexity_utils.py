import pandas as pd
import numpy as np
import torch
import os
import psutil

from torch.utils.flop_counter import FlopCounterMode
import time
from neuralforecast.models import NHITS, NHITS_TREAT, NBEATSx, NBEATSx_TREAT, TFT
from ray.tune.search.hyperopt import HyperOptSearch
from neuralforecast.losses.pytorch import HuberLoss
from neuralforecast.core import NeuralForecast


def get_flops(model, inp, with_backward=False, display=False):
    istrain = model.training
    model.eval()

    flop_counter = FlopCounterMode(mods=model, display=display, depth=None)
    with flop_counter:
        if with_backward:
            model(inp).sum().backward()
        else:
            model(inp)
    total_flops =  flop_counter.get_total_flops()
    if istrain:
        model.train()
    return total_flops

def get_inp(dataset_name, model_name):

    if 'exog' in dataset_name:
        if dataset_name == 'ohiot1dm_exog':
            df = pd.read_csv('../datasets/ohiot1dm_static.csv')
        elif dataset_name=='simglucose_exog':
            df = pd.read_csv('../datasets/simglucose_static.csv')
        v = torch.tensor(df.iloc[:4, 1:].values, dtype=torch.float)

        inp = {'insample_y':torch.randn(4, 120),
                   'insample_mask':torch.ones(4, 120),
                    'futr_exog': None, #torch.randn(4, 256, 1),
                    'hist_exog': torch.randn(4, 120, 3),
                    'stat_exog': v,
                    'batch_idx': torch.zeros(4, 1),
          }

    elif ('exog' not in dataset_name)&(model_name!='autotft'):
        if dataset_name=='ohiot1dm':
            df = pd.read_csv('../datasets/ohiot1dm_static.csv')
        elif dataset_name=='simglucose':
            df = pd.read_csv('../datasets/simglucose_static.csv')
        v = torch.tensor(df.iloc[:4, 1:].values, dtype=torch.float)

        inp = {'insample_y':torch.randn(4, 120),
                   'insample_mask':torch.ones(4, 120),
                    'futr_exog': [],
                    'hist_exog': [],
                    'stat_exog': v,
          }

    elif ('exog' not in dataset_name)&(model_name=='autotft'):
        if dataset_name=='ohiot1dm':
            df = pd.read_csv('../datasets/ohiot1dm_static.csv')
        elif dataset_name=='simglucose':
            df = pd.read_csv('../datasets/simglucose_static.csv')
        v = torch.tensor(df.iloc[:4, 1:].values, dtype=torch.float)

        inp = {'insample_y':torch.randn(4, 120),
                   'insample_mask':torch.ones(4, 120),
                    'futr_exog': None,
                    'hist_exog': None,
                    'stat_exog': v,
                    'batch_idx': torch.zeros(4, 1),
          }

    return inp

def count_parameters(model):
    total_params = 0
    for p in model.parameters():
        if p.requires_grad:
            total_params += p.numel()
    return total_params

def get_inference_time(model, inp, num_runs=100):
    # Ensure model is in evaluation mode
    model.eval()
    num_runs = 100
    
    times = []
    for run in range(num_runs):
        with torch.no_grad():  # Disable gradients for faster execution
            start_time = time.perf_counter()
            model(inp)  # Single forward pass
            end_time = time.perf_counter()
        
            times.append((end_time - start_time) * 1000)
    
    # Compute average inference time
    avg_time = np.mean(np.array(times))

    return avg_time

# Function to get the current memory usage in MB
def get_cpu_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 ** 2)  # Convert to MB

def get_mem(model, inp):
    # Create the model and move it to the CPU
    device = torch.device("cpu")
    model = model.to(device)
    
    # Track memory usage before inference
    initial_memory = get_cpu_memory_usage()
    #print(f"Initial CPU Memory Usage (MB): {initial_memory}")
    output = model(inp)
    final_memory = get_cpu_memory_usage()
    #print(f"Final CPU Memory Usage (MB): {final_memory}")
    #print(f"Memory used for inference (MB): {final_memory - initial_memory}")

    return final_memory - initial_memory



def get_model(model_name, dataset_name, hparams):

    if model_name == 'autonhits':
        model = NHITS(h = 6,
                    input_size = 120,
                    loss = HuberLoss(),
                    valid_loss= HuberLoss(),
                    learning_rate = hparams['learning_rate'],
                    max_steps = 2000,
                    val_check_steps = 100,
                    batch_size = 4,
                    valid_batch_size = None,
                    windows_batch_size = 256,
                    inference_windows_batch_size = -1,
                    step_size = 1,
                    num_lr_decays = 3,
                    early_stop_patience_steps = 5,
                    scaler_type = hparams['scaler_type'],
                    stat_exog_list = hparams['stat_exog_list'],
                    hist_exog_list = hparams['hist_exog_list'],
                    futr_exog_list = hparams['futr_exog_list'],
                    rrandom_seed = hparams['random_seed'],
                    alias = hparams['alias'],
                    stack_types = ['identity', 'identity', 'identity'],
                    n_blocks = [1, 1, 1],
                    mlp_units = [[1024, 1024], [1024, 1024], [1024, 1024]],
                    n_pool_kernel_size = [1, 1, 1],
                    n_freq_downsample = [1, 1, 1],
                    dropout_prob_theta = 0.0,
                     )
    
    elif model_name == 'autonhitstreat':
        model = NHITS_TREAT(h = 6,
                    input_size = 120,
                    loss = HuberLoss(),
                    valid_loss= HuberLoss(),
                    learning_rate = hparams['learning_rate'],
                    max_steps = 2000,
                    val_check_steps = 100,
                    batch_size = 4,
                    valid_batch_size = None,
                    windows_batch_size = 256,
                    inference_windows_batch_size = -1,
                    step_size = 1,
                    num_lr_decays = 3,
                    early_stop_patience_steps = 5,
                    scaler_type = hparams['scaler_type'],
                    stat_exog_list = hparams['stat_exog_list'],
                    hist_exog_list = hparams['hist_exog_list'],
                    futr_exog_list = hparams['futr_exog_list'],
                    random_seed = hparams['random_seed'],
                    alias = hparams['alias'],
                    stack_types = hparams['stack_types'],
                    n_blocks = [1, 1, 1],
                    mlp_units = [[1024, 1024], [1024, 1024], [1024, 1024]],
                    n_pool_kernel_size = [1, 1, 1],
                    n_freq_downsample = [1, 1, 1],
                    dropout_prob_theta = 0.0,
                    concentrator_type = hparams['concentrator_type'],
                    n_series = hparams['n_series'],
                    init_ka1 = 1.5,
                    init_ka2 = 1.5,
                    init_ka3 = 1.5,
                    freq = 5
                     )
        
    elif model_name == 'autonbeatsx':
        model = NBEATSx(h = 6,
                    input_size = 120,
                    loss = HuberLoss(),
                    valid_loss= HuberLoss(),
                    learning_rate = hparams['learning_rate'],
                    max_steps = 2000,
                    val_check_steps = 100,
                    batch_size = 4,
                    valid_batch_size = None,
                    windows_batch_size = 256,
                    inference_windows_batch_size = -1,
                    step_size = 1,
                    num_lr_decays = 3,
                    early_stop_patience_steps = 5,
                    scaler_type = hparams['scaler_type'],
                    stat_exog_list = hparams['stat_exog_list'],
                    hist_exog_list = hparams['hist_exog_list'],
                    futr_exog_list = hparams['futr_exog_list'],
                    random_seed = hparams['random_seed'],
                    alias = hparams['alias'],
                    n_harmonics = 2,
                    n_polynomials = 2,
                    stack_types = ['identity', 'trend', 'seasonality'],
                    n_blocks = [1, 1, 1],
                    mlp_units = [[1024, 1024], [1024, 1024], [1024, 1024]],
                    dropout_prob_theta = 0.0,
                     )

    elif model_name == 'autonbeatsxtreatcts':
        model = NBEATSx_TREAT(h = 6,
                    input_size = 120,
                    loss = HuberLoss(),
                    valid_loss= HuberLoss(),
                    learning_rate = hparams['learning_rate'],
                    max_steps = 2000,
                    val_check_steps = 100,
                    batch_size = 4,
                    valid_batch_size = None,
                    windows_batch_size = 256,
                    inference_windows_batch_size = -1,
                    step_size = 1,
                    num_lr_decays = 3,
                    early_stop_patience_steps = 5,
                    scaler_type = hparams['scaler_type'],
                    stat_exog_list = hparams['stat_exog_list'],
                    hist_exog_list = hparams['hist_exog_list'],
                    futr_exog_list = hparams['futr_exog_list'],
                    random_seed = hparams['random_seed'],
                    alias = hparams['alias'],
                    n_harmonics = 2,
                    n_polynomials = 2,
                    stack_types = ['concentrator', 'trend', 'seasonality'],
                    n_blocks = [1, 1, 1],
                    mlp_units = [[1024, 1024], [1024, 1024], [1024, 1024]],
                    dropout_prob_theta = 0.0,
                    concentrator_type = hparams['concentrator_type'],
                    n_series = hparams['n_series'],
                    init_ka1 = 1.5,
                    init_ka2 = 1.5,
                    init_ka3 = 1.5,
                    freq = 5
                     )

    elif model_name == 'autotft':
        if 'init_ka3' not in hparams.keys():
            print('no init_ka3')
            model = TFT(h = 6,
                    input_size = 120,
                    loss = HuberLoss(),
                    valid_loss= HuberLoss(),
                    learning_rate = hparams['learning_rate'],
                    max_steps = 2000,
                    val_check_steps = 100,
                    batch_size = 4,
                    valid_batch_size = None,
                    windows_batch_size = 256,
                    inference_windows_batch_size = -1,
                    step_size = 1,
                    num_lr_decays = 3,
                    early_stop_patience_steps = 5,
                    scaler_type = hparams['scaler_type'],
                    stat_exog_list = hparams['stat_exog_list'],
                    hist_exog_list = hparams['hist_exog_list'],
                    futr_exog_list = hparams['futr_exog_list'],
                    random_seed = hparams['random_seed'],
                    alias = hparams['alias'],
                    hidden_size = hparams['hidden_size'],
                    n_head = hparams['n_head'],
                    attn_dropout = 0.0,
                    dropout = 0.0,
                    tgt_size = hparams['tgt_size'],
                    use_concentrator = hparams['use_concentrator'],
                    concentrator_type = hparams['concentrator_type'],
                    n_series = hparams['n_series'],
                    init_ka1 = hparams['init_ka1'],
                    init_ka2 = hparams['init_ka1'],
                    #init_ka3 = hparams['init_ka3'],
                    freq = hparams['freq'],
                     )
        else:
            model = TFT(h = 6,
                    input_size = 120,
                    loss = HuberLoss(),
                    valid_loss= HuberLoss(),
                    learning_rate = hparams['learning_rate'],
                    max_steps = 2000,
                    val_check_steps = 100,
                    batch_size = 4,
                    valid_batch_size = None,
                    windows_batch_size = 256,
                    inference_windows_batch_size = -1,
                    step_size = 1,
                    num_lr_decays = 3,
                    early_stop_patience_steps = 5,
                    scaler_type = hparams['scaler_type'],
                    stat_exog_list = hparams['stat_exog_list'],
                    hist_exog_list = hparams['hist_exog_list'],
                    futr_exog_list = hparams['futr_exog_list'],
                    random_seed = hparams['random_seed'],
                    alias = hparams['alias'],
                    hidden_size = hparams['hidden_size'],
                    n_head = hparams['n_head'],
                    attn_dropout = 0.0,
                    dropout = 0.0,
                    tgt_size = hparams['tgt_size'],
                    use_concentrator = hparams['use_concentrator'],
                    concentrator_type = hparams['concentrator_type'],
                    n_series = hparams['n_series'],
                    init_ka1 = hparams['init_ka1'],
                    init_ka2 = hparams['init_ka1'],
                    init_ka3 = hparams['init_ka3'],
                    freq = hparams['freq'],
                     )

    return model 

