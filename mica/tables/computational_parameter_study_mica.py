import torch
from torch.utils.flop_counter import FlopCounterMode
import numpy as np
import time
from copy import deepcopy
import pandas as pd

from torch.optim.lr_scheduler import StepLR
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import MOMENT, PatchTSTMultivariate


def get_model_size(model):
    """
    Calculate model size in parameters and MB
    
    Args:
        model: PyTorch model
        
    Returns:
        dict with parameter counts and memory sizes
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    num_params = sum(p.numel() for p in model.parameters())
    num_params_millions = num_params / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params_millions = trainable_params / 1e6

    return trainable_params_millions


def get_flops(model, inp, with_backward=False, display=False):
    """
    Calculate FLOPs for forward (and optionally backward) pass
    
    Args:
        model: PyTorch model
        inp: Input tensor or tuple of tensors
        with_backward: If True, count backward pass FLOPs too
        display: If True, print detailed FLOP breakdown
        
    Returns:
        Total FLOPs (int)
    """
    model.eval()
    flop_counter = FlopCounterMode(mods=model, display=display, depth=None)
    with flop_counter:
        if with_backward:
            # Forward + backward
            output = model(inp)
            if isinstance(output, tuple):
                output = output[0]
            output.sum().backward()
        else:
            # Forward only
            model(inp)
    
    total_flops = flop_counter.get_total_flops()
    total_gflops = total_flops / 1e9
    
    return total_gflops

def get_inference_time(model, inp, num_runs=100, warmup_runs=10, device=None):
    """
    Measure inference time with proper warmup and averaging (optimized version)
    
    Args:
        model: PyTorch model
        inp: Input tensor or tuple of tensors
        num_runs: Number of inference runs to average over
        warmup_runs: Number of warmup runs (not counted in timing)
        device: Device to run on (None = use input device)
        
    Returns:
        Mean inference time in milliseconds
    """
    model.eval()
    
    # Determine device
    if device is None:
        if isinstance(inp, torch.Tensor):
            device = inp.device
        else:
            device = inp[0].device if isinstance(inp, (list, tuple)) else torch.device('cpu')
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(inp)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    # Timed runs - use CUDA events for GPU
    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_event.record()
                _ = model(inp)
                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))  # Already in ms
        
        return sum(times) / len(times)
    
    else:
        # CPU timing
        with torch.no_grad():
            start_time = time.perf_counter()
            for _ in range(num_runs):
                _ = model(inp)
            end_time = time.perf_counter()
        
        return ((end_time - start_time) / num_runs) * 1000  # Convert to ms


def analyze_model(model, inp, with_backward=False, display=False, 
                  measure_inference=True, num_runs=100, warmup_runs=10):
    """
    Complete model analysis: FLOPs, size metrics, and inference time
    
    Args:
        model: PyTorch model
        inp: Input tensor or tuple of tensors
        with_backward: If True, include backward pass FLOPs
        display: If True, print detailed FLOP breakdown
        measure_inference: If True, measure inference time
        num_runs: Number of inference runs to average over
        warmup_runs: Number of warmup runs for timing
        
    Returns:
        dict with comprehensive model statistics
    """

    trainable_params = get_model_size(model)
    flops = get_flops(model, inp, with_backward=with_backward, display=display)
    if measure_inference:
        ms = get_inference_time(model, inp, num_runs=num_runs, warmup_runs=warmup_runs)

    results = {
        'gflops': flops,
       # 'total_params': num_params,
        'trainable_params': trainable_params,
        'inference_speed': ms,
    }
    
    return results

def get_table(models, inp):
    model_params = {}
    device = next(iter(inp.values())).device

    for model in models.items():
        print(model[0])
        model[1].to(device)  # Move model to GPU
        results = analyze_model(model[1], inp=inp, with_backward=False)
        model_params[model[0]] = results

        # Clear GPU cache between models
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    model_params_df = pd.DataFrame(model_params).T
    model_params_df = model_params_df.round(3)
    # for col in ['gflops', 'trainable_params', 'inference_speed']:
    #     model_params_df[col] = format_with_increase(model_params_df[col])
    model_params_df.index.name = 'model' 
    model_params_df.reset_index(inplace=True, drop=False)

    model = [i.split(' ')[0] for i in model_params_df['model'].values]
    variant = [(' ').join(i.split(' ')[1:]) for i in model_params_df['model'].values]
    model_params_df['model'] = model
    model_params_df['variant'] = variant
    model_params_df = model_params_df[['model', 'variant', 'gflops', 'trainable_params', 'inference_speed']] 

    # Merge duplicate model names - only show model name on first occurrence
    prev_dataset = None
    for idx in range(len(model_params_df)):
        current_dataset = model_params_df.at[idx, 'model']
        if current_dataset == prev_dataset:
            model_params_df.at[idx, 'model'] = ''
        else:
            prev_dataset = current_dataset

    return model_params_df

def get_model_config(args):
    """
    Create configuration dict that overrides MOMENT's __init__ defaults.
    Any matching parameter names will replace the defaults.
    
    Usage:
        config = get_model_config(args)
        model = MOMENT(h=forecast_horizon, **config)
    """
    config = {
        # From args
        'input_size': args.input_size,
        'n_series': args.n_series,
        
        # Training
        'max_steps': 12000,
        'val_check_steps': 500,
        'early_stop_patience_steps': 20,
        'random_seed': 1,
        'learning_rate': 1e-3,
        'batch_size': args.n_series,
        'windows_batch_size': args.windows_batch_size, #64
        'inference_windows_batch_size': args.windows_batch_size, #64
        
        # Architecture
        'transformer_backbone': "google/t5-efficient-tiny",
        'hidden_size': 256,
        'linear_hidden_size': 1024,
        'n_heads': 4,
        'n_layers': 4,
        'd_k': 32,
        'd_v': 32,
        
        # Patching
        'patch_len': 8,
        'stride': 8,
        'padding_patch': 'end',
        
        # Positional encoding
        'pe_type': 'sincos',
        'learn_pe': False,
        
        # RevIN
        'revin': True,
        'revin_affine': False,
        'revin_subtract_last': False,
        
        # Regularization
        'dropout': 0.0,
        'head_dropout': 0.0,
        
        # Head
        'multivariate_head': False,
        
        # Other
        'scaler_type': 'standard',
        'loss': MAE(),
        
        # LR Scheduler
        'lr_scheduler': StepLR,
        'lr_scheduler_kwargs': {
            'step_size': 4000,
            'gamma': 0.5
        },

        # Infini
        'infini_mixer_type': 'none',
        'infini_channel_exclusion': False,
        'layerwise_beta': True,
        'channelwise_beta': False,
        'mlpmixer_hidden_size': 128,
        'mlpmixer_n_layers': 2,
        'mlpmixer_dropout': 0.0,
    }
    
    return config

def format_with_increase(values, baseline_idx=0, decimals=3):
    """Format values with percentage increase relative to baseline"""
    baseline = values.iloc[baseline_idx]
    formatted = []
    for i, val in enumerate(values):
        if i == baseline_idx:
            formatted.append(f"{val:.{decimals}f}")
        else:
            pct_change = ((val - baseline) / baseline) * 100
            if pct_change >= 0:
                formatted.append(f"{val:.{decimals}f} ($\\uparrow {pct_change:.1f}\\%$)")
            else:
                formatted.append(f"{val:.{decimals}f} ($\\downarrow {abs(pct_change):.1f}\\%$)")
    return formatted

class Args:
    n_series = 7       # Weather/ETT typical
    h = 48             # Standard horizon
    input_size = 96   # Standard lookback (or 512)
    windows_batch_size = 1
args = Args()

config = get_model_config(args)

config_headmixer = deepcopy(config)
config_headmixer['multivariate_head'] = True

config_infini = deepcopy(config)
config_infini['infini_mixer_type'] = 'betas'
config_infini['layerwise_beta'] = False
config_infini['channelwise_beta'] = False

config_infini_channelwise = deepcopy(config)
config_infini_channelwise['infini_mixer_type'] = 'betas'
config_infini_channelwise['layerwise_beta'] = False
config_infini_channelwise['channelwise_beta'] = True

config_infini_layerwise = deepcopy(config)
config_infini_layerwise['infini_mixer_type'] = 'betas'
config_infini_layerwise['layerwise_beta'] = True
config_infini_layerwise['channelwise_beta'] = False

config_infini_layerwise_channelwise = deepcopy(config)
config_infini_layerwise_channelwise['infini_mixer_type'] = 'betas'
config_infini_layerwise_channelwise['layerwise_beta'] = True
config_infini_layerwise_channelwise['channelwise_beta'] = True

config_infini_mlp = deepcopy(config)
config_infini_mlp['infini_mixer_type'] = 'mlp'

config_infini_mlpquery = deepcopy(config)
config_infini_mlpquery['infini_mixer_type'] = 'mlp_query'

patchtst_vanilla = PatchTSTMultivariate(h=args.h, **config)
patchtst_headmixer = PatchTSTMultivariate(h=args.h, **config_headmixer)
patchtst_infini = PatchTSTMultivariate(h=args.h, **config_infini)
patchtst_infini_channelwise = PatchTSTMultivariate(h=args.h, **config_infini_channelwise)
patchtst_infini_layerwise = PatchTSTMultivariate(h=args.h, **config_infini_layerwise)
patchtst_infini_layerwise_channelwise = PatchTSTMultivariate(h=args.h, **config_infini_layerwise_channelwise)
patchtst_infini_mlp = PatchTSTMultivariate(h=args.h, **config_infini_mlp)
patchtst_infini_mlpquery = PatchTSTMultivariate(h=args.h, **config_infini_mlpquery)

moment_vanilla = MOMENT(h=args.h, **config)
moment_headmixer = MOMENT(h=args.h, **config_headmixer)
moment_infini = MOMENT(h=args.h, **config_infini)
moment_infini_channelwise = MOMENT(h=args.h, **config_infini_channelwise)
moment_infini_layerwise = MOMENT(h=args.h, **config_infini_layerwise)
moment_infini_layerwise_channelwise = MOMENT(h=args.h, **config_infini_layerwise_channelwise)
moment_infini_mlp = MOMENT(h=args.h, **config_infini_mlp)
moment_infini_mlpquery = MOMENT(h=args.h, **config_infini_mlpquery)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

inp = {
    'insample_y': torch.randn(args.windows_batch_size, args.input_size, args.n_series).to(device),
    'insample_mask': torch.ones(args.windows_batch_size, args.input_size, args.n_series).to(device),
}

patchtst_models = {
    'PatchTST Univariate': patchtst_vanilla,
    'PatchTST Multivariate Head': patchtst_headmixer,
    'PatchTST MICA (Shared $\\beta$)': patchtst_infini,
    'PatchTST MICA (Channelwise $\\beta$)': patchtst_infini_channelwise,
    'PatchTST MICA (Layerwise $\\beta$)': patchtst_infini_layerwise,
    'PatchTST MICA (Layerwise Channelwise $\\beta$)': patchtst_infini_layerwise_channelwise,
    'PatchTST MICA (MLP)': patchtst_infini_mlp,
    'PatchTST MICA (MLP w/ Query)': patchtst_infini_mlpquery,
}

moment_models = {
    'Moment Univariate': moment_vanilla,
    'Moment Multivariate Head': moment_headmixer,
    'Moment MICA (Shared $\\beta$)': moment_infini,
    'Moment MICA (Channelwise $\\beta$)': moment_infini_channelwise,
    'Moment MICA (Layerwise $\\beta$)': moment_infini_layerwise,
    'Moment MICA (Layerwise Channelwise $\\beta$)': moment_infini_layerwise_channelwise,
    'Moment MICA (MLP)': moment_infini_mlp,
    'Moment MICA (MLP w/ Query)': moment_infini_mlpquery,
}

patchtst_table = get_table(patchtst_models, inp)
moment_table = get_table(moment_models, inp)

final_table = pd.concat([patchtst_table, moment_table], axis=0)
final_table.to_csv('./flops_mica_table.csv', index=False)
