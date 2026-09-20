import torch
from torch.utils.flop_counter import FlopCounterMode
import numpy as np
import time
from copy import deepcopy
import pandas as pd

from torch.optim.lr_scheduler import StepLR
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import MOMENT, PatchTSTMultivariate, Crossformer, iTransformer, iTransformerT5, TimerXL, MLPMultivariate, TSMixer, TimeMixer, Chronos2


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


def analyze_model(model, inp, with_backward=False, display=False, num_runs=100, warmup_runs=10):
    """
    Complete model analysis: FLOPs, size metrics, and inference time
    
    Args:
        model: PyTorch model
        inp: Input tensor or tuple of tensors
        with_backward: If True, include backward pass FLOPs
        display: If True, print detailed FLOP breakdown
        num_runs: Number of inference runs to average over
        warmup_runs: Number of warmup runs for timing
        
    Returns:
        dict with comprehensive model statistics
    """

    trainable_params = get_model_size(model)
    flops = get_flops(model, inp, with_backward=with_backward, display=display)
    ms = get_inference_time(model, inp, num_runs=num_runs, warmup_runs=warmup_runs)

    results = {
        'gflops': flops,
        'trainable_params': trainable_params,
        'inference_speed': ms,
    }
    
    return results

def get_table(models, inp):
    model_params = {}
    device = next(iter(inp.values())).device

    for model in models.items():
        print(model[0])
        # One model failing must not cost the whole table. This bites in the
        # context-length ablation: at input_size < h some architectures cannot
        # form a window at all, and without this the entire run is lost.
        try:
            model[1].to(device)  # Move model to GPU
            results = analyze_model(model[1], inp=inp, with_backward=False)
        except Exception as e:
            print(f"  [table] {model[0]}: SKIPPED ({type(e).__name__}: {e})")
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            continue

        if model[0] == 'Chronos-2':
            results['trainable_params'] = 120.0

        model_params[model[0]] = results

        # Clear GPU cache between models
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    model_params_df = pd.DataFrame(model_params).T
    model_params_df = model_params_df.round(3)
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

def get_model_config(args, model_type='moment'):
    """
    Create configuration dict for different model types.
    
    Args:
        args: Arguments object with input_size, n_series, windows_batch_size
        model_type: One of ['moment', 'patchtst', 'itransformer', 'itransformert5', 
                    'crossformer', 'timerxl', 'tsmixer', 'mlpmultivariate']
    
    Returns:
        Configuration dictionary for the specified model type
    """
    
    # Common training parameters
    common_config = {
        'hist_exog_list': None,
        'futr_exog_list': None,
        'stat_exog_list': None,
        'input_size': args.input_size,
        'n_series': args.n_series,
        'max_steps': 12000,
        'val_check_steps': 500,
        'early_stop_patience_steps': 20,
        'random_seed': 1,
        'learning_rate': 1e-3,
        'batch_size': args.n_series,
        'windows_batch_size': args.windows_batch_size,
        'inference_windows_batch_size': args.windows_batch_size,
        'scaler_type': 'standard',
        'loss': MAE(),
        'lr_scheduler': StepLR,
        'lr_scheduler_kwargs': {
            'step_size': 4000,
            'gamma': 0.5
        },
    }
    
    # Model-specific configurations
    if model_type in ['moment', 'patchtst']:
        config = {
            **common_config,
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
            # MICA/Infini
            'infini_mixer_type': 'none',
            'infini_channel_weight_type': 'uniform',
            'infini_channel_exclusion': False,
            'layerwise_beta': True,
            'channelwise_beta': False,
            'mlpmixer_hidden_size': 128,
            'mlpmixer_n_layers': 2,
            'mlpmixer_dropout': 0.0,
        }
        
    elif model_type in ['itransformer', 'itransformert5']:
        config = {
            **common_config,
            # Architecture
            'hidden_size': 256,
            'linear_hidden_size': 1024,
            'n_heads': 4,
            'n_layers': 4,
            'd_k': 32,
            'd_v': 32,
            # Regularization
            'dropout': 0.0,
            'head_dropout': 0.0,
            # RevIN
            'revin': True,
            'revin_affine': False,
            'revin_subtract_last': False,
            # Multivariate
            'multivariate_head': False,
            'univariate': False,  # Set to False for multivariate mode
        }
        
    elif model_type == 'crossformer':
        config = {
            **common_config,
            # Architecture
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
            # Regularization
            'dropout': 0.0,
            'head_dropout': 0.0,
            # RevIN
            'revin': True,
            'revin_affine': False,
            'revin_subtract_last': False,
            # Multivariate
            'univariate': False,  # Set to False for multivariate mode
        }
        
    elif model_type == 'timerxl':
        config = {
            **common_config,
            # Architecture
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
            # Regularization
            'dropout': 0.0,
            'head_dropout': 0.0,
            # RevIN
            'revin': True,
            'revin_affine': False,
            'revin_subtract_last': False,
            # Head
            'multivariate_head': False,
            # Multivariate
            'univariate': False,  # Set to False for multivariate mode
        }
        
    elif model_type == 'tsmixer':
        config = {
            **common_config,
            # Architecture
            'n_block': 4,  # Number of mixer blocks (analogous to n_layers)
            'ff_dim': 256,  # Feed-forward dimension (analogous to hidden_size)
            # Regularization
            'dropout': 0.0,
            # RevIN
            'revin': True,
            'revin_affine': False,
            'revin_subtract_last': False,
            # Multivariate
            'univariate': False,  # Set to False for multivariate mode
        }

    elif model_type == 'timemixer':
        config = {
            **common_config,
            # Architecture
            'd_model': 256,
            'd_ff': 1024,
            'e_layers': 4,
            'decomp_method': 'moving_avg',
            'down_sampling_method': 'avg',
            'down_sampling_layers': 1,
            'down_sampling_window': 2, 
            'moving_avg': 7,
            'top_k': 5,
            'dropout': 0.0,
            # RevIN
            'revin': True,
            'revin_affine': False,
            'revin_subtract_last': False,
            # Multivariate
            'channel_independence': 0, # channel-dependence
        }
        
    elif model_type == 'mlpmultivariate':
        config = {
            **common_config,
            # Architecture
            'hidden_size': 256,
            'num_layers': 4,  # Number of MLP layers
            # Multivariate
            'univariate': False,  # Set to False for multivariate mode
        }
    
    elif model_type == 'chronos2':
        config = {
            **common_config,
            # Architecture
            "top_k": 1,              # Always pick most likely value
            "top_p": 1.0,            # Doesn't matter when top_k=1
            'univariate': False,  # Set to False for multivariate mode
        }
        
    elif model_type in ['toto2', 'timesfm3']:
        # Zero-shot foundation models. They take no architecture hyperparameters
        # here -- the checkpoint fixes the architecture -- and they are frozen, so
        # the training keys in common_config are inert. `checkpoint` is left at the
        # wrapper default (Toto-2.0-22m / timesfm-3.0-pytorch); override it here to
        # run the larger-checkpoint ablation.
        config = {
            'input_size': args.input_size,
            'n_series': args.n_series,
            'windows_batch_size': args.windows_batch_size,
            'inference_windows_batch_size': args.windows_batch_size,
            'random_seed': 1,
            'loss': MAE(),
        }

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from: "
                        f"['moment', 'patchtst', 'itransformer', 'itransformert5', "
                        f"'crossformer', 'timerxl', 'tsmixer', 'timemixer', 'mlpmultivariate', "
                        f"'chronos2', 'toto2', 'timesfm3']")
    
    return config


# Table configs are selected on the command line rather than by editing this
# file, so the C=7 / C=600 / context-length-ablation tables can be produced by a
# sweep instead of by hand-uncommenting a block.
#   n_series counts used before: [7, 15, 50, 100, 150, 200, 300, 400, 500, 600]
#   input_size (context) ablation: [64, 16, 32, 64, 96, 128, 256, 312, 512, 8192]
import argparse

_p = argparse.ArgumentParser(description="FLOPs / params / inference-time baseline table")
_p.add_argument('--n_series', type=int, default=600)
_p.add_argument('--h', type=int, default=48)
_p.add_argument('--input_size', type=int, default=96)
_p.add_argument('--windows_batch_size', type=int, default=1)
# Was hardcoded to cuda:2, which fails on any allocation with fewer than 3 GPUs
# (e.g. a --gres=gpu:1 slurm job).
_p.add_argument('--device', type=str, default='cuda:0')
# Results go to scratch, not the repo in $HOME (491G, 78% full) -- same
# convention as the sweep's exp_results/logs.
_p.add_argument('--out_dir', type=str,
                default='/zfsauton/scratch/wpotosna/neuralforecast_mica/table_results')
_p.add_argument('--tag', type=str, default='',
                help='suffix for the output CSV; use it to distinguish per-env runs')
_p.add_argument('--only', type=str, default='',
                help="comma-separated substrings; only matching models are built "
                     "(e.g. --only 'Toto-2' in envs/toto2)")
_p.add_argument('--no_save', action='store_true')
args = _p.parse_args()

# Each model is built lazily through a factory, for two reasons:
#   - the zero-shot models only import in their own conda env;
#   - envs/toto2 ships a newer `transformers` that cannot construct MICA's
#     T5/MOMENT models at all, so eager instantiation there kills the whole run.
# A model that cannot be built in the current env is reported and skipped, so
# each env contributes the rows it can and the CSVs are merged afterwards.

def _zeroshot_factory(cls_name, checkpoint=None):
    # checkpoint=None leaves the wrapper default (Toto-2.0-22m /
    # timesfm-3.0-pytorch); pass one to size-ablate the same architecture.
    def make():
        import sys, os
        _training = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'training')
        if _training not in sys.path:
            sys.path.insert(0, _training)
        import zeroshot_external_models as _zsm
        model_type = 'toto2' if 'Toto' in cls_name else 'timesfm3'
        cfg = get_model_config(args, model_type=model_type)
        if checkpoint is not None:
            cfg['checkpoint'] = checkpoint
        return getattr(_zsm, cls_name)(h=args.h, **cfg)
    return make


def _mica_factories():
    config_mp = get_model_config(args, model_type='moment')
    config_mp_infini = deepcopy(config_mp)
    config_mp_infini['infini_mixer_type'] = 'mlp_query'
    config_mp_infini['layerwise_beta'] = False
    config_mp_infini['channelwise_beta'] = False
    return {
        'PatchTST MICA (MLP w/ Query)': lambda: PatchTSTMultivariate(h=args.h, **config_mp_infini),
        'Moment MICA (MLP w/ Query)':   lambda: MOMENT(h=args.h, **config_mp_infini),
        'iTransformer':     lambda: iTransformer(h=args.h, **get_model_config(args, model_type='itransformer')),
        'iTransformer-T5':  lambda: iTransformerT5(h=args.h, **get_model_config(args, model_type='itransformer')),
        'Crossformer':      lambda: Crossformer(h=args.h, **get_model_config(args, model_type='crossformer')),
        'Timer-XL':         lambda: TimerXL(h=args.h, **get_model_config(args, model_type='timerxl')),
        'TSMixer':          lambda: TSMixer(h=args.h, **get_model_config(args, model_type='tsmixer')),
        'TimeMixer':        lambda: TimeMixer(h=args.h, **get_model_config(args, model_type='timemixer')),
        'MLP':              lambda: MLPMultivariate(h=args.h, **get_model_config(args, model_type='mlpmultivariate')),
        'Chronos-2':        lambda: Chronos2(h=args.h, **get_model_config(args, model_type='chronos2')),
    }


FACTORIES = {}
try:
    FACTORIES.update(_mica_factories())
except Exception as _e:
    print(f"  [models] MICA baselines unavailable in this env ({type(_e).__name__}: {_e})")
# Toto-2 at two sizes: 22M matches the accuracy sweep's default, 313M is
# size-matched to TimesFM-3 (331M) so the two can be compared on architecture
# rather than scale.
FACTORIES['Toto-2 (22M)'] = _zeroshot_factory('Toto2')
FACTORIES['Toto-2 (313M)'] = _zeroshot_factory('Toto2', 'Datadog/Toto-2.0-313m')
FACTORIES['TimesFM-3 (331M)'] = _zeroshot_factory('TimesFM3')

_wanted = [w.strip() for w in args.only.split(',') if w.strip()] if args.only else None

models = {}
for _label, _make in FACTORIES.items():
    if _wanted and not any(w.lower() in _label.lower() for w in _wanted):
        continue
    try:
        models[_label] = _make()
        print(f"  [models] {_label}: built")
    except Exception as _e:
        print(f"  [models] {_label}: SKIPPED ({type(_e).__name__}: {_e})")

if not models:
    raise SystemExit("No models could be built in this env -- check --only / env deps.")

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.manual_seed(1)
inp = {
    'insample_y': torch.randn(args.windows_batch_size, args.input_size, args.n_series).to(device),
    'insample_mask': torch.ones(args.windows_batch_size, args.input_size, args.n_series).to(device),
}


table = get_table(models, inp)
print(table.to_string(index=False))

if not args.no_save:
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    _tag = f'_{args.tag}' if args.tag else ''
    _path = os.path.join(
        args.out_dir,
        f'flops_baseline_table_n{args.n_series}_is{args.input_size}{_tag}.csv')
    table.to_csv(_path, index=False)
    print(f'wrote {_path}')
