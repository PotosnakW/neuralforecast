import copy

from neuralforecast.auto import AutoiTransformerT5, AutoiTransformer, AutoTSMixer, AutoMLPMultivariate, AutoMOMENT, AutoPatchTSTMultivariate, AutoTimerXL, AutoCrossformer, AutoTimeMixer, AutoChronos2
from neuralforecast.losses.pytorch import MAE

import optuna

from torch.optim.lr_scheduler import StepLR


def get_models(args):

    max_steps = 12000
    val_check_steps = 500
    transformer_backbone = "google/t5-efficient-tiny" 
    hidden_size = 256
    linear_hidden_size = 1024
    n_heads = 4
    n_layers = 4
    learning_rate = 1e-3
    scaler_type = 'standard'
    early_stop_patience_steps = 20
    batch_size = args.n_series
    windows_batch_size = 64
    inference_windows_batch_size = 64
    patch_len = 8
    stride = 8
    d_k = 32
    d_v = 32
    dropout = 0.0
    head_dropout = 0.0
    pe_type = 'sincos'
    learn_pe = False
    revin = True
    revin_affine = False
    revin_subtract_last = False # Not implementated
    padding_patch = 'end'
    multivariate_head = False
    lr_scheduler=StepLR
    lr_scheduler_kwargs={
        'step_size': 4000,
        'gamma': 0.5
    }
    loss = MAE()

    if args.experiment_name == 'vanilla_t5tiny':
        vanilla_config = {
            'input_size': args.input_size,
            'n_series': args.n_series,
            'patch_len': patch_len,
            'stride': stride,
            'max_steps': max_steps,
            'val_check_steps': val_check_steps,
            'windows_batch_size': windows_batch_size,
            'inference_windows_batch_size': inference_windows_batch_size,
            #'transformer_backbone': transformer_backbone,
            'hidden_size': hidden_size,
            'linear_hidden_size': linear_hidden_size,
            'n_heads': n_heads,
            'd_k': d_k,
            'd_v': d_v,
            'n_layers': n_layers,
            'pe_type': pe_type,
            'learn_pe': learn_pe,
            'dropout': dropout,
            'head_dropout': head_dropout,
            'revin': revin,
            'revin_affine': revin_affine,
            'revin_subtract_last': revin_subtract_last,
            'padding_patch': padding_patch,
            'infini_mixer_type': 'none',
            'multivariate_head': multivariate_head,
            'learning_rate': learning_rate,
            'early_stop_patience_steps': early_stop_patience_steps,
            'batch_size': batch_size,
            'valid_batch_size': batch_size,
            'scaler_type': scaler_type,
            'lr_scheduler': lr_scheduler,
            'lr_scheduler_kwargs': lr_scheduler_kwargs,
            'random_seed': args.random_seed,
        }

        vanilla_headmixer_config = copy.deepcopy(vanilla_config)
        vanilla_headmixer_config['multivariate_head'] = True

        models = [
            AutoMOMENT(
                h=args.h,
                config=vanilla_config,
                loss=loss,
                search_alg=None,
                num_samples=args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoMOMENT_vanilla'
            ),
            AutoMOMENT(
                h=args.h,
                config=vanilla_headmixer_config,
                loss=loss,
                search_alg=None,
                num_samples=args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoMOMENT_vanilla_headmixer'
            ),
            AutoPatchTSTMultivariate(
                h=args.h,
                config=vanilla_config,
                loss=loss,
                search_alg=None,
                num_samples=args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoPatchTSTMultivariate_vanilla'
            ),
            AutoPatchTSTMultivariate(
                h=args.h,
                config=vanilla_headmixer_config,
                loss=loss,
                search_alg=None,
                num_samples=args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoPatchTSTMultivariate_vanilla_headmixer'
            ),
        ]

    elif args.experiment_name == 'vanilla_pca_t5tiny':
        vanilla_config = {
            'input_size': args.input_size,
            'n_series': args.n_series,
            'patch_len': patch_len,
            'stride': stride,
            'max_steps': max_steps,
            'val_check_steps': val_check_steps,
            'windows_batch_size': windows_batch_size,
            'inference_windows_batch_size': inference_windows_batch_size,
            #'transformer_backbone': transformer_backbone,
            'hidden_size': hidden_size,
            'linear_hidden_size': linear_hidden_size,
            'n_heads': n_heads,
            'd_k': d_k,
            'd_v': d_v,
            'n_layers': n_layers,
            'pe_type': pe_type,
            'learn_pe': learn_pe,
            'dropout': dropout,
            'head_dropout': head_dropout,
            'revin': revin,
            'revin_affine': revin_affine,
            'revin_subtract_last': revin_subtract_last,
            'padding_patch': padding_patch,
            'infini_mixer_type': 'none',
            'multivariate_head': multivariate_head,
            'learning_rate': learning_rate,
            'early_stop_patience_steps': early_stop_patience_steps,
            'batch_size': batch_size,
            'valid_batch_size': batch_size,
            'scaler_type': scaler_type,
            'lr_scheduler': lr_scheduler,
            'lr_scheduler_kwargs': lr_scheduler_kwargs,
            'random_seed': args.random_seed,
        }

        models = [
            AutoMOMENT(
                h=args.h,
                config=vanilla_config,
                loss=loss,
                search_alg=None,
                num_samples=args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoMOMENT_vanilla_pca'
            ),
            AutoPatchTSTMultivariate(
                h=args.h,
                config=vanilla_config,
                loss=loss,
                search_alg=None,
                num_samples=args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoPatchTSTMultivariate_vanilla_pca'
            ),
        ]

    elif args.experiment_name == 'infini_mlpmixer_t5tiny':
        def mlpmixer_ciexcl_config(trial):
            return {
                'input_size': args.input_size,
                'n_series': args.n_series,
                'patch_len': patch_len,
                'stride': stride,
                'max_steps': max_steps,
                'val_check_steps': val_check_steps,
                'windows_batch_size': windows_batch_size,
                'inference_windows_batch_size': inference_windows_batch_size,
                #'transformer_backbone': transformer_backbone,
                'hidden_size': hidden_size,
                'linear_hidden_size': linear_hidden_size,
                'n_heads': n_heads,
                'd_k': d_k,
                'd_v': d_v,
                'n_layers': n_layers,
                'pe_type': pe_type,
                'learn_pe': learn_pe,
                'dropout': dropout,
                'head_dropout': head_dropout,
                'revin': revin,
                'revin_affine': revin_affine,
                'revin_subtract_last': revin_subtract_last,
                'padding_patch': padding_patch,
                'infini_mixer_type': 'mlp',
                'infini_channel_exclusion': True,
                'mlpmixer_hidden_size': trial.suggest_categorical('mlpmixer_hidden_size', [128, 256, 384, 512]),
                'mlpmixer_n_layers': trial.suggest_categorical('mlpmixer_n_layers', [2, 3, 4]),
                'mlpmixer_dropout': trial.suggest_categorical('mlpmixer_dropout', [0.0, 0.1, 0.2]),
                'multivariate_head': multivariate_head,
                'learning_rate': learning_rate,
                'early_stop_patience_steps': early_stop_patience_steps,
                'batch_size': batch_size,
                'valid_batch_size': batch_size,
                'scaler_type': scaler_type,
                'lr_scheduler': lr_scheduler,
                'lr_scheduler_kwargs': lr_scheduler_kwargs,
                'random_seed': args.random_seed,
            }
        
        def mlpmixer_ciincl_config(trial):
            return {
                'input_size': args.input_size,
                'n_series': args.n_series,
                'patch_len': patch_len,
                'stride': stride,
                'max_steps': max_steps,
                'val_check_steps': val_check_steps,
                'windows_batch_size': windows_batch_size,
                'inference_windows_batch_size': inference_windows_batch_size,
                #'transformer_backbone': transformer_backbone,
                'hidden_size': hidden_size,
                'linear_hidden_size': linear_hidden_size,
                'n_heads': n_heads,
                'd_k': d_k,
                'd_v': d_v,
                'n_layers': n_layers,
                'pe_type': pe_type,
                'learn_pe': learn_pe,
                'dropout': dropout,
                'head_dropout': head_dropout,
                'revin': revin,
                'revin_affine': revin_affine,
                'revin_subtract_last': revin_subtract_last,
                'padding_patch': padding_patch,
                'infini_mixer_type': 'mlp',
                'infini_channel_exclusion': False,
                'mlpmixer_hidden_size': trial.suggest_categorical('mlpmixer_hidden_size', [128, 256, 384, 512]),
                'mlpmixer_n_layers': trial.suggest_categorical('mlpmixer_n_layers', [2, 3, 4]),
                'mlpmixer_dropout': trial.suggest_categorical('mlpmixer_dropout', [0.0, 0.1, 0.2]),
                'multivariate_head': multivariate_head,
                'learning_rate': learning_rate,
                'early_stop_patience_steps': early_stop_patience_steps,
                'batch_size': batch_size,
                'valid_batch_size': batch_size,
                'scaler_type': scaler_type,
                'lr_scheduler': lr_scheduler,
                'lr_scheduler_kwargs': lr_scheduler_kwargs,
                'random_seed': args.random_seed,
            }
        
        models = [
            AutoMOMENT(
                h=args.h,
                config=mlpmixer_ciexcl_config,
                loss=loss,
                search_alg=optuna.samplers.TPESampler(seed=0),
                backend='optuna',
                num_samples=5, #args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoMOMENT_mlpmixer_ciexcl'
            ),
            AutoMOMENT(
                h=args.h,
                config=mlpmixer_ciincl_config,
                loss=loss,
                search_alg=optuna.samplers.TPESampler(seed=0),
                backend='optuna',
                num_samples=5, #args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoMOMENT_mlpmixer_ciincl'
            ),
            AutoPatchTSTMultivariate(
                h=args.h,
                config=mlpmixer_ciexcl_config,
                loss=loss,
                search_alg=optuna.samplers.TPESampler(seed=0),
                backend='optuna',
                num_samples=5, #args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoPatchTSTMultivariate_mlpmixer_ciexcl'
            ),
            AutoPatchTSTMultivariate(
                h=args.h,
                config=mlpmixer_ciincl_config,
                loss=loss,
                search_alg=optuna.samplers.TPESampler(seed=0),
                backend='optuna',
                num_samples=5, #args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoPatchTSTMultivariate_mlpmixer_ciincl'
            ),
        ]

    elif args.experiment_name == 'infini_mlpquerymixer_t5tiny':
        def mlpquerymixer_ciexcl_config(trial):
            return {
                'input_size': args.input_size,
                'n_series': args.n_series,
                'patch_len': patch_len,
                'stride': stride,
                'max_steps': max_steps,
                'val_check_steps': val_check_steps,
                'windows_batch_size': windows_batch_size,
                'inference_windows_batch_size': inference_windows_batch_size,
                #'transformer_backbone': transformer_backbone,
                'hidden_size': hidden_size,
                'linear_hidden_size': linear_hidden_size,
                'n_heads': n_heads,
                'd_k': d_k,
                'd_v': d_v,
                'n_layers': n_layers,
                'pe_type': pe_type,
                'learn_pe': learn_pe,
                'dropout': dropout,
                'head_dropout': head_dropout,
                'revin': revin,
                'revin_affine': revin_affine,
                'revin_subtract_last': revin_subtract_last,
                'padding_patch': padding_patch,
                'infini_mixer_type': 'mlp_query',
                'infini_channel_exclusion': True,
                'mlpmixer_hidden_size': trial.suggest_categorical('mlpmixer_hidden_size', [128, 256, 384, 512]),
                'mlpmixer_n_layers': trial.suggest_categorical('mlpmixer_n_layers', [2, 3, 4]),
                'mlpmixer_dropout': trial.suggest_categorical('mlpmixer_dropout', [0.0, 0.1, 0.2]),
                'multivariate_head': multivariate_head,
                'learning_rate': learning_rate,
                'early_stop_patience_steps': early_stop_patience_steps,
                'batch_size': batch_size,
                'valid_batch_size': batch_size,
                'scaler_type': scaler_type,
                'lr_scheduler': lr_scheduler,
                'lr_scheduler_kwargs': lr_scheduler_kwargs,
                'random_seed': args.random_seed,
            }
        
        def mlpquerymixer_ciincl_config(trial):
            return {
                'input_size': args.input_size,
                'n_series': args.n_series,
                'patch_len': patch_len,
                'stride': stride,
                'max_steps': max_steps,
                'val_check_steps': val_check_steps,
                'windows_batch_size': windows_batch_size,
                'inference_windows_batch_size': inference_windows_batch_size,
                #'transformer_backbone': transformer_backbone,
                'hidden_size': hidden_size,
                'linear_hidden_size': linear_hidden_size,
                'n_heads': n_heads,
                'd_k': d_k,
                'd_v': d_v,
                'n_layers': n_layers,
                'pe_type': pe_type,
                'learn_pe': learn_pe,
                'dropout': dropout,
                'head_dropout': head_dropout,
                'revin': revin,
                'revin_affine': revin_affine,
                'revin_subtract_last': revin_subtract_last,
                'padding_patch': padding_patch,
                'infini_mixer_type': 'mlp_query',
                'infini_channel_exclusion': False,
                'mlpmixer_hidden_size': trial.suggest_categorical('mlpmixer_hidden_size', [128, 256, 384, 512]),
                'mlpmixer_n_layers': trial.suggest_categorical('mlpmixer_n_layers', [2, 3, 4]),
                'mlpmixer_dropout': trial.suggest_categorical('mlpmixer_dropout', [0.0, 0.1, 0.2]),
                'multivariate_head': multivariate_head,
                'learning_rate': learning_rate,
                'early_stop_patience_steps': early_stop_patience_steps,
                'batch_size': batch_size,
                'valid_batch_size': batch_size,
                'scaler_type': scaler_type,
                'lr_scheduler': lr_scheduler,
                'lr_scheduler_kwargs': lr_scheduler_kwargs,
                'random_seed': args.random_seed,
            }
        
        models = [
            AutoMOMENT(
                h=args.h,
                config=mlpquerymixer_ciexcl_config,
                loss=loss,
                search_alg=optuna.samplers.TPESampler(seed=0),
                backend='optuna',
                num_samples=5, #args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoMOMENT_mlpquerymixer_ciexcl'
            ),
            AutoMOMENT(
                h=args.h,
                config=mlpquerymixer_ciincl_config,
                loss=loss,
                search_alg=optuna.samplers.TPESampler(seed=0),
                backend='optuna',
                num_samples=5, #args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoMOMENT_mlpquerymixer_ciincl'
            ),
            AutoPatchTSTMultivariate(
                h=args.h,
                config=mlpquerymixer_ciexcl_config,
                loss=loss,
                search_alg=optuna.samplers.TPESampler(seed=0),
                backend='optuna',
                num_samples=5, #args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoPatchTSTMultivariate_mlpquerymixer_ciexcl'
            ),
            AutoPatchTSTMultivariate(
                h=args.h,
                config=mlpquerymixer_ciincl_config,
                loss=loss,
                search_alg=optuna.samplers.TPESampler(seed=0),
                backend='optuna',
                num_samples=5, #args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoPatchTSTMultivariate_mlpquerymixer_ciincl'
            ),
        ]

    elif args.experiment_name == 'infini_layerwise_t5tiny':
        infini_config1 = {
            'input_size': args.input_size,
            'n_series': args.n_series,
            'patch_len': patch_len,
            'stride': stride,
            'max_steps': max_steps,
            'val_check_steps': val_check_steps,
            'windows_batch_size': windows_batch_size,
            'inference_windows_batch_size': inference_windows_batch_size,
            #'transformer_backbone': transformer_backbone,
            'hidden_size': hidden_size,
            'linear_hidden_size': linear_hidden_size,
            'n_heads': n_heads,
            'd_k': d_k,
            'd_v': d_v,
            'n_layers': n_layers,
            'pe_type': pe_type,
            'learn_pe': learn_pe,
            'dropout': dropout,
            'head_dropout': head_dropout,
            'revin': revin,
            'revin_affine': revin_affine,
            'revin_subtract_last': revin_subtract_last,
            'padding_patch': padding_patch,
            'infini_mixer_type': 'betas',
            'infini_channel_exclusion': True,
            'layerwise_beta': True,
            'channelwise_beta': False,
            'multivariate_head': multivariate_head,
            'learning_rate': learning_rate,
            'early_stop_patience_steps': early_stop_patience_steps,
            'batch_size': batch_size,
            'valid_batch_size': batch_size,
            'scaler_type': scaler_type,
            'lr_scheduler': lr_scheduler,
            'lr_scheduler_kwargs': lr_scheduler_kwargs,
            'random_seed': args.random_seed,
        }

        infini_config2 = copy.deepcopy(infini_config1)
        infini_config2['infini_channel_exclusion'] = False

        models = [
            AutoMOMENT(
                h=args.h,
                config=infini_config1,
                loss=loss,
                search_alg=None,
                num_samples=args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoMOMENT_infini_layerwise_ciexcl'
            ),
            AutoMOMENT(
                h=args.h,
                config=infini_config2,
                loss=loss,
                search_alg=None,
                num_samples=args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoMOMENT_infini_layerwise_ciincl'
            ),
            AutoPatchTSTMultivariate(
                h=args.h,
                config=infini_config1,
                loss=loss,
                search_alg=None,
                num_samples=args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoPatchTSTMultivariate_infini_layerwise_ciexcl'
            ),
            AutoPatchTSTMultivariate(
                h=args.h,
                config=infini_config2,
                loss=loss,
                search_alg=None,
                num_samples=args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoPatchTSTMultivariate_infini_layerwise_ciincl'
            ),
        ]

    elif args.experiment_name == 'infini_layerwise_channelwise_t5tiny':
        infini_config1 = {
            'input_size': args.input_size,
            'n_series': args.n_series,
            'patch_len': patch_len,
            'stride': stride,
            'max_steps': max_steps,
            'val_check_steps': val_check_steps,
            'windows_batch_size': windows_batch_size,
            'inference_windows_batch_size': inference_windows_batch_size,
            #'transformer_backbone': transformer_backbone,
            'hidden_size': hidden_size,
            'linear_hidden_size': linear_hidden_size,
            'n_heads': n_heads,
            'd_k': d_k,
            'd_v': d_v,
            'n_layers': n_layers,
            'pe_type': pe_type,
            'learn_pe': learn_pe,
            'dropout': dropout,
            'head_dropout': head_dropout,
            'revin': revin,
            'revin_affine': revin_affine,
            'revin_subtract_last': revin_subtract_last,
            'padding_patch': padding_patch,
            'infini_mixer_type': 'betas',
            'infini_channel_exclusion': True,
            'layerwise_beta': True,
            'channelwise_beta': True,
            'multivariate_head': multivariate_head,
            'learning_rate': learning_rate,
            'early_stop_patience_steps': early_stop_patience_steps,
            'batch_size': batch_size,
            'valid_batch_size': batch_size,
            'scaler_type': scaler_type,
            'lr_scheduler': lr_scheduler,
            'lr_scheduler_kwargs': lr_scheduler_kwargs,
            'random_seed': args.random_seed,
        }

        infini_config2 = copy.deepcopy(infini_config1)
        infini_config2['infini_channel_exclusion'] = False

        models = [
            AutoMOMENT(
                h=args.h,
                config=infini_config1,
                loss=loss,
                search_alg=None,
                num_samples=args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoMOMENT_infini_layerwise_channelwise_ciexcl'
            ),
            AutoMOMENT(
                h=args.h,
                config=infini_config2,
                loss=loss,
                search_alg=None,
                num_samples=args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoMOMENT_infini_layerwise_channelwise_ciincl'
            ),
            AutoPatchTSTMultivariate(
                h=args.h,
                config=infini_config1,
                loss=loss,
                search_alg=None,
                num_samples=args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoPatchTSTMultivariate_infini_layerwise_channelwise_ciexcl'
            ),
            AutoPatchTSTMultivariate(
                h=args.h,
                config=infini_config2,
                loss=loss,
                search_alg=None,
                num_samples=args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoPatchTSTMultivariate_infini_layerwise_channelwise_ciincl'
            ),
        ]

    elif args.experiment_name == 'infini_channelwise_t5tiny':
        infini_config1 = {
            'input_size': args.input_size,
            'n_series': args.n_series,
            'patch_len': patch_len,
            'stride': stride,
            'max_steps': max_steps,
            'val_check_steps': val_check_steps,
            'windows_batch_size': windows_batch_size,
            'inference_windows_batch_size': inference_windows_batch_size,
            #'transformer_backbone': transformer_backbone,
            'hidden_size': hidden_size,
            'linear_hidden_size': linear_hidden_size,
            'n_heads': n_heads,
            'd_k': d_k,
            'd_v': d_v,
            'n_layers': n_layers,
            'pe_type': pe_type,
            'learn_pe': learn_pe,
            'dropout': dropout,
            'head_dropout': head_dropout,
            'revin': revin,
            'revin_affine': revin_affine,
            'revin_subtract_last': revin_subtract_last,
            'padding_patch': padding_patch,
            'infini_mixer_type': 'betas',
            'infini_channel_exclusion': True,
            'layerwise_beta': False,
            'channelwise_beta': True,
            'multivariate_head': multivariate_head,
            'learning_rate': learning_rate,
            'early_stop_patience_steps': early_stop_patience_steps,
            'batch_size': batch_size,
            'valid_batch_size': batch_size,
            'scaler_type': scaler_type,
            'lr_scheduler': lr_scheduler,
            'lr_scheduler_kwargs': lr_scheduler_kwargs,
            'random_seed': args.random_seed,
        }

        infini_config2 = copy.deepcopy(infini_config1)
        infini_config2['infini_channel_exclusion'] = False

        models = [
            AutoMOMENT(
                h=args.h,
                config=infini_config1,
                loss=loss,
                search_alg=None,
                num_samples=args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoMOMENT_infini_channelwise_ciexcl'
            ),
            AutoMOMENT(
                h=args.h,
                config=infini_config2,
                loss=loss,
                search_alg=None,
                num_samples=args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoMOMENT_infini_channelwise_ciincl'
            ),
            AutoPatchTSTMultivariate(
                h=args.h,
                config=infini_config1,
                loss=loss,
                search_alg=None,
                num_samples=args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoPatchTSTMultivariate_infini_channelwise_ciexcl'
            ),
            AutoPatchTSTMultivariate(
                h=args.h,
                config=infini_config2,
                loss=loss,
                search_alg=None,
                num_samples=args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoPatchTSTMultivariate_infini_channelwise_ciincl'
            ),
        ]

    elif args.experiment_name == 'infini_t5tiny':
        infini_config1 = {
            'input_size': args.input_size,
            'n_series': args.n_series,
            'patch_len': patch_len,
            'stride': stride,
            'max_steps': max_steps,
            'val_check_steps': val_check_steps,
            'windows_batch_size': windows_batch_size,
            'inference_windows_batch_size': inference_windows_batch_size,
            #'transformer_backbone': transformer_backbone,
            'hidden_size': hidden_size,
            'linear_hidden_size': linear_hidden_size,
            'n_heads': n_heads,
            'd_k': d_k,
            'd_v': d_v,
            'n_layers': n_layers,
            'pe_type': pe_type,
            'learn_pe': learn_pe,
            'dropout': dropout,
            'head_dropout': head_dropout,
            'revin': revin,
            'revin_affine': revin_affine,
            'revin_subtract_last': revin_subtract_last,
            'padding_patch': padding_patch,
            'infini_mixer_type': 'betas',
            'infini_channel_exclusion': True,
            'layerwise_beta': False,
            'channelwise_beta': False,
            'multivariate_head': multivariate_head,
            'learning_rate': learning_rate,
            'early_stop_patience_steps': early_stop_patience_steps,
            'batch_size': batch_size,
            'valid_batch_size': batch_size,
            'scaler_type': scaler_type,
            'lr_scheduler': lr_scheduler,
            'lr_scheduler_kwargs': lr_scheduler_kwargs,
            'random_seed': args.random_seed,
        }

        infini_config2 = copy.deepcopy(infini_config1)
        infini_config2['infini_channel_exclusion'] = False

        models = [
            AutoMOMENT(
                h=args.h,
                config=infini_config1,
                loss=loss,
                search_alg=None,
                num_samples=args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoMOMENT_infini_ciexcl'
            ),
            AutoMOMENT(
                h=args.h,
                config=infini_config2,
                loss=loss,
                search_alg=None,
                num_samples=args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoMOMENT_infini_ciincl'
            ),
            AutoPatchTSTMultivariate(
                h=args.h,
                config=infini_config1,
                loss=loss,
                search_alg=None,
                num_samples=args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoPatchTSTMultivariate_infini_ciexcl'
            ),
            AutoPatchTSTMultivariate(
                h=args.h,
                config=infini_config2,
                loss=loss,
                search_alg=None,
                num_samples=args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoPatchTSTMultivariate_infini_ciincl'
            ),
        ]

    elif args.experiment_name == 'multivariateMLP_baseline':
        config = {
            'input_size': args.input_size,
            'n_series': args.n_series,
            'max_steps': max_steps,
            'val_check_steps': val_check_steps,
            'windows_batch_size': windows_batch_size,
            'inference_windows_batch_size': inference_windows_batch_size,
            'hidden_size': hidden_size,
            'num_layers': n_layers,
            'learning_rate': learning_rate,
            'early_stop_patience_steps': early_stop_patience_steps,
            'batch_size': batch_size,
            'valid_batch_size': batch_size,
            'scaler_type': scaler_type,
            'lr_scheduler': lr_scheduler,
            'lr_scheduler_kwargs': lr_scheduler_kwargs,
            'random_seed': args.random_seed,
            'univariate': False,
        }

        models = [
            AutoMLPMultivariate(
                h=args.h, 
                config=config,
                loss=loss,
                search_alg=None,
                num_samples=args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoMLPMultivariate_multivariate'
                ),
        ]

    elif args.experiment_name == 'tsmixer_baseline':
        config = {
            'input_size': args.input_size,
            'n_series': args.n_series,
            'max_steps': max_steps,
            'val_check_steps': val_check_steps,
            'windows_batch_size': windows_batch_size,
            'inference_windows_batch_size': inference_windows_batch_size,
            'n_block': n_layers,
            'ff_dim': hidden_size,
            'dropout': dropout,
            'revin': revin,
            'revin_affine': revin_affine,
            'revin_subtract_last': revin_subtract_last,
            'learning_rate': learning_rate,
            'early_stop_patience_steps': early_stop_patience_steps,
            'batch_size': batch_size,
            'valid_batch_size': batch_size,
            'scaler_type': scaler_type,
            'lr_scheduler': lr_scheduler,
            'lr_scheduler_kwargs': lr_scheduler_kwargs,
            'random_seed': args.random_seed,
            'univariate': False,
        }
    
        models=[
            AutoTSMixer(
                h=args.h, 
                config=config,
                loss=loss,
                search_alg=None,
                num_samples=args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoTSMixer_multivariate'
            ),
        ]
        
    elif args.experiment_name == 'itransformer_baseline':
        config = {
            'input_size': args.input_size,
            'n_series': args.n_series,
            'max_steps': max_steps,
            'val_check_steps': val_check_steps,
            'windows_batch_size': windows_batch_size,
            'inference_windows_batch_size': inference_windows_batch_size,
            #'transformer_backbone': transformer_backbone,
            'hidden_size': hidden_size,
            'linear_hidden_size': linear_hidden_size,
            'n_heads': n_heads,
            'd_k': d_k,
            'd_v': d_v,
            'n_layers': n_layers,
            'dropout': dropout,
            'head_dropout': head_dropout,
            'revin': revin,
            'revin_affine': revin_affine,
            'revin_subtract_last': revin_subtract_last,
            'multivariate_head': multivariate_head,
            'learning_rate': learning_rate,
            'early_stop_patience_steps': early_stop_patience_steps,
            'batch_size': batch_size,
            'valid_batch_size': batch_size,
            'scaler_type': scaler_type,
            'lr_scheduler': lr_scheduler,
            'lr_scheduler_kwargs': lr_scheduler_kwargs,
            'random_seed': args.random_seed,
            'univariate': False,
        }
    
        models=[
            AutoiTransformer(
                h=args.h, 
                config=config,
                loss=loss,
                search_alg=None,
                num_samples=args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoiTransformer_multivariate'
            ),
            AutoiTransformerT5(
                h=args.h, 
                config=config,
                loss=loss,
                search_alg=None,
                num_samples=args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoiTransformerT5_multivariate'
            ),
        ]

    elif args.experiment_name == 'timerxl_baseline':
        config = {
            'input_size': args.input_size,
            'n_series': args.n_series,
            'patch_len': patch_len,
            'stride': stride,
            'max_steps': max_steps,
            'val_check_steps': val_check_steps,
            'windows_batch_size': windows_batch_size,
            'inference_windows_batch_size': inference_windows_batch_size,
            'hidden_size': hidden_size,
            'linear_hidden_size': linear_hidden_size,
            'n_heads': n_heads,
            'd_k': d_k,
            'd_v': d_v,
            'n_layers': n_layers,
            'pe_type': pe_type,
            'learn_pe': learn_pe,
            'dropout': dropout,
            'head_dropout': head_dropout,
            'revin': revin,
            'revin_affine': revin_affine,
            'revin_subtract_last': revin_subtract_last,
            'padding_patch': padding_patch,
            'multivariate_head': multivariate_head,
            'learning_rate': learning_rate,
            'early_stop_patience_steps': early_stop_patience_steps,
            'batch_size': batch_size,
            'valid_batch_size': batch_size,
            'scaler_type': scaler_type,
            'lr_scheduler': lr_scheduler,
            'lr_scheduler_kwargs': lr_scheduler_kwargs,
            'random_seed': args.random_seed,
            'univariate': False,
        }

        models=[
            AutoTimerXL(
                h=args.h, 
                config=config,
                loss=loss,
                search_alg=None,
                num_samples=args.num_samples,
                cpus=20,
                n_series=args.n_series,
                alias='AutoTimerXL_multivariate'
            ),
        ]

    elif args.experiment_name == 'crossformer_baseline':
        config = {
            'input_size': args.input_size,
            'n_series': args.n_series,
            'patch_len': patch_len,
            'stride': stride,
            'max_steps': max_steps,
            'val_check_steps': val_check_steps,
            'windows_batch_size': windows_batch_size,
            'inference_windows_batch_size': inference_windows_batch_size,
            'hidden_size': hidden_size,
            'linear_hidden_size': linear_hidden_size,
            'n_heads': n_heads,
            'd_k': d_k,
            'd_v': d_v,
            'n_layers': n_layers,
            'pe_type': pe_type,
            'learn_pe': learn_pe,
            'dropout': dropout,
            'head_dropout': head_dropout,
            'revin': revin,
            'revin_affine': revin_affine,
            'revin_subtract_last': revin_subtract_last,
            'padding_patch': padding_patch,
            'learning_rate': learning_rate,
            'early_stop_patience_steps': early_stop_patience_steps,
            'batch_size': batch_size,
            'valid_batch_size': batch_size,
            'scaler_type': scaler_type,
            'lr_scheduler': lr_scheduler,
            'lr_scheduler_kwargs': lr_scheduler_kwargs,
            'random_seed': args.random_seed,
            'univariate': False,
        }

        models=[
            AutoCrossformer(
                    h=args.h, 
                    config=config,
                    loss=loss,
                    search_alg=None,
                    num_samples=args.num_samples,
                    cpus=20,
                    n_series=args.n_series,
                    alias='AutoCrossformer_multivariate'
                ),
        ]

    elif args.experiment_name == 'timemixer_baseline':
        config = {
            'input_size': args.input_size,
            'n_series': args.n_series,
            'max_steps': max_steps,
            'val_check_steps': val_check_steps,
            'windows_batch_size': windows_batch_size,
            'inference_windows_batch_size': inference_windows_batch_size,
            'd_model': hidden_size,
            'd_ff': linear_hidden_size,
            'e_layers': n_layers,
            'decomp_method': 'moving_avg',
            'down_sampling_method': 'avg',
            'down_sampling_layers': 1,
            'down_sampling_window': 2, 
            'moving_avg': args.h // 2,
            'top_k': 5,
            'dropout': dropout,
            'revin': revin,
            'revin_affine': revin_affine,
            'revin_subtract_last': revin_subtract_last,
            'learning_rate': learning_rate,
            'early_stop_patience_steps': early_stop_patience_steps,
            'batch_size': batch_size,
            'valid_batch_size': batch_size,
            'scaler_type': scaler_type,
            'lr_scheduler': lr_scheduler,
            'lr_scheduler_kwargs': lr_scheduler_kwargs,
            'random_seed': args.random_seed,
            'channel_independence': 0 # channel-dependence
        }

        models=[
            AutoTimeMixer(
                    h=args.h, 
                    config=config,
                    loss=loss,
                    search_alg=None,
                    num_samples=args.num_samples,
                    cpus=20,
                    n_series=args.n_series,
                    alias='AutoTimeMixer_multivariate'
                ),
        ]

    elif args.experiment_name == 'statsforecast':
        from statsforecast.models import AutoETS

        if args.freq == 'H':
            season_length = 24
        elif args.freq == 'D':
            season_length = 7
        elif args.freq == 'W':
            season_length = 52
        elif args.freq == 'M':
            season_length = 12
        elif args.freq == 'Q':
            season_length = 4
        elif args.freq == 'Y':
            season_length = 1
        elif args.freq == 'T' or args.freq == 'min':
            season_length = 60 
        elif args.freq == 'S':
            season_length = 60 
        else:
            season_length = 1

        models = [
            AutoETS(season_length = season_length),   
        ]

    elif args.experiment_name == 'chronos2.0_baseline':
        config = {
            'input_size': args.input_size,
            'n_series': args.n_series,
            'max_steps': max_steps,
            'val_check_steps': val_check_steps,
            "top_k": 1,              # Always pick most likely value
            "top_p": 1.0,            # Doesn't matter when top_k=1
            "univariate": False,
        }

        models=[
            AutoChronos2(
                h=args.h,
                config=config,
                cpus=20,
                n_series=args.n_series,
                alias='Chronos_multivariate'
            ),
        ]

    else:
        raise ValueError(
            f"Unknown experiment name: {args.experiment_name}. "
        )

    return models
