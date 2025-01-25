import numpy as np

from neuralforecast.models import T5Flex, MLP, PatchTST, NHITS, NBEATS, PatchTST, VanillaTransformer, TFT, Autoformer, Informer, iTransformer, DLinear, NLinear, TCN, LSTM, TimesNet, TSMixer
from neuralforecast.losses.pytorch import MAE, MSE, HuberLoss, DistributionLoss

from model_config import Model_Configs


def get_models(args):
    h = args.h
    context_len = 256
    max_steps = 10000 
    val_check_steps = 100
    early_stop_patience_steps = 20
    batch_size = 4
    input_token_len = 96
    stride = 8
    random_seed = args.random_seed
    ckpt_path = None


    if args.experiment_name == 'tokenlen_ablation':
        tokenizer_type = 'patch_fixed_length'
        scaler_type = 'standard'
        attn_mask = "bidirectional"
        padding_patch = None #'end'
        proj_embd_type = 'linear'
        proj_head_type = 'linear'
        pe = 'sincos_relative'
        num_decoder_layers=0
        loss = MAE()

        token_lens = [8, 16, 32, 64, 96, 128]

        models = [T5Flex(
                 h=h,
                 context_len=context_len,
                 max_steps=max_steps,
                 val_check_steps=val_check_steps,
                 early_stop_patience_steps=early_stop_patience_steps,
                 batch_size=batch_size,
                 input_token_len=token_len,
                 output_token_len=h,
                 backbone_type="google/t5-efficient-tiny",
                 attn_mask=attn_mask,
                 stride=stride,
                 num_decoder_layers=num_decoder_layers,
                 proj_embd_type=proj_embd_type, 
                 proj_head_type=proj_head_type,
                 tokenizer_type=tokenizer_type,
                 padding_patch=padding_patch, 
                 pe=pe, 
                 scaler_type=scaler_type,
                 loss=loss,
                 random_seed=random_seed,
                 alias=f't5flex_tokenlen{token_len}',
                 ckpt_path=ckpt_path,
                ) #Patching
                 for token_len in token_lens
            ]

    if args.experiment_name == 'contextlen_ablation':
        tokenizer_type = 'patch_fixed_length'
        scaler_type = 'standard'
        attn_mask = "bidirectional"
        padding_patch = None #'end'
        proj_embd_type = 'linear'
        proj_head_type = 'linear'
        pe = 'sincos_relative'  
        num_decoder_layers=0
        loss = MAE()

        models = [T5Flex(
                 h=h,
                 context_len=context_len,
                 max_steps=max_steps,
                 val_check_steps=val_check_steps,
                 early_stop_patience_steps=early_stop_patience_steps,
                 batch_size=batch_size,
                 input_token_len=token_len,
                 output_token_len=h,
                 backbone_type="google/t5-efficient-tiny",
                 attn_mask=attn_mask,
                 stride=stride,
                 num_decoder_layers=num_decoder_layers,
                 proj_embd_type=proj_embd_type, 
                 proj_head_type=proj_head_type,
                 tokenizer_type=tokenizer_type,
                 padding_patch=padding_patch, 
                 pe=pe, 
                 scaler_type=scaler_type,
                 loss=loss,
                 random_seed=random_seed,
                 alias=f't5flex_256',
                 ckpt_path=ckpt_path,
                ),
                T5Flex(  # We already have the same model trained in other experiments
                 h=h,
                 context_len=512,
                 max_steps=max_steps,
                 val_check_steps=val_check_steps,
                 early_stop_patience_steps=early_stop_patience_steps,
                 batch_size=batch_size,
                 input_token_len=input_token_len,
                 output_token_len=h,
                 backbone_type="google/t5-efficient-tiny",
                 attn_mask=attn_mask,
                 stride=stride,
                 num_decoder_layers=num_decoder_layers,
                 proj_embd_type=proj_embd_type, 
                 proj_head_type=proj_head_type,
                 tokenizer_type=tokenizer_type,
                 padding_patch=padding_patch, 
                 pe=pe, 
                 scaler_type=scaler_type,
                 loss=loss,
                 random_seed=random_seed,
                 alias=f't5flex_512',
                 ckpt_path=ckpt_path,
                ),
            ]
    
    if args.experiment_name == 'size_ablation':
        tokenizer_type = 'patch_fixed_length'
        scaler_type = 'standard'
        attn_mask = "bidirectional"
        padding_patch = None #'end'
        proj_embd_type = 'linear'
        proj_head_type = 'linear'
        pe = 'sincos_relative'  
        num_decoder_layers=0
        loss = MAE()

        models = [T5Flex(
                 h=h,
                 context_len=context_len,
                 max_steps=max_steps,
                 val_check_steps=val_check_steps,
                 early_stop_patience_steps=early_stop_patience_steps,
                 batch_size=batch_size,
                 input_token_len=input_token_len,
                 output_token_len=h,
                 backbone_type="google/t5-efficient-tiny",
                 attn_mask=attn_mask,
                 stride=stride,
                 num_decoder_layers=num_decoder_layers,
                 proj_embd_type=proj_embd_type, 
                 proj_head_type=proj_head_type,
                 tokenizer_type=tokenizer_type,
                 padding_patch=padding_patch, 
                 pe=pe, 
                 scaler_type=scaler_type,
                 loss=loss,
                 random_seed=random_seed,
                 alias='t5flex_tiny',
                 ckpt_path=ckpt_path,
                ), #Patching
                 T5Flex(
                 h=h,
                 context_len=context_len,
                 max_steps=max_steps,
                 val_check_steps=val_check_steps,
                 early_stop_patience_steps=early_stop_patience_steps,
                 batch_size=batch_size,
                 input_token_len=input_token_len, 
                 output_token_len=h,
                 backbone_type="google/t5-efficient-mini",
                 attn_mask=attn_mask,
                 stride=stride,
                 num_decoder_layers=num_decoder_layers,
                 proj_embd_type=proj_embd_type, 
                 proj_head_type=proj_head_type,
                 tokenizer_type=tokenizer_type,
                 padding_patch=padding_patch, 
                 pe=pe, 
                 scaler_type=scaler_type,
                 loss=loss,
                 random_seed=random_seed,
                 alias='t5flex_mini',
                 ckpt_path=ckpt_path,
                ), #Patching
                 T5Flex(
                 h=h,
                 context_len=context_len,
                 max_steps=max_steps,
                 val_check_steps=val_check_steps,
                 early_stop_patience_steps=early_stop_patience_steps,
                 batch_size=batch_size,
                 input_token_len=input_token_len, 
                 output_token_len=h,
                 backbone_type="google/t5-efficient-small",
                 attn_mask=attn_mask,
                 stride=stride,
                 num_decoder_layers=num_decoder_layers,
                 proj_embd_type=proj_embd_type, 
                 proj_head_type=proj_head_type,
                 tokenizer_type=tokenizer_type,
                 padding_patch=padding_patch, 
                 pe=pe, 
                 scaler_type=scaler_type,
                 loss=loss,
                 random_seed=random_seed,
                 alias='t5flex_small',
                 ckpt_path=ckpt_path,
                ), #Patching
                 T5Flex(
                 h=h,
                 context_len=context_len,
                 max_steps=max_steps,
                 val_check_steps=val_check_steps,
                 early_stop_patience_steps=early_stop_patience_steps,
                 batch_size=batch_size,
                 input_token_len=input_token_len, 
                 output_token_len=h,
                 backbone_type="google/t5-efficient-base",
                 attn_mask=attn_mask,
                 stride=stride,
                 num_decoder_layers=num_decoder_layers,
                 proj_embd_type=proj_embd_type, 
                 proj_head_type=proj_head_type,
                 tokenizer_type=tokenizer_type,
                 padding_patch=padding_patch, 
                 pe=pe, 
                 scaler_type=scaler_type,
                 loss=loss,
                 random_seed=random_seed,
                 alias='t5flex_base',
                 ckpt_path=ckpt_path,
                ), #Patching
            ]
    
    if args.experiment_name == 'tokenization_ablation':
        scaler_type = 'standard'
        backbone_type = "google/t5-efficient-mini"
        attn_mask = "bidirectional"
        proj_embd_type="linear"
        proj_head_type="linear"
        pe='sincos_relative'
        num_decoder_layers=0
        context_len=192
        
        models = [T5Flex(
                 h=h,
                 context_len=context_len,
                 max_steps=max_steps,
                 val_check_steps=val_check_steps,
                 early_stop_patience_steps=early_stop_patience_steps,
                 batch_size=batch_size,
                 input_token_len=1, 
                 output_token_len=h,
                 backbone_type=backbone_type,
                 attn_mask=attn_mask,
                 stride=1,
                 num_decoder_layers=num_decoder_layers,
                 proj_embd_type=proj_embd_type, 
                 proj_head_type=proj_head_type,
                 tokenizer_type='patch_fixed_length',
                 padding_patch=None, #'end', 
                 pe=pe, 
                 scaler_type=scaler_type,
                 loss=loss,
                 random_seed=random_seed,
                 alias='t5flex_none',
                 ckpt_path=ckpt_path,
                ), #None
                T5Flex(
                 h=h,
                 context_len=context_len,
                 max_steps=max_steps,
                 val_check_steps=val_check_steps,
                 early_stop_patience_steps=early_stop_patience_steps,
                 batch_size=batch_size,
                 input_token_len=input_token_len, 
                 output_token_len=h,
                 backbone_type=backbone_type,
                 attn_mask=attn_mask,
                 stride=stride,
                 num_decoder_layers=num_decoder_layers,
                 proj_embd_type=proj_embd_type, 
                 proj_head_type=proj_head_type,
                 tokenizer_type='patch_fixed_length',
                 padding_patch=None, #'end', 
                 pe=pe, 
                 scaler_type=scaler_type,
                 loss=loss,
                 random_seed=random_seed,
                 alias='t5flex_patch',
                 ckpt_path=ckpt_path,
                ), #Patching
                T5Flex(
                 h=h,
                 context_len=context_len,
                 max_steps=max_steps,
                 val_check_steps=val_check_steps,
                 early_stop_patience_steps=early_stop_patience_steps,
                 batch_size=batch_size,
                 input_token_len=1, 
                 output_token_len=h,
                 backbone_type=backbone_type,
                 attn_mask=attn_mask,
                 stride=1,
                 num_decoder_layers=num_decoder_layers,
                 proj_embd_type=proj_embd_type, 
                 proj_head_type=proj_head_type,
                 tokenizer_type='bins',
                 padding_patch=None,
                 pe=pe, 
                 scaler_type=scaler_type,
                 random_seed=random_seed,
                 ckpt_path=ckpt_path,
                 alias='t5flex_bin',
                 loss=DistributionLoss(distribution='Categorical',
                                       level=[80, 90], 
                                       num_cats=10,
                                      )
                 ), #Binning
                 T5Flex(
                 h=h,
                 context_len=context_len,
                 max_steps=max_steps,
                 val_check_steps=val_check_steps,
                 early_stop_patience_steps=early_stop_patience_steps,
                 batch_size=batch_size,
                 input_token_len=1, 
                 output_token_len=h,
                 backbone_type=backbone_type,
                 attn_mask=attn_mask,
                 stride=1,
                 num_decoder_layers=num_decoder_layers,
                 proj_embd_type=proj_embd_type, 
                 proj_head_type=proj_head_type,
                 tokenizer_type ='lags',
                 padding_patch=None,
                 lag=2, 
                 scaler_type=scaler_type,
                 loss=MAE(),
                 random_seed=random_seed,
                 alias='t5flex_lag',
                 ckpt_path=ckpt_path,
                 ), #Lags
            ]
        
    if args.experiment_name == 'pe_ablation':
        tokenizer_type = 'patch_fixed_length'
        scaler_type = 'standard'
        backbone_type = "google/t5-efficient-tiny"
        attn_mask = "bidirectional"
        proj_embd_type = "linear"
        proj_head_type = "linear"
        padding_patch = None #'end'
        num_decoder_layers = 0
        loss = MAE()
        
        models = [T5Flex(
                 h=h,
                 context_len=context_len,
                 max_steps=max_steps,
                 val_check_steps=val_check_steps,
                 early_stop_patience_steps=early_stop_patience_steps,
                 batch_size=batch_size,
                 input_token_len=input_token_len, 
                 output_token_len=h,
                 backbone_type=backbone_type,
                 attn_mask=attn_mask,
                 stride=stride,
                 num_decoder_layers=num_decoder_layers,
                 proj_embd_type=proj_embd_type, 
                 proj_head_type=proj_head_type,
                 tokenizer_type=tokenizer_type, 
                 padding_patch=padding_patch, 
                 pe='sincos', 
                 scaler_type=scaler_type,
                 loss=loss,
                 random_seed=random_seed,
                 alias='t5flex_sincos',
                 ckpt_path=ckpt_path,
                ), 
                 T5Flex(
                 h=h,
                 context_len=context_len,
                 max_steps=max_steps,
                 val_check_steps=val_check_steps,
                 early_stop_patience_steps=early_stop_patience_steps,
                 batch_size=batch_size,
                 input_token_len=input_token_len, 
                 output_token_len=h,
                 backbone_type=backbone_type,
                 attn_mask=attn_mask,
                 stride=stride,
                 num_decoder_layers=num_decoder_layers,
                 proj_embd_type=proj_embd_type, 
                 proj_head_type=proj_head_type,
                 tokenizer_type=tokenizer_type,
                 padding_patch=padding_patch,
                 pe='relative', 
                 scaler_type=scaler_type,
                 loss=loss,
                 random_seed=random_seed,
                 alias='t5flex_relative',
                 ckpt_path=ckpt_path,
                 ), 
                 T5Flex(
                 h=h,
                 context_len=context_len,
                 max_steps=max_steps,
                 val_check_steps=val_check_steps,
                 early_stop_patience_steps=early_stop_patience_steps,
                 batch_size=batch_size,
                 input_token_len=input_token_len, 
                 output_token_len=h,
                 backbone_type=backbone_type,
                 attn_mask=attn_mask,
                 stride=stride,
                 num_decoder_layers=num_decoder_layers,
                 proj_embd_type=proj_embd_type, 
                 proj_head_type=proj_head_type,
                 tokenizer_type=tokenizer_type,
                 padding_patch=padding_patch,
                 pe='sincos_relative', 
                 scaler_type=scaler_type,
                 loss=loss,
                 random_seed=random_seed,
                 alias='t5flex_sincosrelative',
                 ckpt_path=ckpt_path,
                 ), 
                 T5Flex(
                 h=h,
                 context_len=context_len,
                 max_steps=max_steps,
                 val_check_steps=val_check_steps,
                 early_stop_patience_steps=early_stop_patience_steps,
                 batch_size=batch_size,
                 input_token_len=input_token_len, 
                 output_token_len=h,
                 backbone_type=backbone_type,
                 attn_mask=attn_mask,
                 stride=stride,
                 num_decoder_layers=num_decoder_layers,
                 proj_embd_type=proj_embd_type, 
                 proj_head_type=proj_head_type,
                 tokenizer_type=tokenizer_type,
                 padding_patch=padding_patch,
                 pe='rope', 
                 scaler_type=scaler_type,
                 loss=loss,
                 random_seed=random_seed,
                 alias='t5flex_rope',
                 ckpt_path=ckpt_path,
                 ), 
            ]

    if args.experiment_name == 'loss_ablation':
        tokenizer_type = 'patch_fixed_length'
        scaler_type = 'standard'
        backbone_type = "google/t5-efficient-tiny"
        attn_mask = "bidirectional"
        proj_embd_type="linear" 
        proj_head_type="linear"
        padding_patch = None #'end'
        pe = 'sincos_relative'
        num_decoder_layers=0
        
        models = [T5Flex(
                 h=h,
                 context_len=context_len,
                 max_steps=max_steps,
                 val_check_steps=val_check_steps,
                 early_stop_patience_steps=early_stop_patience_steps,
                 batch_size=batch_size,
                 input_token_len=input_token_len, 
                 output_token_len=h,
                 backbone_type=backbone_type,
                 attn_mask=attn_mask,
                 stride=stride,
                 num_decoder_layers=num_decoder_layers,
                 proj_embd_type=proj_embd_type, 
                 proj_head_type=proj_head_type,
                 tokenizer_type=tokenizer_type, 
                 padding_patch=padding_patch, 
                 pe=pe, 
                 scaler_type=scaler_type,
                 loss=MAE(),
                 random_seed=random_seed,
                 alias='t5flex_mae',
                 ckpt_path=ckpt_path,
                ), 
                 T5Flex(
                 h=h,
                 context_len=context_len,
                 max_steps=max_steps,
                 val_check_steps=val_check_steps,
                 early_stop_patience_steps=early_stop_patience_steps,
                 batch_size=batch_size,
                 input_token_len=input_token_len, 
                 output_token_len=h,
                 backbone_type=backbone_type,
                 attn_mask=attn_mask,
                 stride=stride,
                 num_decoder_layers=num_decoder_layers,
                 proj_embd_type=proj_embd_type, 
                 proj_head_type=proj_head_type,
                 tokenizer_type=tokenizer_type,
                 padding_patch=padding_patch,
                 pe=pe, 
                 scaler_type=scaler_type,
                 loss=MSE(),
                 random_seed=random_seed,
                 alias='t5flex_mse',
                 ckpt_path=ckpt_path,
                 ), 
                 T5Flex(
                 h=h,
                 context_len=context_len,
                 max_steps=max_steps,
                 val_check_steps=val_check_steps,
                 early_stop_patience_steps=early_stop_patience_steps,
                 batch_size=batch_size,
                 input_token_len=input_token_len, 
                 output_token_len=h,
                 backbone_type=backbone_type,
                 attn_mask=attn_mask,
                 stride=stride,
                 num_decoder_layers=num_decoder_layers,
                 proj_embd_type=proj_embd_type, 
                 proj_head_type=proj_head_type,
                 tokenizer_type=tokenizer_type,
                 padding_patch=padding_patch,
                 pe=pe, 
                 scaler_type=scaler_type,
                 loss=HuberLoss(),
                 random_seed=random_seed,
                 alias='t5flex_huber',
                 ckpt_path=ckpt_path,
                 ), 
                 T5Flex(
                 h=h,
                 context_len=context_len,
                 max_steps=max_steps,
                 val_check_steps=val_check_steps,
                 early_stop_patience_steps=early_stop_patience_steps,
                 batch_size=batch_size,
                 input_token_len=input_token_len, 
                 output_token_len=h,
                 backbone_type=backbone_type,
                 attn_mask=attn_mask,
                 stride=stride,
                 num_decoder_layers=num_decoder_layers,
                 proj_embd_type=proj_embd_type, 
                 proj_head_type=proj_head_type,
                 tokenizer_type=tokenizer_type,
                 padding_patch=padding_patch,
                 pe=pe, 
                 scaler_type=scaler_type,
                 random_seed=random_seed,
                 ckpt_path=ckpt_path,
                 alias='t5flex_studentt',
                 loss=DistributionLoss(distribution='StudentT',
                                       level=[80, 90], 
                                      )
                 ), 
            ]
                  
    if args.experiment_name == 'architecture_ablation':
        tokenizer_type = 'patch_fixed_length'
        scaler_type = 'standard'
        backbone_type = "google/t5-efficient-tiny"
        proj_embd_type="linear"
        proj_head_type="linear"
        padding_patch = None #'end'
        pe = 'sincos_relative'
        loss = MAE()
        
        models = [T5Flex(
                 h=h,
                 context_len=context_len,
                 max_steps=max_steps,
                 val_check_steps=val_check_steps,
                 early_stop_patience_steps=early_stop_patience_steps,
                 batch_size=batch_size,
                 input_token_len=input_token_len, 
                 output_token_len=h,
                 backbone_type=backbone_type,
                 attn_mask='bidirectional',
                 stride=stride,
                 num_decoder_layers=0,
                 proj_embd_type=proj_embd_type, 
                 proj_head_type=proj_head_type,
                 tokenizer_type=tokenizer_type, 
                 padding_patch=padding_patch, 
                 pe=pe, 
                 scaler_type=scaler_type,
                 loss=loss,
                 random_seed=random_seed,
                 alias='t5flex_encoderb',
                 ckpt_path=ckpt_path,
                ), 
                 T5Flex(
                 h=h,
                 context_len=context_len,
                 max_steps=max_steps,
                 val_check_steps=val_check_steps,
                 early_stop_patience_steps=early_stop_patience_steps,
                 batch_size=batch_size,
                 input_token_len=input_token_len, 
                 output_token_len=h,
                 backbone_type=backbone_type,
                 num_decoder_layers=0,
                 attn_mask='causal',
                 stride=stride,
                 proj_embd_type=proj_embd_type, 
                 proj_head_type=proj_head_type,
                 tokenizer_type=tokenizer_type,
                 padding_patch=padding_patch,
                 pe=pe, 
                 scaler_type=scaler_type,
                 loss=loss,
                 random_seed=random_seed,
                 alias='t5flex_encoderc',
                 ckpt_path=ckpt_path,
                 ), 
            ]
            
    if args.experiment_name == 'proj_ablation':
        tokenizer_type = 'patch_fixed_length'
        scaler_type = 'standard'
        backbone_type = "google/t5-efficient-tiny"
        attn_mask = "bidirectional"
        padding_patch = None #'end'
        pe = 'sincos_relative'  
        num_decoder_layers = 0
        loss = MAE()
        
        models = [T5Flex(
                 h=h,
                 context_len=context_len,
                 max_steps=max_steps,
                 val_check_steps=val_check_steps,
                 early_stop_patience_steps=early_stop_patience_steps,
                 batch_size=batch_size,
                 input_token_len=input_token_len, 
                 output_token_len=h,
                 backbone_type=backbone_type,
                 attn_mask=attn_mask,
                 stride=stride,
                 num_decoder_layers=num_decoder_layers,
                 proj_embd_type='linear', 
                 proj_head_type='linear',
                 tokenizer_type=tokenizer_type,
                 padding_patch=padding_patch, 
                 pe=pe, 
                 scaler_type=scaler_type,
                 loss=loss,
                 random_seed=random_seed,
                 alias='t5flex_linearproj',
                 ckpt_path=ckpt_path,
                ), #Patching
                 T5Flex(
                 h=h,
                 context_len=context_len,
                 max_steps=max_steps,
                 val_check_steps=val_check_steps,
                 early_stop_patience_steps=early_stop_patience_steps,
                 batch_size=batch_size,
                 input_token_len=input_token_len, 
                 output_token_len=h,
                 backbone_type=backbone_type,
                 attn_mask=attn_mask,
                 stride=stride,
                 num_decoder_layers=num_decoder_layers,
                 proj_embd_type='residual', 
                 proj_head_type='residual',
                 tokenizer_type=tokenizer_type,
                 padding_patch=padding_patch, 
                 pe=pe, 
                 scaler_type=scaler_type,
                 loss=loss,
                 random_seed=random_seed,
                 alias='t5flex_residualproj',
                 ckpt_path=ckpt_path,
                ), #Patching
            ]
                  
    if args.experiment_name == 'scaler_ablation':
        tokenizer_type = 'patch_fixed_length'
        backbone_type = "google/t5-efficient-tiny"
        attn_mask = "bidirectional"
        proj_embd_type = 'linear'
        proj_head_type = 'linear'
        padding_patch = None #'end'
        pe = 'sincos_relative'
        num_decoder_layers=0
        loss = MAE()
        
        models = [T5Flex(
                 h=h,
                 context_len=context_len,
                 max_steps=max_steps,
                 val_check_steps=val_check_steps,
                 early_stop_patience_steps=early_stop_patience_steps,
                 batch_size=batch_size,
                 input_token_len=input_token_len, 
                 output_token_len=h,
                 backbone_type=backbone_type,
                 attn_mask=attn_mask,
                 stride=stride,
                 num_decoder_layers=num_decoder_layers,
                 proj_embd_type=proj_embd_type, 
                 proj_head_type=proj_head_type,
                 tokenizer_type=tokenizer_type,
                 padding_patch=padding_patch, 
                 pe=pe, 
                 scaler_type='standard',
                 loss=loss,
                 random_seed=random_seed,
                 alias='t5flex_standard',
                 ckpt_path=ckpt_path,
                ), #Patching
                 T5Flex(
                 h=h,
                 context_len=context_len,
                 max_steps=max_steps,
                 val_check_steps=val_check_steps,
                 early_stop_patience_steps=early_stop_patience_steps,
                 batch_size=batch_size,
                 input_token_len=input_token_len, 
                 output_token_len=h,
                 backbone_type=backbone_type,
                 attn_mask=attn_mask,
                 stride=stride,
                 num_decoder_layers=num_decoder_layers,
                 proj_embd_type=proj_embd_type, 
                 proj_head_type=proj_head_type,
                 tokenizer_type=tokenizer_type,
                 padding_patch=padding_patch, 
                 pe=pe, 
                 scaler_type='robust',
                 loss=loss,
                 random_seed=random_seed,
                 alias='t5flex_robust',
                 ckpt_path=ckpt_path,
                ), #Patching
            ]

    if args.experiment_name == 'decomp_ablation':
        tokenizer_type = 'patch_fixed_length'
        scaler_type = 'standard'
        backbone_type = "google/t5-efficient-tiny"
        attn_mask = "bidirectional"
        proj_embd_type = 'linear'
        proj_head_type = 'linear'
        pe = 'sincos_relative'  
        padding_patch = None #'end'
        num_decoder_layers = 0
        loss = MAE()
        
        models = [T5Flex(
                 h=h,
                 context_len=context_len,
                 max_steps=max_steps,
                 val_check_steps=val_check_steps,
                 early_stop_patience_steps=early_stop_patience_steps,
                 batch_size=batch_size,
                 input_token_len=input_token_len, 
                 output_token_len=h,
                 backbone_type=backbone_type,
                 attn_mask=attn_mask,
                 stride=stride,
                 num_decoder_layers=num_decoder_layers,
                 proj_embd_type=proj_embd_type,
                 proj_head_type=proj_head_type,
                 tokenizer_type=tokenizer_type,
                 padding_patch=padding_patch, 
                 pe=pe, 
                 scaler_type=scaler_type,
                 decomposition_type='dlinear_trend_seasonality',
                 moving_avg_window=25,
                 loss=loss,
                 random_seed=random_seed,
                 alias='t5flex_tsdecomp',
                 ckpt_path=ckpt_path,
                )
            ]

    if args.experiment_name == 'nont5models':
        patch_lens=[8, 16, 32, 64, 96, 128]
            
        models = [PatchTST(**Model_Configs.get_patchtst_config(
                        h=h,
                        patch_len=patch_len,
                        random_seed=random_seed,
                        ),
                    ) for patch_len in patch_lens
                 ]+\
                [DLinear(**Model_Configs.get_dlinear_config(
                        h=h,
                        random_seed=random_seed,
                        )
                    ),
                 NLinear(**Model_Configs.get_nlinear_config(
                        h=h,
                        random_seed=random_seed,
                        )
                    ),
                 MLP(**Model_Configs.get_mlp_config(
                        h=h,
                        random_seed=random_seed,
                        )
                    ),
                 NHITS(**Model_Configs.get_nhits_config(
                        h=h,
                        random_seed=random_seed,
                        )
                    ),
                 NBEATS(**Model_Configs.get_nbeats_config(
                        h=h,
                        random_seed=random_seed,
                        )
                    ),
                 TSMixer(**Model_Configs.get_tsmixer_config(
                        h=h,
                        n_series=1,
                        random_seed=random_seed,
                        )
                    ),
                 LSTM(**Model_Configs.get_lstm_config(
                        h=h,
                        random_seed=random_seed,
                        )
                    ),
                 TCN(**Model_Configs.get_tcn_config(
                        h=h,
                        random_seed=random_seed,
                        )
                    ),
                 TimesNet(**Model_Configs.get_timesnet_config(
                        h=h,
                        random_seed=random_seed,
                        )
                    ),
                 VanillaTransformer(**Model_Configs.get_vanillatransformer_config(
                        h=h,
                        random_seed=random_seed,
                        )
                    ),
                 TFT(**Model_Configs.get_tft_config(
                        h=h,
                        random_seed=random_seed,
                        )
                    ),
                 Autoformer(**Model_Configs.get_autoformer_config(
                        h=h,
                        random_seed=random_seed,
                        )
                    ),
                 Informer(**Model_Configs.get_informer_config(
                        h=h,
                        random_seed=random_seed,
                        )
                    ),
                 iTransformer(**Model_Configs.get_itransformer_config(
                        h=h,
                        n_series=1,
                        random_seed=random_seed,
                        )
                    ),
                  ]

    return models
