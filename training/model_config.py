from neuralforecast.losses.pytorch import MAE, MSE, HuberLoss, DistributionLoss

class Model_Configs:

    def __init__(self,
                 stat_exog_list=None, 
                 hist_exog_list=None, 
                 futr_exog_list=None,
                 horizon=192,
                 max_steps=10000,
                 val_check_steps=100,
                 # max_steps=5,
                 # val_check_steps=2,
                 input_size=256,
                 hidden_size=256,
                 linear_hidden_size=1024,
                 batch_size=4,
                 windows_batch_size=256,
                 inference_windows_batch_size=256,
                 learning_rate=1e-3,
                 random_seed=0,
                 num_layers=4,
                 encoder_num_layers=4,
                 decoder_num_layers=0,
                 scaler_type='standard',
                 early_stop_patience_steps=20, 
                 stride=8,
                 dropout=0,
                 n_heads=4,
                 context_size=10,
                 kernel_size=6,
                 pe='sincos',
                 learn_pe=False,
                 revin=True,
                 revin_affine=False,
                 attn_mask = 'bidirectional',
                 loss = MAE(),
                 ckpt_path=None,
                ):
        
        self.h = horizon
        self.max_steps = max_steps
        self.val_check_steps = val_check_steps
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_hidden_size = linear_hidden_size
        self.batch_size = batch_size
        self.windows_batch_size = windows_batch_size
        self.inference_windows_batch_size = inference_windows_batch_size
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.num_layers = num_layers
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.scaler_type = scaler_type
        self.stat_exog_list = stat_exog_list
        self.hist_exog_list = hist_exog_list
        self.futr_exog_list = futr_exog_list
        self.early_stop_patience_steps = early_stop_patience_steps
        self.stride=stride
        self.dropout = dropout
        self.n_heads = n_heads
        self.context_size=context_size
        self.kernel_size=kernel_size
        self.pe = pe
        self.learn_pe = learn_pe
        self.revin = revin
        self.revin_affine = revin_affine
        self.attn_mask = attn_mask
        self.loss = loss
        self.ckpt_path = ckpt_path

    @classmethod
    def get_nhits_config(cls, **kwargs):
        instance = cls()

        for key, value in kwargs.items():
            setattr(instance, key, value)

        space = {'h': instance.h,
                 'max_steps': instance.max_steps,
                 'val_check_steps': instance.val_check_steps,
                 'input_size': instance.input_size,
                 'stat_exog_list': instance.stat_exog_list,
                 'hist_exog_list': instance.hist_exog_list,
                 'futr_exog_list': instance.futr_exog_list,
                 'stack_types': instance.num_layers*['identity'],
                 'n_blocks': instance.num_layers*[1],
                 'mlp_units': instance.num_layers*[[instance.hidden_size, 
                                                    instance.hidden_size, 
                                                   ]],
                 'n_pool_kernel_size': instance.num_layers*[1],
                 'n_freq_downsample': instance.num_layers*[1],
                 'dropout_prob_theta': instance.dropout,
                 'batch_size': instance.batch_size,
                 'windows_batch_size': instance.windows_batch_size,
                 'scaler_type': instance.scaler_type,
                 'early_stop_patience_steps': instance.early_stop_patience_steps,
                 'learning_rate': instance.learning_rate,
                 'random_seed': instance.random_seed,
                 'loss': instance.loss,
                 'alias': f'nhits_is:{instance.input_size}_hs:{instance.hidden_size}_nl:{instance.num_layers}',
                }
        return space
    
    @classmethod
    def get_nbeats_config(cls, **kwargs):
        instance = cls()

        for key, value in kwargs.items():
            setattr(instance, key, value)

        space = {'h': instance.h,
                 'max_steps': instance.max_steps,
                 'val_check_steps': instance.val_check_steps,
                 'input_size': instance.input_size,
                 'stat_exog_list': instance.stat_exog_list,
                 'hist_exog_list': instance.hist_exog_list,
                 'futr_exog_list': instance.futr_exog_list,
                 'dropout_prob_theta': instance.dropout,
                 'batch_size': instance.batch_size,
                 'windows_batch_size': instance.windows_batch_size,
                 'scaler_type': instance.scaler_type,
                 'early_stop_patience_steps': instance.early_stop_patience_steps,
                 'learning_rate': instance.learning_rate,
                 'random_seed': instance.random_seed,
                 'loss': instance.loss,
                 'stack_types': instance.num_layers//2*['identity']+['trend', 'seasonality'],
                 'n_blocks': instance.num_layers*[1],
                 'mlp_units': instance.num_layers*[[instance.hidden_size, 
                                                    instance.hidden_size, 
                                                   ]],
                 'alias': f'nbeats_is:{instance.input_size}_hs:{instance.hidden_size}_nl:{instance.num_layers}',
                }
        return space
    
    @classmethod
    def get_mlp_config(cls, **kwargs):
        instance = cls()

        for key, value in kwargs.items():
            setattr(instance, key, value)

        space = {'h': instance.h,
                 'max_steps': instance.max_steps,
                 'val_check_steps': instance.val_check_steps,
                 'input_size': instance.input_size,
                 'stat_exog_list': instance.stat_exog_list,
                 'hist_exog_list': instance.hist_exog_list,
                 'futr_exog_list': instance.futr_exog_list,
                 'num_layers': instance.num_layers,
                 'hidden_size': instance.hidden_size,
                 'learning_rate': instance.learning_rate,
                 'early_stop_patience_steps': instance.early_stop_patience_steps,
                 'batch_size': instance.batch_size,
                 'windows_batch_size': instance.windows_batch_size,
                 'scaler_type': instance.scaler_type,
                 'random_seed': instance.random_seed,
                 'loss': instance.loss,
                 'ckpt_path':instance.ckpt_path,
                 'alias': f'mlp_is:{instance.input_size}_hs:{instance.hidden_size}_nl:{instance.num_layers}',
                }
        return space

    @classmethod
    def get_tft_config(cls, **kwargs):
        instance = cls()

        for key, value in kwargs.items():
            setattr(instance, key, value)

        space = {'h': instance.h,
                 'input_size': instance.input_size,
                 'max_steps': instance.max_steps,
                 'val_check_steps': instance.val_check_steps,
                 'stat_exog_list': instance.stat_exog_list,
                 'hist_exog_list': instance.hist_exog_list,
                 'futr_exog_list': instance.futr_exog_list,
                 'hidden_size': instance.hidden_size,
                 'batch_size': instance.batch_size,
                 'windows_batch_size': instance.windows_batch_size,
                 'scaler_type': instance.scaler_type,
                 'early_stop_patience_steps': instance.early_stop_patience_steps,
                 'learning_rate': instance.learning_rate,
                 'random_seed': instance.random_seed,
                 'loss': instance.loss,
                 # TFT specific parameters
                 'n_head': instance.n_heads,
                 'dropout': instance.dropout,
                 'attn_dropout': instance.dropout,
                 'ckpt_path':instance.ckpt_path,
                 'alias': f'tft_is:{instance.input_size}_hs:{instance.hidden_size}_nl:{instance.num_layers}_nheads:{instance.n_heads}',
                }
        return space
    
    @classmethod
    def get_patchtst_config(cls, **kwargs):
        instance = cls()

        for key, value in kwargs.items():
            setattr(instance, key, value)

        space = {'h': instance.h,
                 'input_size': instance.input_size,
                 'max_steps': instance.max_steps,
                 'val_check_steps': instance.val_check_steps,
                 'stat_exog_list': instance.stat_exog_list,
                 'hist_exog_list': instance.hist_exog_list,
                 'futr_exog_list': instance.futr_exog_list,
                 'hidden_size': instance.hidden_size,
                 'linear_hidden_size': instance.linear_hidden_size,
                 'batch_size': instance.batch_size,
                 'windows_batch_size': instance.windows_batch_size,
                 'inference_windows_batch_size': instance.inference_windows_batch_size,
                 'scaler_type': "identity", #instance.scaler_type,
                 'early_stop_patience_steps': instance.early_stop_patience_steps,
                 'learning_rate': instance.learning_rate,
                 'random_seed': instance.random_seed,
                 'loss': instance.loss,
                 # Patchtst specific parameters:
                 'stride': instance.stride,
                 'n_heads': instance.n_heads,
                 'dropout': instance.dropout,
                 'attn_dropout': instance.dropout,
                 'patch_len': instance.patch_len,
                 'encoder_layers':instance.encoder_num_layers,
                 'learn_pe': instance.learn_pe,
                 'pe': instance.pe,
                 'revin': instance.revin,
                 'revin_affine': instance.revin_affine,
                 #'attn_mask': instance.attn_mask,
                 'ckpt_path':instance.ckpt_path,
                 'alias': f'patchtst_is:{instance.input_size}_hs:{instance.hidden_size}_nl:{instance.encoder_num_layers}_nheads:{instance.n_heads}_pl:{instance.patch_len}_pe:zeros_revin:{instance.revin}_attn:{instance.attn_mask}',
                }
        return space
  
    @classmethod
    def get_vanillatransformer_config(cls, **kwargs):
        instance = cls()

        for key, value in kwargs.items():
            setattr(instance, key, value)

        space = {'h': instance.h,
                 'input_size': instance.input_size,
                 'max_steps': instance.max_steps,
                 'val_check_steps': instance.val_check_steps,
                 'stat_exog_list': instance.stat_exog_list,
                 'hist_exog_list': instance.hist_exog_list,
                 'futr_exog_list': instance.futr_exog_list,
                 'hidden_size': instance.hidden_size,
                 'batch_size': instance.batch_size,
                 'windows_batch_size': instance.windows_batch_size,
                 'scaler_type': instance.scaler_type,
                 'early_stop_patience_steps': instance.early_stop_patience_steps,
                 'learning_rate': instance.learning_rate,
                 'random_seed': instance.random_seed,
                 'loss': instance.loss,
                 # Vanillatransformer specific parameters:
                 'n_head': instance.n_heads,
                 'dropout': instance.dropout,
                 'ckpt_path':instance.ckpt_path,
                 'alias': f'vanillatransformer_is:{instance.input_size}_hs:{instance.hidden_size}_nl:{instance.num_layers}_nheads:{instance.n_heads}',
                }
        return space
    
    @classmethod
    def get_informer_config(cls, **kwargs):
        instance = cls()

        for key, value in kwargs.items():
            setattr(instance, key, value)

        space = {'h': instance.h,
                 'input_size': instance.input_size,
                 'max_steps': instance.max_steps,
                 'val_check_steps': instance.val_check_steps,
                 'stat_exog_list': instance.stat_exog_list,
                 'hist_exog_list': instance.hist_exog_list,
                 'futr_exog_list': instance.futr_exog_list,
                 'hidden_size': instance.hidden_size,
                 'batch_size': instance.batch_size,
                 'windows_batch_size': instance.windows_batch_size,
                 'scaler_type': instance.scaler_type,
                 'early_stop_patience_steps': instance.early_stop_patience_steps,
                 'learning_rate': instance.learning_rate,
                 'random_seed': instance.random_seed,
                 'loss': instance.loss,
                 # Informer specific parameters:
                 'n_head': instance.n_heads,
                 'dropout': instance.dropout,
                 'ckpt_path':instance.ckpt_path,
                 'alias': f'informer_is:{instance.input_size}_hs:{instance.hidden_size}_nl:{instance.num_layers}_nheads:{instance.n_heads}',
                }
        return space
    
    @classmethod
    def get_autoformer_config(cls, **kwargs):
        instance = cls()

        for key, value in kwargs.items():
            setattr(instance, key, value)

        space = {'h': instance.h,
                 'input_size': instance.input_size,
                 'max_steps': instance.max_steps,
                 'val_check_steps': instance.val_check_steps,
                 'stat_exog_list': instance.stat_exog_list,
                 'hist_exog_list': instance.hist_exog_list,
                 'futr_exog_list': instance.futr_exog_list,
                 'hidden_size': instance.hidden_size,
                 'batch_size': instance.batch_size,
                 'windows_batch_size': instance.windows_batch_size,
                 'scaler_type': instance.scaler_type,
                 'early_stop_patience_steps': instance.early_stop_patience_steps,
                 'learning_rate': instance.learning_rate,
                 'random_seed': instance.random_seed,
                 'loss': instance.loss,
                 # Autoformer specific parameters:
                 'n_head': instance.n_heads,
                 'dropout': instance.dropout,
                 'ckpt_path':instance.ckpt_path,
                 'alias': f'autoformer_is:{instance.input_size}_hs:{instance.hidden_size}_nl:{instance.num_layers}_nheads:{instance.n_heads}',
                }
        return space
    
    @classmethod
    def get_fedformer_config(cls, **kwargs):
        instance = cls()

        for key, value in kwargs.items():
            setattr(instance, key, value)

        space = {'h': instance.h,
                 'input_size': instance.input_size,
                 'max_steps': instance.max_steps,
                 'val_check_steps': instance.val_check_steps,
                 'stat_exog_list': instance.stat_exog_list,
                 'hist_exog_list': instance.hist_exog_list,
                 'futr_exog_list': instance.futr_exog_list,
                 'hidden_size': instance.hidden_size,
                 'batch_size': instance.batch_size,
                 'windows_batch_size': instance.windows_batch_size,
                 'scaler_type': instance.scaler_type,
                 'early_stop_patience_steps': instance.early_stop_patience_steps,
                 'learning_rate': instance.learning_rate,
                 'random_seed': instance.random_seed,
                 'loss': instance.loss,
                 # Fedformer specific parameters:
                 'n_head': instance.n_heads,
                 'dropout': instance.dropout,
                 'ckpt_path':instance.ckpt_path,
                 'alias': f'fedformer_is:{instance.input_size}_hs:{instance.hidden_size}_nl:{instance.num_layers}_nheads:{instance.n_heads}',
                }
        return space

    @classmethod
    def get_itransformer_config(cls, **kwargs):
        instance = cls()

        for key, value in kwargs.items():
            setattr(instance, key, value)

        space = {'h': instance.h,
                 'input_size': instance.input_size,
                 'max_steps': instance.max_steps,
                 'val_check_steps': instance.val_check_steps,
                 'stat_exog_list': instance.stat_exog_list,
                 'hist_exog_list': instance.hist_exog_list,
                 'futr_exog_list': instance.futr_exog_list,
                 'hidden_size': instance.hidden_size,
                 'batch_size': instance.batch_size,
                 'scaler_type': instance.scaler_type,
                 'early_stop_patience_steps': instance.early_stop_patience_steps,
                 'learning_rate': instance.learning_rate,
                 'random_seed': instance.random_seed,
                 'loss': instance.loss,
                 # itransformer specific parameters:
                 'n_series': instance.n_series,
                 'n_heads': instance.n_heads,
                 'dropout': instance.dropout,
                 'ckpt_path': instance.ckpt_path,
                 'alias': f'itransformer_is:{instance.input_size}_hs:{instance.hidden_size}_nl:{instance.num_layers}_nheads:{instance.n_heads}',
                }
        return space

    @classmethod
    def get_lstm_config(cls, **kwargs):
        instance = cls()

        for key, value in kwargs.items():
            setattr(instance, key, value)

        space = {'h': instance.h,
                 'input_size': instance.input_size,
                 'max_steps': instance.max_steps,
                 'val_check_steps': instance.val_check_steps,
                 'stat_exog_list': instance.stat_exog_list,
                 'hist_exog_list': instance.hist_exog_list,
                 'futr_exog_list': instance.futr_exog_list,
                 'batch_size': instance.batch_size,
                 'scaler_type': instance.scaler_type,
                 'early_stop_patience_steps': instance.early_stop_patience_steps,
                 'learning_rate': instance.learning_rate,
                 'random_seed': instance.random_seed,
                 'loss': instance.loss,
                 # lstm specific parameters:
                 'encoder_n_layers': instance.num_layers,
                 'encoder_hidden_size': instance.hidden_size,
                 'encoder_bias': True,
                 'encoder_dropout': instance.dropout,
                 'context_size': instance.context_size,
                 'decoder_hidden_size': instance.hidden_size,
                 'decoder_layers': instance.num_layers,
                 'ckpt_path':instance.ckpt_path,
                 'alias': f'lstm_is:{instance.input_size}_hs:{instance.hidden_size}_nl:{instance.num_layers}_contextsize:{instance.context_size}',
                }
        return space

    @classmethod
    def get_tcn_config(cls, **kwargs):
        instance = cls()

        for key, value in kwargs.items():
            setattr(instance, key, value)

        space = {'h': instance.h,
                 'input_size': instance.input_size,
                 'max_steps': instance.max_steps,
                 'val_check_steps': instance.val_check_steps,
                 'stat_exog_list': instance.stat_exog_list,
                 'hist_exog_list': instance.hist_exog_list,
                 'futr_exog_list': instance.futr_exog_list,
                 'batch_size': instance.batch_size,
                 'scaler_type': instance.scaler_type,
                 'early_stop_patience_steps': instance.early_stop_patience_steps,
                 'learning_rate': instance.learning_rate,
                 'random_seed': instance.random_seed,
                 'loss': instance.loss,
                 # tcn specific parameters:
                 'kernel_size': instance.kernel_size,
                 'encoder_hidden_size': instance.hidden_size,
                 'context_size': instance.context_size,
                 'decoder_hidden_size': instance.hidden_size,
                 'decoder_layers': instance.num_layers,
                 'ckpt_path':instance.ckpt_path,
                 'alias': f'tcn_is:{instance.input_size}_hs:{instance.hidden_size}_nl:{instance.num_layers}_contextsize:{instance.context_size}_kernelsize:{instance.kernel_size}',
                }
        return space

    @classmethod
    def get_timesnet_config(cls, **kwargs):
        instance = cls()

        for key, value in kwargs.items():
            setattr(instance, key, value)

        space = {'h': instance.h,
                 'input_size': instance.input_size,
                 'max_steps': instance.max_steps,
                 'val_check_steps': instance.val_check_steps,
                 'stat_exog_list': instance.stat_exog_list,
                 'hist_exog_list': instance.hist_exog_list,
                 'futr_exog_list': instance.futr_exog_list,
                 #'hidden_size': 64, #instance.hidden_size,
                 'hidden_size': instance.hidden_size,
                 'batch_size': instance.batch_size,
                 'windows_batch_size': instance.windows_batch_size,
                 'scaler_type': instance.scaler_type,
                 'early_stop_patience_steps': instance.early_stop_patience_steps,
                 'learning_rate': instance.learning_rate,
                 'random_seed': instance.random_seed,
                 'loss': instance.loss,
                 # timesnet specific parameters:
                 #'conv_hidden_size': 64, #128, #instance.hidden_size,
                 'conv_hidden_size': instance.hidden_size,
                 'num_kernels': instance.kernel_size,
                 'encoder_layers': instance.num_layers,
                 'ckpt_path':instance.ckpt_path,
                 'alias': f'timesnet_is:{instance.input_size}_hs:{instance.hidden_size}_nl:{instance.num_layers}_num_kernels:{instance.kernel_size}',
                }
        return space

    @classmethod
    def get_dlinear_config(cls, **kwargs):
        instance = cls()

        for key, value in kwargs.items():
            setattr(instance, key, value)

        space = {'h': instance.h,
                 'input_size': instance.input_size,
                 'max_steps': instance.max_steps,
                 'val_check_steps': instance.val_check_steps,
                 'stat_exog_list': instance.stat_exog_list,
                 'hist_exog_list': instance.hist_exog_list,
                 'futr_exog_list': instance.futr_exog_list,
                 'batch_size': instance.batch_size,
                 'windows_batch_size': instance.windows_batch_size,
                 'scaler_type': instance.scaler_type,
                 'early_stop_patience_steps': instance.early_stop_patience_steps,
                 'learning_rate': instance.learning_rate,
                 'random_seed': instance.random_seed,
                 'loss': instance.loss,
                 # dlinear specific parameters:
                 'moving_avg_window': 25,
                 'ckpt_path':instance.ckpt_path,
                 'alias': f'dlinear_is:{instance.input_size}_ma_window:{25}',
                }
        return space
    
    @classmethod
    def get_nlinear_config(cls, **kwargs):
        instance = cls()

        for key, value in kwargs.items():
            setattr(instance, key, value)

        space = {'h': instance.h,
                 'input_size': instance.input_size,
                 'max_steps': instance.max_steps,
                 'val_check_steps': instance.val_check_steps,
                 'stat_exog_list': instance.stat_exog_list,
                 'hist_exog_list': instance.hist_exog_list,
                 'futr_exog_list': instance.futr_exog_list,
                 'batch_size': instance.batch_size,
                 'windows_batch_size': instance.windows_batch_size,
                 'scaler_type': instance.scaler_type,
                 'early_stop_patience_steps': instance.early_stop_patience_steps,
                 'learning_rate': instance.learning_rate,
                 'random_seed': instance.random_seed,
                 'loss': instance.loss,
                 # nlinear specific parameters:
                 'ckpt_path':instance.ckpt_path,
                 'alias': f'nlinear_is:{instance.input_size}',
                }
        return space

    @classmethod
    def get_tsmixer_config(cls, **kwargs):
        instance = cls()

        for key, value in kwargs.items():
            setattr(instance, key, value)

        space = {'h': instance.h,
                 'input_size': instance.input_size,
                 'max_steps': instance.max_steps,
                 'val_check_steps': instance.val_check_steps,
                 'stat_exog_list': instance.stat_exog_list,
                 'hist_exog_list': instance.hist_exog_list,
                 'futr_exog_list': instance.futr_exog_list,
                 'batch_size': instance.batch_size,
                 'dropout': instance.dropout,
                 'scaler_type': 'identity', #instance.scaler_type,
                 'early_stop_patience_steps': instance.early_stop_patience_steps,
                 'learning_rate': instance.learning_rate,
                 'random_seed': instance.random_seed,
                 'loss': instance.loss,
                 # tsmixer specific parameters:
                 'revin': instance.revin,
                 'n_series': instance.n_series,
                 'n_block': instance.num_layers,
                 'ff_dim': instance.linear_hidden_size,
                 'ckpt_path':instance.ckpt_path,
                 'alias': f'tsmixer_is:{instance.input_size}_n_block:{instance.num_layers}',
                }
        return space
