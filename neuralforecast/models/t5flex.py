# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/models.patchtst.ipynb.

# %% auto 0
__all__ = ['TSTbackbone', 'Flatten_Head', 'TSTEncoder', 'T5Flex']

# %% ../../nbs/models.patchtst.ipynb 5
import math
import numpy as np
from typing import Optional  # , Any, Tuple
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common._base_flex import BaseFlex
from ..common._projections import ProjectionHead 
from ..common._T5backbone import T5backbone
from ..common._TTMbackbone import TTMbackbone

from ..losses.pytorch import MAE


# %% ../../nbs/models.patchtst.ipynb 15
class TSTbackbone(nn.Module):
    """
    TSTbackbone
    """

    def __init__(
        self,
        config: dict,
        c_in: int,
        c_out: int,
        context_len: int,
        h: int,
        input_token_len: int,
        output_token_len: int,
        lag: int,
        stride: int,
        d_model: int,
        key_padding_mask: str = "auto",
        attn_mask: str = "bidirectional",
        pe: str = "zeros",
        learn_pe: bool = True,
        head_dropout=0,
        padding_patch=None,
        head_type="flatten",
        backbone_type="T5",
        individual=False,
    ):

        super().__init__()

        # Patching
        token_num = int((context_len - input_token_len) / stride + 1)
        if padding_patch == "end":  # can be modified to general case
            token_num += 1
        config['token_num'] = token_num

        # Backbone
        if backbone_type=='T5':
            self.backbone = T5backbone(
                config,
            )
        elif backbone_type=='tsmixer':
            self.backbone = TTMbackbone(
                config
            )

        self.head_nf = d_model * token_num
        self.n_vars = c_in
        self.c_out = c_out
        self.head_type = head_type
        self.individual = individual

        proj_hd = ProjectionHead(
            self.individual,
            self.n_vars,
            self.head_nf,
            output_token_len,
            c_out,
            head_dropout,
        )

        if head_type == "flatten":
            proj_hd.flatten_head()  # Initialize layers in ProjectionHead
            self.head = proj_hd
        elif head_type == "residual_network":
            proj_hd.residual_network()  # Initialize layers in ProjectionHead
            self.head = proj_hd

    def forward(self, x):  
        #z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x patch_len x patch_num]
        
        n_vars = x.shape[1]
        if self.decoder:
            xe = x[:, :, :-1, :].clone()
            xd = x[:, :, -1, :].clone()
            xd = self.W_P(xd)  # x: [bs x nvars x patch_num x hidden_size]
        else:
            xe = x.clone()
            xd = None

        xe = self.W_P(xe)  # x: [bs x nvars x patch_num x hidden_size]
        
        # model
        z = self.backbone(xe, xd)  # z: [bs x nvars x patch_num x hidden_size]
        #embeddings = z.clone() # WILLA ADDED THIS
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x hidden_size x patch_num]
        z = self.head(z)  # z: [bs x nvars x h]

        return z
        #return z, embeddings # WILLA ADDED THIS

# %% ../../nbs/models.patchtst.ipynb 17
class T5Flex(BaseFlex):
    """PatchTST

    The PatchTST model is an efficient Transformer-based model for multivariate time series forecasting.

    It is based on two key components:
    - segmentation of time series into windows (patches) which are served as input tokens to Transformer
    - channel-independence, where each channel contains a single univariate time series.

    **Parameters:**<br>
    `h`: int, Forecast horizon. <br>
    `context_len`: int, autorregresive inputs size, y=[1,2,3,4] input_size=2 -> y_[t-2:t]=[1,2].<br>
    `stat_exog_list`: str list, static exogenous columns.<br>
    `hist_exog_list`: str list, historic exogenous columns.<br>
    `futr_exog_list`: str list, future exogenous columns.<br>
    `exclude_insample_y`: bool=False, the model skips the autoregressive features y[t-input_size:t] if True.<br>
    `encoder_layers`: int, number of layers for encoder.<br>
    `n_heads`: int=16, number of multi-head's attention.<br>
    `hidden_size`: int=128, units of embeddings and encoders.<br>
    `linear_hidden_size`: int=256, units of linear layer.<br>
    `dropout`: float=0.1, dropout rate for residual connection.<br>
    `head_dropout`: float=0.1, dropout rate for Flatten head layer.<br>
    `attn_dropout`: float=0.1, dropout rate for attention layer.<br>
    `patch_len`: int=32, length of patch. Note: patch_len = min(patch_len, input_size + stride).<br>
    `stride`: int=16, stride of patch.<br>
    `activation`: str='ReLU', activation from ['gelu','relu'].<br>
    `learn_pos_embedding`: bool=True, bool to learn positional embedding.<br>
    `loss`: PyTorch module, instantiated train loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
    `valid_loss`: PyTorch module=`loss`, instantiated valid loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
    `max_steps`: int=1000, maximum number of training steps.<br>
    `learning_rate`: float=1e-3, Learning rate between (0, 1).<br>
    `num_lr_decays`: int=-1, Number of learning rate decays, evenly distributed across max_steps.<br>
    `early_stop_patience_steps`: int=-1, Number of validation iterations before early stopping.<br>
    `val_check_steps`: int=100, Number of training steps between every validation loss check.<br>
    `batch_size`: int=32, number of different series in each batch.<br>
    `valid_batch_size`: int=None, number of different series in each validation and test batch, if None uses batch_size.<br>
    `windows_batch_size`: int=1024, number of windows to sample in each training batch, default uses all.<br>
    `inference_windows_batch_size`: int=1024, number of windows to sample in each inference batch.<br>
    `start_padding_enabled`: bool=False, if True, the model will pad the time series with zeros at the beginning, by input size.<br>
    `step_size`: int=1, step size between each window of temporal data.<br>
    `scaler_type`: str='identity', type of scaler for temporal inputs normalization see [temporal scalers](https://nixtla.github.io/neuralforecast/common.scalers.html).<br>
    `random_seed`: int, random_seed for pytorch initializer and numpy generators.<br>
    `num_workers_loader`: int=os.cpu_count(), workers to be used by `TimeSeriesDataLoader`.<br>
    `drop_last_loader`: bool=False, if True `TimeSeriesDataLoader` drops last non-full batch.<br>
    `alias`: str, optional,  Custom name of the model.<br>
    `optimizer`: Subclass of 'torch.optim.Optimizer', optional, user specified optimizer instead of the default choice (Adam).<br>
    `optimizer_kwargs`: dict, optional, list of parameters used by the user specified `optimizer`.<br>
    `lr_scheduler`: Subclass of 'torch.optim.lr_scheduler.LRScheduler', optional, user specified lr_scheduler instead of the default choice (StepLR).<br>
    `lr_scheduler_kwargs`: dict, optional, list of parameters used by the user specified `lr_scheduler`.<br>
    `**trainer_kwargs`: int,  keyword trainer arguments inherited from [PyTorch Lighning's trainer](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer).<br>

    **References:**<br>
    -[Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2022). "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"](https://arxiv.org/pdf/2211.14730.pdf)
    """

    # Class attributes
    SAMPLING_TYPE = "windows"
    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False

    def __init__(
        self,
        h,
        context_len,
        stat_exog_list=None,
        hist_exog_list=None,
        futr_exog_list=None,
        exclude_insample_y=False,
        num_layers: int = 3,
        num_decoder_layers: int = 0,
        num_heads: int = 16,
        d_model: int = 128,
        d_ff: int = 256,
        dropout: float = 0.1,
        dropout_rate: float = 0.1, ##attn_dropout
        head_dropout: float = 0.0,
        input_token_len: int = 16,
        output_token_len: int = 16,
        lag: int=1,
        stride: int = 8,
        activation: str = "gated-gelu",
        key_padding_mask: str = "auto",
        batch_normalization: bool = False,
        learn_pos_embed: bool = True,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 5000,
        learning_rate: float = 1e-4,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size=1024,
        inference_windows_batch_size: int = 1024,
        start_padding_enabled=False,
        padding_patch = None,  # Padding at the end
        step_size: int = 1,
        scaler_type: str = "identity",
        random_seed: int = 1,
        num_workers_loader: int = 0,
        drop_last_loader: bool = False,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        pe: str = "zeros", 
        learn_pe: bool = True,
        tokenizer_type = 'patch_fixed_length',
        attn_mask: str = "bidirectional",
        head_type: str = "flatten", 
        backbone_type: str = "T5",
        **trainer_kwargs
    ):
        super(T5Flex, self).__init__(
            h=h,
            context_len=context_len,
            input_token_len=input_token_len, 
            output_token_len=output_token_len, 
            stride=stride,
            hist_exog_list=hist_exog_list,
            stat_exog_list=stat_exog_list,
            futr_exog_list=futr_exog_list,
            exclude_insample_y=exclude_insample_y,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            step_size=step_size,
            scaler_type=scaler_type,
            num_workers_loader=num_workers_loader,
            drop_last_loader=drop_last_loader,
            random_seed=random_seed,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            tokenizer_type=tokenizer_type,
            lag=lag, 
            padding_patch=padding_patch,
            **trainer_kwargs
        )

        # Enforce correct patch_len, regardless of user input
        if tokenizer_type == 'lags':
            assert input_token_len==1, \
                f"Assertion failed: input_token_len={input_token_len}, expected 1"
            assert stride==1, \
                f"Assertion failed: stride={stride}, expected 1"
            assert padding_patch==None, \
                f"Assertion failed: padding_patch={padding_patch}, expected None"

        elif 'patch' in tokenizer_type:
            input_token_len = min(context_len + stride, input_token_len)
            output_token_len = min(h, output_token_len)
            
        elif tokenizer_type == 'bins':
            assert input_token_len==1, \
                f"Assertion failed: input_token_len={input_token_len}, expected 1"
            assert stride==1, \
                f"Assertion failed: stride={stride}, expected 1"
            assert padding_patch==None, \
                f"Assertion failed: padding_patch={padding_patch}, expected None"

        c_out = self.loss.outputsize_multiplier
        c_in = 1  # Always univariate
        individual = False  # Separate heads for each time series
        if tokenizer_type!='patch_adaptive_len': 
            adaptive_patching_levels = 0

        config = {key: value for key, value in self.hparams.items() 
                  if key != 'loss'}
        config['c_in'] = c_in
        config['decoder_d_model'] = d_model
        config['d_kv'] = d_model // num_heads
        config['use_cache'] = False
        config['enable_gradient_checkpointing'] = False

        self.model = TSTbackbone(
            config=config,
            c_in=c_in,
            c_out=c_out,
            context_len=context_len,
            h=h,
            input_token_len=input_token_len,
            output_token_len=output_token_len,
            lag=lag,
            stride=stride,
            d_model=d_model,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            pe=pe,
            learn_pe=learn_pos_embed,
            head_dropout=head_dropout,
            padding_patch=padding_patch,
            head_type=head_type,
            backbone_type=backbone_type,
        )

    def forward(self, x):  # x: [batch, input_size]
        
        x = self.model(x)
        #x, embeddings = self.model(x) # Willa added 
        
        return x
        #return forecast, embeddings # Willa added
