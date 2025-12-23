import torch
from torch import nn
from typing import Optional
from types import SimpleNamespace

from ..common._base_model import BaseModel
from ..common._timerxl_utils import TimerBlock, TimerLayer, AttentionLayer, TimeAttention
from ..common._modules import RevINMultivariate, Flatten_Head, Patching, PositionalEncoding
from ..losses.pytorch import MAE


class timerxl_backbone(nn.Module):
    """
    Timer-XL: Long-Context Transformers for Unified Time Series Forecasting 

    Paper: https://arxiv.org/abs/2410.04803
    
    GitHub: https://github.com/thuml/Timer-XL
    
    Citation: @article{liu2024timer,
        title={Timer-XL: Long-Context Transformers for Unified Time Series Forecasting},
        author={Liu, Yong and Qin, Guo and Huang, Xiangdong and Wang, Jianmin and Long, Mingsheng},
        journal={arXiv preprint arXiv:2410.04803},
        year={2024}
    }
    """
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size

        self.blocks = TimerBlock(
            attn_layers=[
                TimerLayer(
                    attention=AttentionLayer(
                        TimeAttention(
                            mask_flag=True, 
                            attention_dropout=config.dropout,
                            output_attention=False, # output_attention option returns the same thing (not implemented yet?)
                            d_model=config.hidden_size, 
                            num_heads=config.n_heads,
                            covariate=False, # config.covariate, todo: future work
                            flash_attention=False, #config.flash_attention),
                            d_keys=config.d_k,
                            use_rope=config.pe_type == 'rope',
                        ), 
                        d_model=config.hidden_size, 
                        n_heads=config.n_heads,
                        d_keys=config.d_k,
                        d_values=config.d_v,
                    ),
                    d_model=config.hidden_size,
                    d_ff=config.linear_hidden_size,
                    dropout=config.dropout,
                    activation=config.activation
                ) for l in range(config.n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.hidden_size)
        )

        self.revin = config.revin
        if config.revin:
            self.revin_layer = RevINMultivariate(
                num_features=config.n_series, 
                affine=config.revin_affine,
                subtract_last=config.revin_subtract_last,
            )

        self.padding_patch = config.padding_patch
        patch_num = int((config.input_size - config.patch_len) / config.stride + 1)
        if config.padding_patch == "end":  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, config.stride))
            patch_num += 1
        self.patch_num = patch_num

        self.tokenizer = Patching(
            patch_len=config.patch_len, 
            stride=config.stride, # Timer-XL uses step=self.input_token_len
        )

        self.W_P = nn.Linear(
            config.patch_len, config.hidden_size
        )  # Eq 1: projection of feature vectors onto a d-dim vector space

        # Positional encoding
        self.W_pos = PositionalEncoding(
            pe_type=config.pe_type,
            hidden_size=config.hidden_size,
            learn_pe=config.learn_pe,
        )
        # Residual dropout
        self.dropout = nn.Dropout(config.dropout)

        # Prediction Head
        self.head = Flatten_Head(
                multivariate_head=config.multivariate_head,
                n_vars=config.n_series,
                nf=config.hidden_size * patch_num,
                h=config.h,
                c_out=config.c_out,
                head_dropout=config.head_dropout,
            )

    def forward(self, x_enc):
        batch_size, n_channels, seq_len = x_enc.shape

        #  Normalization (applied over axis=1)
        if self.revin:
            x_enc = x_enc.permute(0, 2, 1) # [batch_size x seq_len x n_channel]
            x_enc = self.revin_layer(x_enc, "norm")
            x_enc = x_enc.permute(0, 2, 1) # [batch_size x n_channel x seq_len]
        
        # Patching
        if self.padding_patch == "end":
            x_enc = self.padding_patch_layer(x_enc) 
        x_enc = self.tokenizer(x=x_enc) # [batch_size x n_channels x n_patch x patch_len]

        # Embeddings
        x_enc = self.W_P(x_enc) # [batch_size x n_channels x n_patch x d_model]
        x_enc += self.W_pos(x_enc) # [batch_size x n_channels x n_patch x d_model]
        x_enc = self.dropout(x_enc)

        # Encoder
        x_enc = x_enc.reshape(batch_size, n_channels * self.patch_num, -1) # [batch_size x n_channels * n_patch x d_model]
        enc_out, attns = self.blocks(x_enc, n_vars=n_channels, n_tokens=self.patch_num) # [batch_size x n_channels * n_patch, d_model]

        # Decoder
        # dec_out = self.head(embed_out)  # [B, C * N, P]
        # dec_out = dec_out.reshape(B, C, N, -1).reshape(B, C, -1)  # [B, C, N * P * c_out]
        enc_out = enc_out.reshape(
            (batch_size, n_channels, self.patch_num, self.hidden_size)) # [batch_size, n_channels, n_patch, d_model]
        dec_out = self.head(enc_out) # [batch_size, n_channels, h * c_out]

        if self.revin:
            dec_out = dec_out.permute(0, 2, 1)
            dec_out = self.revin_layer(dec_out, "denorm")
            dec_out = dec_out.permute(0, 2, 1) 

        return dec_out

class TimerXL(BaseModel):

    # Class attributes
    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False
    MULTIVARIATE = True # If the model produces multivariate forecasts (True) or univariate (False)
    RECURRENT = (
        False  # If the model produces forecasts recursively (True) or direct (False)
    )

    def __init__(
        self,
        h,
        input_size,
        n_series,
        univariate=True,
        stat_exog_list=None,
        hist_exog_list=None,
        futr_exog_list=None,
        exclude_insample_y=False,
        n_layers: int = 4,
        n_heads: int = 4,
        hidden_size: int = 256,
        linear_hidden_size: int = 1024,
        d_k: int = 32,
        d_v: int = 32,
        dropout: float = 0.0,
        head_dropout: float = 0.0,
        patch_len: int = 8,
        stride: int = 8,
        revin: bool = True,
        revin_affine: bool = False,
        revin_subtract_last: bool = True,
        multivariate_head: bool = False,
        pe_type: str = "sincos",
        learn_pe: bool = False,
        padding_patch="end",
        start_padding_enabled=False,
        step_size: int = 1,
        scaler_type: str = "identity",
        activation: str = "gelu",
        random_seed: int = 1,
        drop_last_loader: bool = False,
        alias: Optional[str] = None,
        # Optimization and training
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
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        dataloader_kwargs=None,
        **trainer_kwargs
    ):
        super(TimerXL, self).__init__(
            h=h,
            input_size=input_size, 
            n_series=n_series,
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
            random_seed=random_seed,
            drop_last_loader=drop_last_loader,
            alias=alias,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            **trainer_kwargs
            )

        # Enforce correct patch_len, regardless of user input
        patch_len = min(input_size + stride, patch_len)

        config = {key: value for key, value in self.hparams.items() 
                  if key != 'loss'
                 }
        config['c_out'] = self.loss.outputsize_multiplier
        config['patch_len'] = patch_len
        config = SimpleNamespace(**config)
    
        self.h = h
        self.n_series = n_series
        self.univariate = univariate
        self.model = timerxl_backbone(config)

    def forward(self, windows_batch):
        x = windows_batch[
            "insample_y"
        ]  #   [batch_size (B), input_size (L), n_series (N)]
        B, L, N = x.shape

        if self.univariate:
            x = x.permute(2, 0, 1).reshape(N*B, L, 1)  # [B, L, N] -> [N, B, 1] -> [N*B, L, 1]

        x = x.permute(0, 2, 1) # [batch_size (B), n_series (N), input_size (L)]
        forecast = self.model(x_enc=x) # [batch_size, n_series, horizon*c_out]

        if self.univariate:
            forecast = forecast.squeeze(1).view(N, B, self.h, self.loss.outputsize_multiplier) # [n_series, batch_size, horizon, c_out]
            forecast = forecast.permute(1, 0, 2, 3) # [batch_size, n_series, horizon, c_out]
        else:
            forecast = forecast.view(B, self.n_series, self.h, -1) # [batch_size, n_series, horizon, c_out]
       
        forecast = forecast.permute(0, 2, 3, 1).reshape(B, self.h, -1) # [batch_size, horizon, c_out*n_series] 
        # output is expected in this shape. tsmixer and other neuralforecast multivariate models' decoder output is already in shape # [batch_size, horizon*c_out, n_series] so skipping to forecast.reshape(batch_size, self.h, -1) is valid for those models. 

        return forecast