import torch
from torch import nn
from typing import Optional

from ..common._base_model import BaseModel
from ..common._timerxl_utils import TimerBlock, TimerLayer, AttentionLayer, TimeAttention
from ..common._modules import RevINMultivariate, Flatten_Head, Patching, PositionalEncoding
from ..losses.pytorch import MAE

from ..common._moment_utils import _update_inputs


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
    
        self.blocks = TimerBlock(
            [
                TimerLayer(
                    AttentionLayer(
                        TimeAttention(True, attention_dropout=config.dropout,
                                    output_attention=False, # output_attention option returns the same thing (not implemented yet?)
                                    d_model=config.hidden_size, num_heads=config.n_heads,
                                    covariate=False, # config.covariate, todo: future work
                                    flash_attention=False), #config.flash_attention),
                                    config.hidden_size, config.n_heads),
                    config.hidden_size,
                    config.linear_hidden_size,
                    dropout=config.dropout,
                    activation=config.activation
                ) for l in range(config.n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.hidden_size)
        )

        self.revin = config.revin
        if config.revin:
            self.revin_layer = RevINMultivariate(num_features=config.n_series, 
                                                 affine=config.revin_affine,
                                                 subtract_last=False,
                                                )

        self.padding_patch = config.padding_patch
        patch_num = int((config.input_size - config.patch_len) / config.stride + 1)
        if config.padding_patch == "end":  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, config.stride))
            patch_num += 1
        self.patch_num = patch_num

        self.tokenizer = Patching(
            patch_len=config.patch_len, 
            stride=config.stride,
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
        head_nf = config.hidden_size * patch_num
        self.head = Flatten_Head(
                multivariate_head=config.multivariate_head,
                n_vars=config.n_series,
                nf=head_nf,
                h=config.h,
                c_out=config.c_out,
                head_dropout=config.head_dropout,
            )

    def forward(self, x):
        B, C, L = x.shape

        # if self.use_norm:
        #     means = x.mean(-1, keepdim=True).detach()
        #     x = x - means
        #     stdev = torch.sqrt(
        #         torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5)
        #     x /= stdev
        if self.revin: #Used default neuralforecast RevIN to simplicity/reduce modules
            x = x.permute(0, 2, 1) #[bs x seq_len x nvars]
            x = self.revin_layer(x, "norm")
            x = x.permute(0, 2, 1) #[bs x nvars x seq_len]
    
        # x = x.unfold(
        #     dimension=-1, size=self.input_token_len, step=self.input_token_len) # [B, C, N, P]

        # Patching
        if self.padding_patch == "end":
            x = self.padding_patch_layer(x)
        x = self.tokenizer(x=x) # [batch_size x n_channels x n_patch x patch_len]
        N = x.shape[2]

        embed_out = self.W_P(x) # [B, C, N, D]
        embed_out = self.dropout(embed_out + self.W_pos(embed_out)) # [B, C, N, D]

        # Encoder
        embed_out = embed_out.reshape(B, C * N, -1) # [B, C * N, D]
        embed_out, attns = self.blocks(embed_out, n_vars=C, n_tokens=N)

        # Decoder
        dec_out = self.head(embed_out)  # [B, C * N, P]
        dec_out = dec_out.reshape(B, C, N, -1).reshape(B, C, -1)  # [B, C, N * P * c_out]

        # if self.use_norm:
        #     dec_out = dec_out * stdev + means

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
        stat_exog_list=None,
        hist_exog_list=None,
        futr_exog_list=None,
        exclude_insample_y=False,
        # Transformer / Mixer config
        infini_mixer_type: str = "none",
        infini_channel_exclusion: bool = False,
        layerwise_beta: bool = True,
        channelwise_beta: bool = False,
        transformer_backbone: str = "google/t5-efficient-tiny",
        transformer_type: str = "encoder_only",
        randomly_initialize_backbone: bool = True,
        n_layers: int = 4,
        num_decoder_layers: int = 0,
        n_heads: int = 16,
        hidden_size: int = 128,
        linear_hidden_size: int = 128,
        d_k: int = 32,
        d_v: int = 32,
        dropout: float = 0.1,
        head_dropout: float = 0.0,
        patch_len: int = 16,
        stride: int = 8,
        use_rope: bool = False,
        mlpmixer_hidden_size: int = 128,
        mlpmixer_n_layers: int = 3,
        mlpmixer_dropout: float = 0.1,
        multivariate_head: bool = False,
        pe_type: str = "sincos",
        learn_pe: bool = False,
        use_pca_adapter: bool = False,
        pca_n_series: int = 2,
        padding_patch="end",
        start_padding_enabled=False,
        step_size: int = 1,
        scaler_type: str = "identity",
        revin: str = True,
        revin_affine: str = False,
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

        config = {key: value for key, value in self.hparams.items() 
                  if key != 'loss'
                 }
        config['c_out'] = self.loss.outputsize_multiplier

        config = _update_inputs(config)
        self.h = h
        self.n_series = n_series
        self.model = timerxl_backbone(config)

    def forward(self, windows_batch):
        x = windows_batch[
            "insample_y"
        ]  #   [batch_size (B), input_size (L), n_series (N)]
        #hist_exog = windows_batch["hist_exog"]  #   [B, hist_exog_size (X), L, N]
        #futr_exog = windows_batch["futr_exog"]  #   [B, futr_exog_size (F), L + h, N]
        #stat_exog = windows_batch["stat_exog"]  #   [N, stat_exog_size (S)]

        batch_size = x.shape[0]
        x = x.permute(0, 2, 1) # [batch_size (B), n_series (N), input_size (L)]
        forecast = self.model(x=x) # [batch_size, n_series, horizon*c_out]

        forecast = forecast.view(batch_size, self.n_series, self.h, -1) # [batch_size, n_series, horizon, c_out]
        forecast = forecast.permute(0, 2, 3, 1).reshape(batch_size, self.h, -1) # [batch_size, horizon, c_out*n_series] 
        # output is expected in this shape. tsmixer and other neuralforecast multivariate models' decoder output is already in shape # [batch_size, horizon*c_out, n_series] so skipping to forecast.reshape(batch_size, self.h, -1) is valid for those models. 

        return self.forecast(x)