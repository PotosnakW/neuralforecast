# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/models.vanillatransformer.ipynb.

# %% auto 0
__all__ = ['VanillaTransformer']

# %% ../../nbs/models.vanillatransformer.ipynb 5
import numpy as np
from typing import Optional

import torch
import torch.nn as nn

from neuralforecast.common._modules import (
    TransEncoderLayer,
    TransEncoder,
    TransDecoderLayer,
    TransDecoder,
    DataEmbedding,
    AttentionLayer,
    FullAttention,
)
from ..common._base_model import BaseModel

from ..losses.pytorch import MAE

# %% ../../nbs/models.vanillatransformer.ipynb 8
class VanillaTransformer(BaseModel):
    """VanillaTransformer

    Vanilla Transformer, following implementation of the Informer paper, used as baseline.

    The architecture has three distinctive features:
    - Full-attention mechanism with O(L^2) time and memory complexity.
    - An MLP multi-step decoder that predicts long time-series sequences in a single forward operation rather than step-by-step.

    The Vanilla Transformer model utilizes a three-component approach to define its embedding:
    - It employs encoded autoregressive features obtained from a convolution network.
    - It uses window-relative positional embeddings derived from harmonic functions.
    - Absolute positional embeddings obtained from calendar features are utilized.

    *Parameters:*<br>
    `h`: int, forecast horizon.<br>
    `input_size`: int, maximum sequence length for truncated train backpropagation. Default -1 uses all history.<br>
    `futr_exog_list`: str list, future exogenous columns.<br>
    `hist_exog_list`: str list, historic exogenous columns.<br>
    `stat_exog_list`: str list, static exogenous columns.<br>
        `decoder_input_size_multiplier`: float = 0.5, .<br>
    `hidden_size`: int=128, units of embeddings and encoders.<br>
    `n_head`: int=4, controls number of multi-head's attention.<br>
    `dropout`: float (0, 1), dropout throughout Informer architecture.<br>
        `conv_hidden_size`: int=32, channels of the convolutional encoder.<br>
        `activation`: str=`GELU`, activation from ['ReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'PReLU', 'Sigmoid', 'GELU'].<br>
    `encoder_layers`: int=2, number of layers for the TCN encoder.<br>
    `decoder_layers`: int=1, number of layers for the MLP decoder.<br>
    `loss`: PyTorch module, instantiated train loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
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
    `scaler_type`: str='robust', type of scaler for temporal inputs normalization see [temporal scalers](https://nixtla.github.io/neuralforecast/common.scalers.html).<br>
    `random_seed`: int=1, random_seed for pytorch initializer and numpy generators.<br>
    `drop_last_loader`: bool=False, if True `TimeSeriesDataLoader` drops last non-full batch.<br>
    `alias`: str, optional,  Custom name of the model.<br>
    `optimizer`: Subclass of 'torch.optim.Optimizer', optional, user specified optimizer instead of the default choice (Adam).<br>
    `optimizer_kwargs`: dict, optional, list of parameters used by the user specified `optimizer`.<br>
    `lr_scheduler`: Subclass of 'torch.optim.lr_scheduler.LRScheduler', optional, user specified lr_scheduler instead of the default choice (StepLR).<br>
    `lr_scheduler_kwargs`: dict, optional, list of parameters used by the user specified `lr_scheduler`.<br>
    `dataloader_kwargs`: dict, optional, list of parameters passed into the PyTorch Lightning dataloader by the `TimeSeriesDataLoader`. <br>
    `**trainer_kwargs`: int,  keyword trainer arguments inherited from [PyTorch Lighning's trainer](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer).<br>

        *References*<br>
        - [Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, Wancai Zhang. "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"](https://arxiv.org/abs/2012.07436)<br>
    """

    # Class attributes
    EXOGENOUS_FUTR = True
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False
    MULTIVARIATE = False  # If the model produces multivariate forecasts (True) or univariate (False)
    RECURRENT = (
        False  # If the model produces forecasts recursively (True) or direct (False)
    )

    def __init__(
        self,
        h: int,
        input_size: int,
        stat_exog_list=None,
        hist_exog_list=None,
        futr_exog_list=None,
        decoder_input_size_multiplier: float = 0.5,
        hidden_size: int = 128,
        dropout: float = 0.05,
        n_head: int = 4,
        conv_hidden_size: int = 32,
        activation: str = "gelu",
        encoder_layers: int = 2,
        decoder_layers: int = 1,
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
        step_size: int = 1,
        scaler_type: str = "identity",
        random_seed: int = 1,
        drop_last_loader: bool = False,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        dataloader_kwargs=None,
        **trainer_kwargs,
    ):
        super(VanillaTransformer, self).__init__(
            h=h,
            input_size=input_size,
            hist_exog_list=hist_exog_list,
            stat_exog_list=stat_exog_list,
            futr_exog_list=futr_exog_list,
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
            drop_last_loader=drop_last_loader,
            random_seed=random_seed,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            **trainer_kwargs,
        )

        # Architecture
        self.label_len = int(np.ceil(input_size * decoder_input_size_multiplier))
        if (self.label_len >= input_size) or (self.label_len <= 0):
            raise Exception(
                f"Check decoder_input_size_multiplier={decoder_input_size_multiplier}, range (0,1)"
            )

        if activation not in ["relu", "gelu"]:
            raise Exception(f"Check activation={activation}")

        self.c_out = self.loss.outputsize_multiplier
        self.output_attention = False
        self.enc_in = 1
        self.dec_in = 1

        # Embedding
        self.enc_embedding = DataEmbedding(
            c_in=self.enc_in,
            exog_input_size=self.futr_exog_size,
            hidden_size=hidden_size,
            pos_embedding=True,
            dropout=dropout,
        )
        self.dec_embedding = DataEmbedding(
            self.dec_in,
            exog_input_size=self.futr_exog_size,
            hidden_size=hidden_size,
            pos_embedding=True,
            dropout=dropout,
        )

        # Encoder
        self.encoder = TransEncoder(
            [
                TransEncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            attention_dropout=dropout,
                            output_attention=self.output_attention,
                        ),
                        hidden_size,
                        n_head,
                    ),
                    hidden_size,
                    conv_hidden_size,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(encoder_layers)
            ],
            norm_layer=torch.nn.LayerNorm(hidden_size),
        )
        # Decoder
        self.decoder = TransDecoder(
            [
                TransDecoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=True,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        hidden_size,
                        n_head,
                    ),
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        hidden_size,
                        n_head,
                    ),
                    hidden_size,
                    conv_hidden_size,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(decoder_layers)
            ],
            norm_layer=torch.nn.LayerNorm(hidden_size),
            projection=nn.Linear(hidden_size, self.c_out, bias=True),
        )

    def forward(self, windows_batch):
        # Parse windows_batch
        insample_y = windows_batch["insample_y"]
        futr_exog = windows_batch["futr_exog"]

        if self.futr_exog_size > 0:
            x_mark_enc = futr_exog[:, : self.input_size, :]
            x_mark_dec = futr_exog[:, -(self.label_len + self.h) :, :]
        else:
            x_mark_enc = None
            x_mark_dec = None

        x_dec = torch.zeros(size=(len(insample_y), self.h, 1), device=insample_y.device)
        x_dec = torch.cat([insample_y[:, -self.label_len :, :], x_dec], dim=1)

        enc_out = self.enc_embedding(insample_y, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)  # attns visualization

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        forecast = dec_out[:, -self.h :]
        return forecast
