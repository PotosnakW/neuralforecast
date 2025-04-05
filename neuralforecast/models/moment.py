import logging
import warnings
from math import ceil
from typing import Optional

import torch
from torch import nn

from ..common._base_model import BaseModel
from ..common._modules import RevINMultivariate
from ..common._moment_utils import PatchEmbedding, Patching, Masking, NamespaceWithDefaults, _update_inputs, _validate_inputs

from ..common._t5_infini import T5InfiniModel, T5InfiniEncoderModel
#from transformers.models.t5.modeling_t5 import T5Model, T5EncoderModel

from transformers import T5Config
from ..losses.pytorch import MAE

logger = logging.getLogger(__name__)


class ForecastingHead(nn.Module):
    def __init__(self, 
                 head_nf: int = 768*64,
                 forecast_horizon: int = 96, 
                 c_out: int = 1,
                 head_dropout: int = 0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(head_nf, forecast_horizon * c_out)
    
    def forward(self, x, input_mask : torch.Tensor = None):
        """
        x: [batch_size x n_channels x n_patches x d_model]
        output: [batch_size x n_channels x forecast_horizon]
        """
        x = self.flatten(x)   # x: batch_size x n_channels x n_patches x d_model
        x = self.linear(x)    # x: batch_size x n_channels x n_patches*d_model
        x = self.dropout(x)   # x: batch_size x n_channels x forecast_horizon*c_out
        return x

class Long_Forecaster(nn.Module): 

    def __init__(self, config):

        super().__init__()

        # Normalization, patching and embedding
        # self.normalizer = RevIN(
        #     num_features=1, # WILLA CHECK THIS!!!
        #     affine=revin_affine
        # )

        self.d_model = config.d_model
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.transformer_type = config.transformer_type

        self.revin = config.revin
        if config.revin:
            self.revin_layer = RevINMultivariate(num_features=config.n_series, 
                                                affine=config.revin_affine,
                                                subtract_last=False,
                                               )

        self.tokenizer = Patching(
            patch_len=config.patch_len, 
            stride=config.stride,
        )
        self.patch_embedding = PatchEmbedding(
            d_model=config.d_model, 
            seq_len=config.input_size,
            patch_len=config.patch_len, 
            stride=config.stride, 
            dropout=config.dropout, 
            add_positional_embedding=True,
            value_embedding_bias=False, 
            orth_gain=1.41,
        )
        self.mask_generator = Masking(mask_ratio=0.0) # no masking for forecasting task

        # Transformer backbone
        self.encoder = self._get_huggingface_transformer(config)
        
        # Prediction Head
        num_patches = (
                (max(config.input_size, config.patch_len) - config.patch_len) 
                // config.stride + 1
        )

        head_nf = config.d_model * num_patches
        self.head = ForecastingHead(
                head_nf,
                config.h, 
                config.c_out,
                config.head_dropout,
            )

    def _get_huggingface_transformer(self, configs):
        #ModelClass, EncoderModelClass = T5Model, T5EncoderModel
        ModelClass, EncoderModelClass = T5InfiniModel, T5InfiniEncoderModel  # infini
        
        logger.info(f" ModelClass: {ModelClass.__name__}, EncoderModelClass: {EncoderModelClass.__name__}.")
            
        model_config = T5Config.from_pretrained(
            configs.transformer_backbone)

        setattr(model_config, 'infini_channel_mixing', configs.infini_channel_mixing)
        setattr(model_config, 'use_rope', configs.use_rope)
        setattr(model_config, 'max_sequence_length', configs.input_size / configs.patch_len)
        setattr(model_config, 'n_channels', configs.n_series)
      
        transformer_backbone = ModelClass(model_config)
        logging.info(f"Initializing randomly initialized\
                       transformer from {configs.transformer_backbone}.  ModelClass: {ModelClass.__name__}.")
        
        transformer_backbone = transformer_backbone.get_encoder() #check valid inputs to raise error if not encoder-only
        
        if configs.getattr('enable_gradient_checkpointing', True):
            transformer_backbone.gradient_checkpointing_enable()
            logging.info("Enabling gradient checkpointing.")
        
        return transformer_backbone

    def forward(self, 
                x_enc : torch.Tensor,
                **kwargs):
        """
        x_enc : [batch_size x n_channels x seq_len]
        input_mask : [batch_size x seq_len]
        """

        batch_size, n_channels, seq_len = x_enc.shape
        input_mask = torch.ones(batch_size, seq_len).to(x_enc.device) # [B, L]

        # Normalization 
        if self.revin: #Used default neuralforecast RevIN to simplicity/reduce modules
            x_enc = x_enc.permute(0, 2, 1) #[bs x seq_len x nvars]
            x_enc = self.revin_layer(x_enc, "norm")
            x_enc = x_enc.permute(0, 2, 1) #[bs x nvars x seq_len]
        # x_enc = self.normalizer(x=x_enc, mask=input_mask, mode='norm')
        # x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0) 
        
        # Patching and embedding
        x_enc = self.tokenizer(x=x_enc) # [batch_size x n_channels x num_patch x patch_len]
        enc_in = self.patch_embedding(x_enc, mask=torch.ones_like(input_mask))
    
        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.d_model)) # [B*C, NP, D]
        
        # Encoder
        attention_mask = Masking.convert_seq_to_patch_view(
            mask=input_mask, 
            patch_len=self.patch_len,
            stride=self.stride).repeat_interleave(n_channels, dim=0) #[B*C, NT]

        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask) 
        enc_out = outputs.last_hidden_state
        
        enc_out = enc_out.reshape(
            (-1, n_channels, n_patches, self.d_model)) 
        # [batch_size x n_channels x n_patches x d_model]

        # Decoder
        dec_out = self.head(enc_out)  # z: [batch_size x n_channels x forecast_horizon]
        
        # De-Normalization
        #dec_out = self.normalizer(x=dec_out, mode='denorm') #Used default neuralforecast RevIN to simplicity/reduce modules
        if self.revin:
            dec_out = dec_out.permute(0, 2, 1)
            dec_out = self.revin_layer(dec_out, "denorm")
            dec_out = dec_out.permute(0, 2, 1)

        return dec_out

# %% ../../nbs/models.patchtst.ipynb 17
class MOMENT(BaseModel):
    """T5Flex

    **Parameters:**<br>
    `h`: int, Forecast horizon. <br>
    `context_len`: int, autorregresive inputs size, y=[1,2,3,4] input_size=2 -> y_[t-2:t]=[1,2].<br>
    `stat_exog_list`: str list, static exogenous columns.<br>
    `hist_exog_list`: str list, historic exogenous columns.<br>
    `futr_exog_list`: str list, future exogenous columns.<br>
    `exclude_insample_y`: bool=False, the model skips the autoregressive features y[t-input_size:t] if True.<br>
    `num_layers`: int, number of layers for encoder.<br>
    `num_decoder_layers`: int, number of layers for decoder.<br>
    `num_heads`: int=16, number of multi-head's attention.<br>
    `d_model`: int=128, units of embeddings and encoders.<br>
    `d_ff`: int=256, units of linear layer.<br>
    `dropout`: float=0.1, dropout rate for residual connection.<br>
    `head_dropout`: float=0.1, dropout rate for Flatten head layer.<br>
    `attn_dropout`: float=0.1, dropout rate for attention layer.<br>
    `input_token_len`: int=32, length of input patch. Note: patch_len = min(patch_len, input_size + stride).<br>
    `output_token_len`: int=32, length of output patch prediction. Note: patch_len = min(patch_len, input_size + stride).<br>
    `stride`: int=16, stride of patch.<br>
    `pe`: str="zeros", positional encoding type.<br>
    `learn_pe`: bool=True, bool to learn positional embedding.<br>
    `decomposition_type`: str=None, input decomposition method.<br>
    `top_k`: int=5, top k basis functions for DFT-type decomposition.<br> 
    `moving_avg_window`: int=25, moving average window for moving average decomposition.<br>
    `tokenizer_type`: str='patch_fixed_length', method for input tokenization.<br>
    `lag`: int=1, lag spacing for lag tokenization method.<br>
    `attn_mask`: str="bidirectional", type of attention ['bidirectional' or 'causal'].<br>
    `proj_embd_type`: str="linear", type of input embedding layer ['linear' or 'residual'].<br>
    `proj_head_type`: str="linear", type of output projection layer ['linear' or residual'].<br>
    `backbone_type`: str="T5", model backbone type ['T5', 'google/t5-efficient-{tiny, mini, small, base}'].<br>
    `activation`: str='ReLU', activation function ['gelu','relu'].<br>
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
    EXOGENOUS_FUTR = True
    EXOGENOUS_HIST = True
    EXOGENOUS_STAT = True
    MULTIVARIATE = True  # If the model produces multivariate forecasts (True) or univariate (False)
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
        transformer_backbone = "google/t5-efficient-tiny",
        transformer_type = "encoder_only",
        randomly_initialize_backbone = True,
        infini_channel_mixing = False,
        num_layers: int = 3,
        num_decoder_layers: int = 0,
        num_heads: int = 16,
        d_model: int = 128,
        d_ff: int = 128,
        dropout: float = 0.1,
        head_dropout: float = 0.0,
        patch_len: int = 16,
        stride: int = 8,
        use_rope: bool = False,
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
        revin: str = True,
        revin_affine: str = False,
        random_seed: int = 1,
        drop_last_loader: bool = False,
        alias: Optional[str] = None,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        dataloader_kwargs=None,
        **trainer_kwargs
    ):
        super(MOMENT, self).__init__(
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
        config = _validate_inputs(config)
        self.h = h
        self.input_size = input_size
        self.n_series = n_series
        self.model = Long_Forecaster(config)

    def forward(self, windows_batch):
        # Parse windows_batch
        x = windows_batch[
            "insample_y"
        ]  #   [batch_size (B), input_size (L), n_series (N)]
        #hist_exog = windows_batch["hist_exog"]  #   [B, hist_exog_size (X), L, N]
        #futr_exog = windows_batch["futr_exog"]  #   [B, futr_exog_size (F), L + h, N]
        #stat_exog = windows_batch["stat_exog"]  #   [N, stat_exog_size (S)]

        batch_size = x.shape[0]
        x_enc = x.permute(0, 2, 1) #  [batch_size (B), n_series (N), input_size (L)]
        forecast = self.model(x_enc=x_enc) # [batch_size, n_series, horizon*c_out]
        
        forecast = forecast.view(batch_size, self.n_series, self.h, -1) # [batch_size, n_series, horizon, c_out]
        forecast = forecast.permute(0, 2, 3, 1).reshape(batch_size, self.h, -1) # [batch_size, horizon, c_out*n_series] 
        # output is expected in this shape. tsmixer and other neuralforecast multivariate models' decoder output is already in shape # [batch_size, horizon*c_out, n_series] so skipping to forecast.reshape(batch_size, self.h, -1) is valid for those models. 

        return forecast
