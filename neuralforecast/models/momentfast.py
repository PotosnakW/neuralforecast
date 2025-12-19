import logging
import warnings
from math import ceil
from typing import Optional

import torch
from torch import nn

from ..common._base_model import BaseModel
from ..common._modules import RevINMultivariate, Flatten_Head, Patching, PositionalEncoding

from ..common._t5_infini import T5Model, T5EncoderModel
#from transformers.models.t5.modeling_t5 import T5Model, T5EncoderModel

from transformers import T5Config
from ..losses.pytorch import MAE

from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor

logger = logging.getLogger(__name__)

    
class Long_Forecaster(nn.Module):
    """
    Long-term forecaster using FAST (Factorized Action Space Tokenizer).
    
    Key differences from Long_Forecaster:
    - Uses VQ-VAE discrete tokenization instead of patching
    - Token embeddings instead of linear projection
    - No positional encodings (tokens encode temporal structure)
    - Data normalized to [-1, 1] for FAST
    - Shape without n_patch dimension
    """

    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.transformer_type = config.transformer_type
        self.h = config.h
        self.c_out = config.c_out
        self.n_series = config.n_series
        
        # FAST specific
        self.tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
        
        # Token embedding (replaces W_P projection)
        self.token_embedding = nn.Embedding(
            num_embeddings=1024,
            embedding_dim=config.hidden_size
        )
        
        # Residual dropout
        self.dropout = nn.Dropout(config.dropout)

        # Transformer backbone
        self.encoder = self._get_huggingface_transformer(config)

        # Prediction Head
        self.head = nn.Linear(config.hidden_size, 1024)

    def _get_huggingface_transformer(self, configs):
        from ..common._t5_infini import T5Model
        
        model_config = T5Config.from_pretrained(configs.transformer_backbone)

        setattr(model_config, 'infini_mixer_type', configs.infini_mixer_type)
        setattr(model_config, 'infini_channel_exclusion', configs.infini_channel_exclusion)
        setattr(model_config, 'layerwise_beta', configs.layerwise_beta)
        setattr(model_config, 'channelwise_beta', configs.channelwise_beta)
        setattr(model_config, 'use_rope', configs.use_rope)
        setattr(model_config, 'max_sequence_length', configs.input_size)  # No patching
        setattr(model_config, 'n_channels', configs.n_series)
        setattr(model_config, 'mlpmixer_hidden_size', configs.mlpmixer_hidden_size)
        setattr(model_config, 'mlpmixer_n_layers', configs.mlpmixer_n_layers)
        setattr(model_config, 'mlpmixer_dropout', configs.mlpmixer_dropout)

        transformer_backbone = T5Model(model_config)
        logging.info(f"Initializing randomly initialized transformer from {configs.transformer_backbone}. ModelClass: {T5Model.__name__}.")

        transformer_backbone = transformer_backbone.get_encoder()

        return transformer_backbone

    def forward(self, x_enc: torch.Tensor, **kwargs):
        """
        Args:
            x_enc: [batch_size x n_channels x seq_len]
        
        Returns:
            dec_out: [batch_size, n_channels, horizon*c_out]
        """
        batch_size, n_channels, seq_len = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1)  # [B, seq_len, n_channels]

        # --- Tokenize, truncate, and pad ---
        x_enc = x_enc.cpu().numpy() 
        token_lists = self.tokenizer(x_enc)

        token_tensors = [torch.tensor(t, dtype=torch.long) for t in token_lists]
        orig_lengths = [len(t) for t in token_tensors]
        padded_tokens = pad_sequence(token_tensors, batch_first=True, padding_value=0)
        padded_tokens = padded_tokens.to(self.token_embedding.weight.device)
        padded_tokens = padded_tokens.clamp(0, 1023)

        # --- Token embeddings ---
        x_embeds = self.token_embedding(padded_tokens)  # [B, token_len, hidden_size]
        x_embeds = self.dropout(x_embeds)

        # Attention mask
        attention_mask = torch.ones(batch_size, x_embeds.shape[1], device=self.token_embedding.weight.device)

        # --- Encoder ---
        outputs = self.encoder(
            inputs_embeds=x_embeds,
            attention_mask=attention_mask,
            n_channels=1
        )
        enc_out = outputs.last_hidden_state  # [B, token_len, hidden_size]

        # --- Prediction head ---
        logits = self.head(enc_out)  # [B, token_len, vocab_size]

        # --- Convert logits to token IDs ---
        predicted_token_ids = logits.argmax(dim=-1).tolist()  # [B, token_len]

        decoded = [seq[:L] for seq, L in zip(predicted_token_ids, orig_lengths)]

        # --- Decode to reconstruct actions ---
        dec_out = self.tokenizer.decode(decoded)  # [B, seq_len, n_channels]
        dec_out = torch.from_numpy(dec_out).to(padded_tokens.device).float() 

        # --- Trim to horizon ---
        dec_out = dec_out[:, :self.h, :]  # [B, horizon, n_channels]

        # --- Permute back to [B, n_channels, horizon] ---
        dec_out = dec_out.permute(0, 2, 1)  # [B, n_channels, horizon]

        return dec_out

# %% ../../nbs/models.patchtst.ipynb 17
class MOMENTFAST(BaseModel):
    """MOMENT

    **Parameters:**<br>
    `h`: int, Forecast horizon. <br>
    `context_len`: int, autorregresive inputs size, y=[1,2,3,4] input_size=2 -> y_[t-2:t]=[1,2].<br>
    `stat_exog_list`: str list, static exogenous columns.<br>
    `hist_exog_list`: str list, historic exogenous columns.<br>
    `futr_exog_list`: str list, future exogenous columns.<br>
    `exclude_insample_y`: bool=False, the model skips the autoregressive features y[t-input_size:t] if True.<br>
    `num_layers`: int, number of layers for encoder.<br>
    `num_decoder_layers`: int, number of layers for decoder.<br>
    `n_heads`: int=16, number of multi-head's attention.<br>
    `hidden_size`: int=128, units of embeddings and encoders.<br>
    `linear_hidden_size`: int=256, units of linear layer.<br>
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
        super(MOMENTFAST, self).__init__(
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

        config = SimpleNamespace(**config)
        self.h = h
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
        x_enc = x.permute(0, 2, 1) # [batch_size (B), n_series (N), input_size (L)]
        forecast = self.model(x_enc=x_enc) # [batch_size, n_series, horizon*c_out]

        forecast = forecast.view(batch_size, self.n_series, self.h, -1) # [batch_size, n_series, horizon, c_out]
        forecast = forecast.permute(0, 2, 3, 1).reshape(batch_size, self.h, -1) # [batch_size, horizon, c_out*n_series] 
        # output is expected in this shape. tsmixer and other neuralforecast multivariate models' decoder output is already in shape # [batch_size, horizon*c_out, n_series] so skipping to forecast.reshape(batch_size, self.h, -1) is valid for those models. 

        return forecast
