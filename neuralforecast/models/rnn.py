# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/models.rnn.ipynb.

# %% auto 0
__all__ = ['RNN']

# %% ../../nbs/models.rnn.ipynb 6
from typing import Optional

import torch
import torch.nn as nn

from ..losses.pytorch import MAE
from ..common._base_recurrent import BaseRecurrent
from ..common._modules import MLP
from ..common._modules import Concentrator

# %% ../../nbs/models.rnn.ipynb 7
class RNN(BaseRecurrent):
    """RNN

    Multi Layer Elman RNN (RNN), with MLP decoder.
    The network has `tanh` or `relu` non-linearities, it is trained using
    ADAM stochastic gradient descent. The network accepts static, historic
    and future exogenous data.

    **Parameters:**<br>
    `h`: int, forecast horizon.<br>
    `input_size`: int, maximum sequence length for truncated train backpropagation. Default -1 uses all history.<br>
    `encoder_n_layers`: int=2, number of layers for the RNN.<br>
    `encoder_hidden_size`: int=200, units for the RNN's hidden state size.<br>
    `encoder_activation`: str=`tanh`, type of RNN activation from `tanh` or `relu`.<br>
    `encoder_bias`: bool=True, whether or not to use biases b_ih, b_hh within RNN units.<br>
    `encoder_dropout`: float=0., dropout regularization applied to RNN outputs.<br>
    `context_size`: int=10, size of context vector for each timestamp on the forecasting window.<br>
    `decoder_hidden_size`: int=200, size of hidden layer for the MLP decoder.<br>
    `decoder_layers`: int=2, number of layers for the MLP decoder.<br>
    `futr_exog_list`: str list, future exogenous columns.<br>
    `hist_exog_list`: str list, historic exogenous columns.<br>
    `stat_exog_list`: str list, static exogenous columns.<br>
    `loss`: PyTorch module, instantiated train loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
    `valid_loss`: PyTorch module=`loss`, instantiated valid loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
    `max_steps`: int=1000, maximum number of training steps.<br>
    `learning_rate`: float=1e-3, Learning rate between (0, 1).<br>
    `num_lr_decays`: int=-1, Number of learning rate decays, evenly distributed across max_steps.<br>
    `early_stop_patience_steps`: int=-1, Number of validation iterations before early stopping.<br>
    `val_check_steps`: int=100, Number of training steps between every validation loss check.<br>
    `batch_size`: int=32, number of differentseries in each batch.<br>
    `valid_batch_size`: int=None, number of different series in each validation and test batch.<br>
    `scaler_type`: str='robust', type of scaler for temporal inputs normalization see [temporal scalers](https://nixtla.github.io/neuralforecast/common.scalers.html).<br>
    `random_seed`: int=1, random_seed for pytorch initializer and numpy generators.<br>
    `num_workers_loader`: int=os.cpu_count(), workers to be used by `TimeSeriesDataLoader`.<br>
    `drop_last_loader`: bool=False, if True `TimeSeriesDataLoader` drops last non-full batch.<br>
    `**trainer_kwargs`: int,  keyword trainer arguments inherited from [PyTorch Lighning's trainer](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer).<br>
    """
    
    # Class attributes
    SAMPLING_TYPE = "recurrent"

    def __init__(
        self,
        h: int,
        input_size: int = -1,
        inference_input_size: int = -1,
        encoder_n_layers: int = 2,
        encoder_hidden_size: int = 200,
        encoder_activation: str = "tanh",
        encoder_bias: bool = True,
        encoder_dropout: float = 0.0,
        context_size: int = 10,
        decoder_hidden_size: int = 200,
        decoder_layers: int = 2,
        futr_exog_list=None,
        hist_exog_list=None,
        stat_exog_list=None,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size=32,
        valid_batch_size: Optional[int] = None,
        scaler_type: str = "robust",
        random_seed=1,
        num_workers_loader=0,
        drop_last_loader=False,
        # New parameters
        use_concentrator: bool = False,
        concentrator_type: str = None,
        n_series: int = 1,
        treatment_var_name: str = "treatment",
        init_ka1: float = 1.5,
        init_ka2: float = 1.5,
        freq: int = 1,
        **trainer_kwargs,
    ):
        super(RNN, self).__init__(
            h=h,
            input_size=input_size,
            inference_input_size=inference_input_size,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            scaler_type=scaler_type,
            futr_exog_list=futr_exog_list,
            hist_exog_list=hist_exog_list,
            stat_exog_list=stat_exog_list,
            num_workers_loader=num_workers_loader,
            drop_last_loader=drop_last_loader,
            random_seed=random_seed,
            **trainer_kwargs,
        )

        # ------------------ Concentrator ------------------
        # Asserts
        # if use_concentrator:
        #     assert (
        #         treatment_var_name in hist_exog_list
        #     ), f"Variable {treatment_var_name} not found in hist_exog_list!"
        #     assert (
        #         hist_exog_list[-1] == treatment_var_name
        #     ), f"Variable {treatment_var_name} must be the last element of hist_exog_list!"

        self.use_concentrator = use_concentrator

        if self.use_concentrator:
            self.concentrator = Concentrator(
                n_series=n_series,
                type=concentrator_type,
                treatment_var_name=treatment_var_name,
                init_ka1 =init_ka1,
                init_ka2= init_ka2,
                input_size=input_size,
                freq=freq,
                h=h,
                mask_future=False,
            )
        else:
            self.concentrator = None
        # --------------------------------------------------

        # RNN
        self.encoder_n_layers = encoder_n_layers
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_activation = encoder_activation
        self.encoder_bias = encoder_bias
        self.encoder_dropout = encoder_dropout

        # Context adapter
        self.context_size = context_size

        # MLP decoder
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers

        self.futr_exog_size = len(self.futr_exog_list)
        self.hist_exog_size = len(self.hist_exog_list)
        self.stat_exog_size = len(self.stat_exog_list)

        # RNN input size (1 for target variable y)
        input_encoder = 1 + self.hist_exog_size + self.stat_exog_size

        # Instantiate model
        self.hist_encoder = nn.RNN(
            input_size=input_encoder,
            hidden_size=self.encoder_hidden_size,
            num_layers=self.encoder_n_layers,
            nonlinearity=self.encoder_activation,
            bias=self.encoder_bias,
            dropout=self.encoder_dropout,
            batch_first=True,
        )

        # Context adapter
        self.context_adapter = nn.Linear(
            in_features=self.encoder_hidden_size + self.futr_exog_size * h,
            out_features=self.context_size * h,
        )

        # Decoder MLP
        self.mlp_decoder = MLP(
            in_features=self.context_size + self.futr_exog_size,
            out_features=self.loss.outputsize_multiplier,
            hidden_size=self.decoder_hidden_size,
            num_layers=self.decoder_layers,
            activation="ReLU",
            dropout=0.0,
        )

    def forward(self, windows_batch):

        # Parse windows_batch
        encoder_input = windows_batch["insample_y"]  # [B, seq_len, 1]
        futr_exog = windows_batch["futr_exog"]
        hist_exog = windows_batch["hist_exog"]
        stat_exog = windows_batch["stat_exog"]
        batch_idx = windows_batch["batch_idx"]

        # Concatenate y, historic and static inputs
        # [B, C, seq_len, 1] -> [B, seq_len, C]
        # Contatenate [ Y_t, | X_{t-L},..., X_{t} | S ]
        batch_size, seq_len = encoder_input.shape[:2]
        if self.hist_exog_size > 0:
            hist_exog = hist_exog.permute(0, 2, 1, 3).squeeze(
                -1
            )  # [B, X, seq_len, 1] -> [B, seq_len, X]
            if self.use_concentrator:
                hist_exog = self.concentrator(
                    treatment_exog=hist_exog, idx=batch_idx
                )
            encoder_input = torch.cat((encoder_input, hist_exog), dim=2)

        if self.stat_exog_size > 0:
            stat_exog = stat_exog.unsqueeze(1).repeat(
                1, seq_len, 1
            )  # [B, S] -> [B, seq_len, S]
            encoder_input = torch.cat((encoder_input, stat_exog), dim=2)

        # RNN forward
        hidden_state, _ = self.hist_encoder(
            encoder_input
        )  # [B, seq_len, rnn_hidden_state]

        if self.futr_exog_size > 0:
            futr_exog = futr_exog.permute(0, 2, 3, 1)[
                :, :, 1:, :
            ]  # [B, F, seq_len, 1+H] -> [B, seq_len, H, F]
            hidden_state = torch.cat(
                (hidden_state, futr_exog.reshape(batch_size, seq_len, -1)), dim=2
            )

        # Context adapter
        context = self.context_adapter(hidden_state)
        context = context.reshape(batch_size, seq_len, self.h, self.context_size)

        # Residual connection with futr_exog
        if self.futr_exog_size > 0:
            context = torch.cat((context, futr_exog), dim=-1)

        # Final forecast
        output = self.mlp_decoder(context)
        output = self.loss.domain_map(output)

        return output