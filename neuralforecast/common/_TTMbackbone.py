# Copyright contributors to the TSFM project
#
# This code is based on layers and components from the PatchTSMixer model in the HuggingFace Transformers
# Library: https://github.com/huggingface/transformers/blob/main/src/transformers/models/patchtsmixer/modeling_patchtsmixer.py
"""PyTorch TinyTimeMixer model."""

import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.configuration_utils import PretrainedConfig

from ..common._positional_encodings import PositionalEncoding



# Copyright contributors to the TSFM project
#
"""TinyTimeMixer model configuration"""

from typing import Optional, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

TINYTIMEMIXER_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class TTMConfig(PretrainedConfig):

    model_type = "tinytimemixer"
    attribute_map = {
        "hidden_size": "d_model",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        # Time series specific configuration
        context_len: int = 64,
        input_token_len: int = 12,
        token_num: int = 1,
        c_in: int = 1,
        d_model: int = 128,
        expansion_factor: int = 1,
        num_layers: int = 3,
        dropout: float = 0,
        mode: str = "common_channel",
        gated_attn: bool = True,
        norm_mlp: str = "LayerNorm", #"batch"
        norm_eps: float = 1e-5,
        adaptive_patching_levels: int = 0,
        # decoder parameters
        decoder_num_layers: int = 0,
        decoder_d_model: int = 128,
        decoder_mode: str = "common_channel",
        self_attn: bool = False,
        **kwargs,
    ):

        self.context_len = context_len
        self.input_token_len = input_token_len
        self.token_num = token_num
        self.c_in = c_in
        self.d_model = d_model
        self.expansion_factor = expansion_factor
        self.num_layers = num_layers
        self.dropout = dropout
        self.mode = mode
        self.gated_attn = gated_attn
        self.norm_mlp = norm_mlp
        self.norm_eps = norm_eps
        self.adaptive_patching_levels = adaptive_patching_levels
        # decoder params
        self.decoder_num_layers = decoder_num_layers
        self.decoder_d_model = decoder_d_model
        self.decoder_adaptive_patching_levels = adaptive_patching_levels
        self.decoder_mode = decoder_mode
        self.self_attn = self_attn
        
        super().__init__(**kwargs)
  
    
@dataclass
class TinyTimeMixerEncoderOutput(ModelOutput):
    """
    Base class for `TinyTimeMixerEncoderOutput`, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, d_model)`):
            Hidden-state at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class TinyTimeMixerGatedAttention(nn.Module):
    """
    Module that applies gated attention to input data.

    Args:
        in_size (`int`): The input size.
        out_size (`int`): The output size.
    """

    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.attn_layer = nn.Linear(in_size, out_size)
        self.attn_softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        attn_weight = self.attn_softmax(self.attn_layer(inputs))
        inputs = inputs * attn_weight
        return inputs

    
class TinyTimeMixerBatchNorm(nn.Module):
    """
    Compute batch normalization over the sequence length (time) dimension.
    """

    def __init__(self, config: TTMConfig):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(config.d_model, eps=config.norm_eps)

    def forward(self, inputs: torch.Tensor):
        """
        Parameters:
            inputs (`torch.Tensor` of shape `(batch_size, sequence_length, d_model)`):
                input for Batch norm calculation
        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, d_model)`
        """
        output = inputs.transpose(1, 2)  # output: (batch_size, d_model, sequence_length)
        output = self.batchnorm(output)
        return output.transpose(1, 2)


class TinyTimeMixerNormLayer(nn.Module):
    """Normalization block

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TTMConfig):
        super().__init__()

        self.norm_mlp = config.norm_mlp

        if "batch" in config.norm_mlp.lower():
            self.norm = TinyTimeMixerBatchNorm(config)
        else:
            self.norm = nn.LayerNorm(config.d_model, eps=config.norm_eps)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                Input to the normalization layer.
        Returns:
            `torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`
        """
        if "batch" in self.norm_mlp.lower():
            # reshape the data
            inputs_reshaped = torch.reshape(
                inputs,
                (
                    inputs.shape[0] * inputs.shape[1],
                    inputs.shape[2],
                    inputs.shape[3],
                ),
            )  # inputs_reshaped: [batch_size*num_channels, num_patches, d_model]

            # inputs_reshaped: [batch_size*num_channels, num_patches, d_model]
            inputs_reshaped = self.norm(inputs_reshaped)

            # put back data to the original shape
            inputs = torch.reshape(inputs_reshaped, inputs.shape)

        else:
            inputs = self.norm(inputs)

        return inputs


class TinyTimeMixerMLP(nn.Module):
    def __init__(self, in_features, out_features, config):
        super().__init__()
        num_hidden = in_features * config.expansion_factor
        self.fc1 = nn.Linear(in_features, num_hidden)
        self.dropout1 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(num_hidden, out_features)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                Input to the MLP layer.
        Returns:
            `torch.Tensor` of the same shape as `inputs`
        """
        inputs = self.dropout1(nn.functional.gelu(self.fc1(inputs)))
        inputs = self.fc2(inputs)
        inputs = self.dropout2(inputs)
        return inputs


class TinyTimeMixerChannelFeatureMixerBlock(nn.Module):
    """This module mixes the features in the channel dimension.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TTMConfig):
        super().__init__()

        self.norm = TinyTimeMixerNormLayer(config)
        self.gated_attn = config.gated_attn
        self.mlp = TinyTimeMixerMLP(
            in_features=config.c_in,
            out_features=config.c_in,
            config=config,
        )

        if config.gated_attn:
            self.gating_block = TinyTimeMixerGatedAttention(
                in_size=config.c_in, out_size=config.c_in
            )

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                input to the MLP layer
        Returns:
            `torch.Tensor` of the same shape as `inputs`
        """
        residual = inputs
        inputs = self.norm(inputs)

        inputs = inputs.permute(0, 3, 2, 1)

        if self.gated_attn:
            inputs = self.gating_block(inputs)

        inputs = self.mlp(inputs)

        inputs = inputs.permute(0, 3, 2, 1)

        out = inputs + residual
        return out


class PatchMixerBlock(nn.Module):
    """This module mixes the patch dimension.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TTMConfig):
        super().__init__()

        self.norm = TinyTimeMixerNormLayer(config)

        self.self_attn = config.self_attn
        self.gated_attn = config.gated_attn

        self.mlp = TinyTimeMixerMLP(
            in_features=config.token_num,
            out_features=config.token_num,
            config=config,
        )

        if config.gated_attn:
            self.gating_block = TinyTimeMixerGatedAttention(in_size=config.token_num, out_size=config.token_num)

        if config.self_attn:
            self.self_attn_layer = TinyTimeMixerAttention(
                embed_dim=config.d_model,
                num_heads=config.self_attn_heads,
                dropout=config.dropout,
            )
            self.norm_attn = TinyTimeMixerNormLayer(config)

    def forward(self, hidden_state):
        """
        Args:
            hidden_state (`torch.Tensor`): Input tensor.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        residual = hidden_state

        hidden_state = self.norm(hidden_state)

        if self.self_attn:
            batch_size, n_vars, token_num, d_model = hidden_state.shape
            hidden_state_reshaped = hidden_state.reshape(batch_size * n_vars, token_num, d_model)

            x_attn, _, _ = self.self_attn_layer(hidden_state_reshaped, output_attentions=False)
            x_attn = x_attn.reshape(batch_size, n_vars, token_num, d_model)

        # Transpose so that token_num is the last dimension
        hidden_state = hidden_state.transpose(2, 3)
        hidden_state = self.mlp(hidden_state)

        if self.gated_attn:
            hidden_state = self.gating_block(hidden_state)

        # Transpose back
        hidden_state = hidden_state.transpose(2, 3)

        if self.self_attn:
            hidden_state = self.norm_attn(hidden_state + x_attn)

        out = hidden_state + residual
        return out


class FeatureMixerBlock(nn.Module):
    """This module mixes the hidden feature dimension.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.

    """

    def __init__(self, config: TTMConfig):
        super().__init__()

        self.norm = TinyTimeMixerNormLayer(config)

        self.gated_attn = config.gated_attn

        self.mlp = TinyTimeMixerMLP(
            in_features=config.d_model,
            out_features=config.d_model,
            config=config,
        )

        if config.gated_attn:
            self.gating_block = TinyTimeMixerGatedAttention(in_size=config.d_model, out_size=config.d_model)

    def forward(self, hidden: torch.Tensor):
        """
        Args:
            hidden (`torch.Tensor` of shape `(batch_size, num_patches, d_model)`):
                Input tensor to the layer.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        residual = hidden
        hidden = self.norm(hidden)
        hidden = self.mlp(hidden)

        if self.gated_attn:
            hidden = self.gating_block(hidden)

        out = hidden + residual
        return out


class TinyTimeMixerLayer(nn.Module):
    """
    The `TinyTimeMixer` layer that does all three kinds of mixing.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.

    """

    def __init__(self, config: TTMConfig):
        super().__init__()

        if config.token_num > 1:
            self.patch_mixer = PatchMixerBlock(config=config)

        self.feature_mixer = FeatureMixerBlock(config=config)

        self.mode = config.mode
        self.token_num = config.token_num
        if config.mode == "mix_channel":
            self.channel_feature_mixer = TinyTimeMixerChannelFeatureMixerBlock(config=config)

    def forward(self, hidden: torch.Tensor):
        """
        Args:
            hidden (`torch.Tensor` of shape `(batch_size, token_num, d_model)`):
                Input tensor to the layer.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        if self.mode == "mix_channel":
            hidden = self.channel_feature_mixer(hidden)

        if self.token_num > 1:
            hidden = self.patch_mixer(hidden)
        hidden = self.feature_mixer(hidden)  # hidden: (batch_size x token_num x d_model)
        return hidden


class TinyTimeMixerAdaptivePatchingBlock(nn.Module):
    """
    The `TinyTimeMixer` layer that does all three kinds of mixing.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.

    """

    def __init__(self, config: TTMConfig, adapt_patch_level: int):
        super().__init__()
        temp_config = copy.deepcopy(config)
        self.adapt_patch_level = adapt_patch_level
        adaptive_patch_factor = 2**adapt_patch_level
        self.adaptive_patch_factor = adaptive_patch_factor

        if config.d_model // self.adaptive_patch_factor <= 4:
            # do not allow reduction beyond d_model less than 4
            logger.warning(
                "Disabling adaptive patching at level %s. Either increase d_model or reduce adaptive_patching_levels"
                % (adapt_patch_level)
            )
            self.adaptive_patch_factor = 1

        if config.d_model % self.adaptive_patch_factor != 0:
            raise ValueError("d_model should be divisible by 2^i, where i varies from 0 to adaptive_patching_levels.")
        temp_config.token_num = temp_config.token_num * self.adaptive_patch_factor
        temp_config.d_model = temp_config.d_model // self.adaptive_patch_factor

        self.mixer_layers = nn.ModuleList([TinyTimeMixerLayer(temp_config) for i in range(temp_config.num_layers)])

    def forward(self, hidden: torch.Tensor):
        """
        Args:
            hidden (`torch.Tensor` of shape `(batch_size x nvars x num_patch x d_model)`):
                Input tensor to the layer.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        all_hidden_states = []
        all_hidden_states.append(hidden)
            
        hidden = torch.reshape(
            hidden,
            (
                hidden.shape[0],
                hidden.shape[1],
                hidden.shape[2] * self.adaptive_patch_factor,
                hidden.shape[3] // self.adaptive_patch_factor,
            ),
        )
        all_hidden_states.append(hidden)

        for mod in self.mixer_layers:
            hidden = mod(hidden)
            all_hidden_states.append(hidden)

        hidden = torch.reshape(
            hidden,
            (
                hidden.shape[0],
                hidden.shape[1],
                hidden.shape[2] // self.adaptive_patch_factor,
                hidden.shape[3] * self.adaptive_patch_factor,
            ),
        )
        all_hidden_states.append(hidden)

        return hidden, all_hidden_states


class TinyTimeMixerBlock(nn.Module):
    """The main computing framework of the `TinyTimeMixer` model.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TTMConfig):
        super().__init__()

        num_layers = config.num_layers

        self.adaptive_patching_levels = config.adaptive_patching_levels

        if self.adaptive_patching_levels > 0:
            self.mixers = nn.ModuleList(
                [
                    TinyTimeMixerAdaptivePatchingBlock(config=config, adapt_patch_level=i)
                    for i in reversed(range(config.adaptive_patching_levels))
                ]
            )

        else:
            self.mixers = nn.ModuleList([TinyTimeMixerLayer(config=config) for _ in range(num_layers)])

    def forward(self, hidden_state, output_hidden_states: bool = False):
        """
        Args:
            hidden_state (`torch.Tensor`): The input tensor.
            output_hidden_states (`bool`, *optional*, defaults to False.):
                Whether to output the hidden states as well.

        Returns:
            `torch.Tensor`: The embedding. `list`: List of all hidden states if `output_hidden_states` is set to
            `True`.
        """
        all_hidden_states = []

        embedding = hidden_state

        for mod in self.mixers:
            if self.adaptive_patching_levels > 0:
                embedding, hidden_states = mod(embedding)
                all_hidden_states.extend(hidden_states)
            else:
                embedding = mod(embedding)
                if output_hidden_states:
                    all_hidden_states.append(embedding)

        if output_hidden_states:
            return embedding, all_hidden_states
        else:
            return embedding, None


class TTMbackbone(nn.Module):
    """
    Encoder for TinyTimeMixer which inputs patched time-series and outputs patched embeddings.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TTMConfig):

        super().__init__()

        config = TTMConfig().from_dict(config)
        self.d_model = config.d_model
        
        # Input encoding
        self.W_P = nn.Linear(
            config.input_token_len, config.d_model
        )  # Eq 1: projection of feature vectors onto a d-dim vector space
        
        self.encoder = TinyTimeMixerBlock(config=config)

        if config.decoder_num_layers > 0: 
            decoder_config = copy.deepcopy(config)
            decoder_config.num_layers = config.decoder_num_layers
            decoder_config.d_model = config.decoder_d_model
            decoder_config.adaptive_patching_levels = config.decoder_adaptive_patching_levels
            decoder_config.mode = config.decoder_mode
            self.decoder = TinyTimeMixerBlock(decoder_config)
            
            if config.d_model != config.decoder_d_model:
                self.adapter = nn.Linear(config.d_model, config.decoder_d_model)
            else:
                self.adapter = None
        else:
            self.decoder = None

    def forward(
        self,
        x: torch.Tensor,
        output_hidden_states: Optional[bool] = False,
    ) -> Union[Tuple, TinyTimeMixerEncoderOutput]:
        r"""
        Args:
            past_values (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`):
                Context values of the time series.
                For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series,
                it is greater than 1.

            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, n_vars, num_patches, d_model)`
        """

        x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)  # x: [bs x nvars x patch_num x hidden_size]
        z, hidden_states = self.encoder(
            x,
            output_hidden_states=output_hidden_states
            )# x: [bs x nvars x patch_num x hidden_size]
        
        if self.decoder is not None:
            print('using decoder')

            decoder_input = z.clone()

            if self.adapter is not None:
                decoder_input = self.adapter(
                    decoder_input
                )  # model_output: [batch_size x nvars x num_patch x decoder_d_model]
                
            #last_hidden_state, hidden_states = self.decoder(
            z, hidden_states = self.decoder(
                hidden_state=decoder_input, 
                output_hidden_states=output_hidden_states
                )  # bs x nvars+n_cat x n_patches x d_model

        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x hidden_size x patch_num]

        return z
