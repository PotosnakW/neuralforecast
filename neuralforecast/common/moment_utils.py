import math
import warnings
from argparse import Namespace
from dataclasses import dataclass
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Type
 
SUPPORTED_HUGGINGFACE_MODELS = [
    't5-small', 't5-base', 't5-large', 't5-3b', 't5-11b',
    'google/flan-t5-small', 'google/flan-t5-base', 
    'google/flan-t5-large', 'google/flan-t5-xl', 
    'google/flan-t5-xxl',
    'google/t5-efficient-tiny', 'google/t5-efficient-mini',
    'google/t5-efficient-small', 'google/t5-efficient-medium',
    'google/t5-efficient-large', 'google/t5-efficient-base',
]


class NamespaceWithDefaults(Namespace):
    @classmethod
    def from_namespace(cls, namespace):
        new_instance = cls()

        if isinstance(namespace, dict):
            # Handle the case where namespace is a dictionary
            for key, value in namespace.items():
                setattr(new_instance, key, value)
                
        elif isinstance(namespace, Namespace):
            # Handle the case where namespace is a Namespace object
            for attr in dir(namespace):
                if not attr.startswith('__'):
                    setattr(new_instance, attr, getattr(namespace, attr))
                    
        return new_instance
    
    def getattr(self, key, default=None):
        return getattr(self, key, default)

def get_huggingface_model_dimensions(model_name : str = "flan-t5-base"):
    from transformers import T5Config
    config = T5Config.from_pretrained(model_name)
    return config.d_model

def _update_inputs(configs: Namespace | dict, **kwargs) -> NamespaceWithDefaults:
    if isinstance(configs, dict) and 'model_kwargs' in kwargs:
        return NamespaceWithDefaults(**{**configs, **kwargs['model_kwargs']})
    else:
        return NamespaceWithDefaults.from_namespace(configs)

def _validate_inputs(configs: NamespaceWithDefaults) -> NamespaceWithDefaults:
    if configs.transformer_backbone == "PatchTST" and configs.transformer_type != "encoder_only":
        warnings.warn("PatchTST only supports encoder-only transformer backbones.")
        configs.transformer_type = "encoder_only"
    if configs.transformer_backbone != "PatchTST" and configs.transformer_backbone not in SUPPORTED_HUGGINGFACE_MODELS:
        raise NotImplementedError(f"Transformer backbone {configs.transformer_backbone} not supported."
                                    f"Please choose from {SUPPORTED_HUGGINGFACE_MODELS} or PatchTST.")
    if configs.d_model is None and configs.transformer_backbone in SUPPORTED_HUGGINGFACE_MODELS: 
        configs.d_model = get_huggingface_model_dimensions(configs.transformer_backbone)
        logging.info("Setting d_model to {}".format(configs.d_model))
    elif configs.d_model is None:
        raise ValueError("d_model must be specified if transformer backbone \
                            unless transformer backbone is a Huggingface model.")
        
    if configs.transformer_type not in ["encoder_only", "decoder_only", "encoder_decoder"]:
        raise ValueError("transformer_type must be one of ['encoder_only', 'decoder_only', 'encoder_decoder']")

    if configs.stride != configs.patch_len:
        warnings.warn("Patch stride length is not equal to patch length.")

    return configs

class Masking:
    def __init__(self, 
                 mask_ratio : float = 0.3,
                 patch_len : int = 8,
                 stride : Optional[int] = None):
        """
        Indices with 0 mask are hidden, and with 1 are observed.
        """
        self.mask_ratio = mask_ratio    
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
    
    @staticmethod
    def convert_seq_to_patch_view(mask: torch.Tensor,
                                patch_len: int = 8,
                                stride: Optional[int] = None,
                                multivariate: bool = False):
        """
        Input:
            mask : torch.Tensor of shape [batch_size x seq_len] or [batch_size x channels x seq_len]
            patch_len : int, length of each patch
            stride : int, step size between patches
            multivariate : bool, flag to indicate if the input is multivariate
        Output:
            mask : torch.Tensor of shape [batch_size x n_patches] or [batch_size x channels x n_patches]
        """
        stride = patch_len if stride is None else stride

        if multivariate:
            # Process multivariate case
            batch_size, n_channels, seq_len = mask.shape
            mask = mask.unfold(dimension=-1, size=patch_len, step=stride)
            # mask : [batch_size x channels x n_patches x patch_len]
            return (mask.sum(dim=-1) == patch_len).long()
        else:
            # Process univariate case
            mask = mask.unfold(dimension=-1, size=patch_len, step=stride)
            return (mask.sum(dim=-1) == patch_len).long()
    
    @staticmethod
    def convert_patch_to_seq_view(mask : torch.Tensor,
                                  patch_len : int = 8,):
        """
        Input:
            mask : torch.Tensor of shape [batch_size x n_patches]
        Output:
            mask : torch.Tensor of shape [batch_size x seq_len]
        """
        return mask.repeat_interleave(patch_len, dim=-1)
    
    def generate_mask(self, x : torch.Tensor, input_mask : Optional[torch.Tensor] = None, multichannel: bool = False):
        """
        Input: 
            x : torch.Tensor of shape 
            [batch_size x n_channels x n_patches x patch_len] or
            [batch_size x n_channels x seq_len]
            input_mask: torch.Tensor of shape [batch_size x seq_len] or
            [batch_size x n_patches]
        Output:
            mask : torch.Tensor of shape [batch_size x seq_len]
        """
        if multichannel and x.ndim == 4:
            return self._mask_patch_view_multichannel(x, input_mask=input_mask)
        elif multichannel and x.ndim == 3:
            return self._mask_seq_view_multichnnel(x, input_mask=input_mask)
        elif x.ndim == 4:
            return self._mask_patch_view(x, input_mask=input_mask)
        elif x.ndim == 3:
            return self._mask_seq_view(x, input_mask=input_mask)
    
    def _mask_patch_view(self, x, input_mask=None):
        """
        Input: 
            x : torch.Tensor of shape 
            [batch_size x n_channels x n_patches x patch_len] 
            input_mask: torch.Tensor of shape [batch_size x seq_len]
        Output:
            mask : torch.Tensor of shape [batch_size x n_patches]
        """
        input_mask = self.convert_seq_to_patch_view(input_mask, self.patch_len, self.stride)
        n_observed_patches = input_mask.sum(dim=-1, keepdim=True) # batch_size x 1

        batch_size, n_channels, n_patches, _ = x.shape
        len_keep = torch.ceil(n_observed_patches * (1 - self.mask_ratio)).long()
        noise = torch.rand(batch_size, n_patches, device=x.device)  # noise in [0, 1], batch_size x n_channels x n_patches
        noise = torch.where(input_mask == 1, noise, torch.ones_like(noise))  # only keep the noise of observed patches
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # Ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1) # ids_restore: [batch_size x n_patches]

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.zeros([batch_size, n_patches], device=x.device) # mask: [batch_size x n_patches]
        for i in range(batch_size):
            mask[i, :len_keep[i]] = 1
        
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore) 

        return mask.long()
    
    def _mask_patch_view_multichannel(self, x, input_mask=None):
        """
        Input: 
            x : torch.Tensor of shape 
            [batch_size x n_channels x n_patches x patch_len] 
            input_mask: torch.Tensor of shape [batch_size x seq_len]
        Output:
            mask : torch.Tensor of shape [batch_size x n_channels x n_patches]
        """
        
        input_mask = self.convert_seq_to_patch_view(input_mask, self.patch_len, self.stride)
        n_observed_patches = input_mask.sum(dim=-1, keepdim=True) # batch_size x 1
        batch_size, n_channels, n_patches, _ = x.shape
        #for ech batch make it so that it is [bs, n_channels, n_patches]
        input_mask = input_mask.unsqueeze(1).repeat(1, n_channels, 1)
        
        #v = input_mask.sum(dim=-1, keepdim=True) # batch_size x 1

        len_keep = torch.ceil(n_observed_patches * (1 - self.mask_ratio)).long()
        noise = torch.rand(batch_size, n_channels, n_patches, device=x.device)  # noise in [0, 1], batch_size x n_channels x n_patches
        noise = torch.where(input_mask == 1, noise, torch.ones_like(noise))  # only keep the noise of observed patches
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=2)  # Ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=2) # ids_restore: [batch_size x n_patches]

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.zeros([batch_size, n_channels, n_patches], device=x.device) # mask: [batch_size x n_channels x n_patches]
        for i in range(batch_size):
            for j in range(n_channels):
                mask[i, j, :len_keep[i, 0]] = 1  # Use len_keep[i, 0] to get the correct length for each batch

        
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=2, index=ids_restore) 

        return mask.long()
    
    def _mask_seq_view(self, x, input_mask=None):
        """
        Input: 
            x : torch.Tensor of shape 
            [batch_size x n_channels x seq_len]
            input_mask: torch.Tensor of shape [batch_size x seq_len]
        Output:
            mask : torch.Tensor of shape [batch_size x seq_len]
        """
        x = x.unfold(dimension=-1, 
                     size=self.patch_len, 
                     step=self.stride)
        mask = self._mask_patch_view(x, input_mask=input_mask)
        return self.convert_patch_to_seq_view(mask, self.patch_len).long()
    
    def _mask_seq_view_multichnnel(self, x, input_mask=None):
        """
        Input: 
            x : torch.Tensor of shape 
            [batch_size x n_channels x seq_len]
            input_mask: torch.Tensor of shape [batch_size x seq_len]
        Output:
            mask : torch.Tensor of shape [batch_size x seq_len]
        """
        x = x.unfold(dimension=-1, 
                     size=self.patch_len, 
                     step=self.stride)
        mask = self._mask_patch_view_multichannel(x, input_mask=input_mask) # 1s indicated the model seeing a patch not the patch being masked
        return self.convert_patch_to_seq_view(mask, self.patch_len).long()


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000, model_name="MOMENT"):
        super(PositionalEmbedding, self).__init__()
        self.model_name = model_name
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.model_name == "MOMENT"\
              or self.model_name == "TimesNet"\
                  or self.model_name == "GPT4TS":
            return self.pe[:, :x.size(2)]
        else:
            return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
            kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # x = x.permute(0, 2, 1) 
        x = self.tokenConv(x)
        x = x.transpose(1, 2)
        # batch_size x seq_len x d_model
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, model_name, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model, model_name=model_name)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
        d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)

class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, 
                 d_model : int = 768, 
                 seq_len : int = 512,
                 patch_len : int = 8, 
                 stride : int = 8, 
                 dropout : int = 0.1,
                 add_positional_embedding : bool = False, 
                 value_embedding_bias : bool = False,
                 orth_gain : float = 1.41):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.seq_len = seq_len
        self.stride = stride
        self.d_model = d_model
        self.add_positional_embedding = add_positional_embedding
        
        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=value_embedding_bias)
        self.mask_embedding = nn.Parameter(torch.zeros(d_model))
        # nn.init.trunc_normal_(self.mask_embedding, mean=0.0, std=.02)
        
        if orth_gain is not None:
            torch.nn.init.orthogonal_(self.value_embedding.weight, gain=orth_gain)
            if value_embedding_bias: 
                self.value_embedding.bias.data.zero_()
            # torch.nn.init.orthogonal_(self.mask_embedding, gain=orth_gain) # Fails
        
        # Positional embedding
        if self.add_positional_embedding:
            self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                x : torch.Tensor, 
                mask : torch.Tensor = None) -> torch.Tensor:
        """
        Input:
            x : [batch_size x n_channels x n_patches x patch_len]
            mask : [batch_size x seq_len] v [batch_size x n_channels x seq_len]
        Output:
            x : [batch_size x n_channels x n_patches x d_model]
        """

        mask = Masking.convert_seq_to_patch_view(
            mask, patch_len=self.patch_len).unsqueeze(-1)
        # mask : [batch_size x n_patches x 1]
        n_channels = x.shape[1]
        mask = mask.repeat_interleave(
            self.d_model, dim=-1)
        
        if mask.ndim == 3:
            mask =mask.unsqueeze(1).repeat(1, n_channels, 1, 1)
        # mask : [batch_size x n_channels x n_patches x d_model]
                        
        # Input encoding
        x = mask * self.value_embedding(x) + (1 - mask) * self.mask_embedding
        if self.add_positional_embedding:
            x = x + self.position_embedding(x)
        
        return self.dropout(x)
    
class Patching(nn.Module):
    def __init__(self, 
                 patch_len : int, 
                 stride : int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        if self.stride != self.patch_len:
            warnings.warn("Stride and patch length are not equal. \
                          This may lead to unexpected behavior.")

    def forward(self, x):
        x = x.unfold(dimension=-1, 
                     size=self.patch_len, 
                     step=self.stride)
        # x : [batch_size x n_channels x num_patch x patch_len]
        return x 
