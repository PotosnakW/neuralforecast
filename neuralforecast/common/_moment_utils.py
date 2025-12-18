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
    if configs.hidden_size is None and configs.transformer_backbone in SUPPORTED_HUGGINGFACE_MODELS: 
        configs.hidden_size = get_huggingface_model_dimensions(configs.transformer_backbone)
        logging.info("Setting hidden_size to {}".format(configs.d_model))
    elif configs.hidden_size is None:
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
