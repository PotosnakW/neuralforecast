import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Union

from transformers.models.t5.modeling_t5 import T5Stack, T5Block, T5LayerNorm, T5Model, T5Config, T5EncoderModel, T5LayerCrossAttention, T5LayerSelfAttention, T5LayerFF

from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
    TokenClassifierOutput,
)

from ._modules import MLP


class T5Attention(nn.Module): # Default T5Attention copied from HuggingFace for version control
    def __init__(
        self,
        config: T5Config,
        has_relative_attention_bias=False,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.layer_idx = layer_idx
        if layer_idx is None and self.is_decoder:
            logger.warning_once(
                f"Instantiating a decoder {self.__class__.__name__} without passing `layer_idx` is not recommended and "
                "will to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None, cache_position=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        if cache_position is None:
            context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        else:
            context_position = cache_position[:, None].to(device)
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        n_channels,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, 1, 1, key_length) (non-causal encoder) or (batch_size, 1, seq_length, key_length) (causal decoder)
        batch_size, seq_length = hidden_states.shape[:2]

        # if key_value_states are provided this layer is used as a cross-attention layer for the decoder
        is_cross_attention = key_value_states is not None

        query_states = self.q(hidden_states)
        query_states = query_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                curr_past_key_value = past_key_value.cross_attention_cache
            else:
                curr_past_key_value = past_key_value.self_attention_cache

        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_value is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = curr_past_key_value.key_cache[self.layer_idx]
            value_states = curr_past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self.k(current_states)
            value_states = self.v(current_states)
            key_states = key_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
            value_states = value_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention:
                    past_key_value.is_updated[self.layer_idx] = True

        if position_bias is None:
            key_length = key_states.shape[-2]
            # cache position is 0-indexed so we add 1 to get the real length of queries (aka with past)
            real_seq_length = query_length if query_length is not None else cache_position[-1] + 1
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, seq_length, key_length), device=hidden_states.device, dtype=hidden_states.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=hidden_states.device, cache_position=cache_position
                )
                position_bias = position_bias[:, :, -seq_length:, :]

            if mask is not None:
                # causal_mask = mask[:, :, :, : key_states.shape[-2]]
                # position_bias = position_bias + causal_mask
                mask = mask.view(batch_size, 1, 1, key_states.shape[-2])
                mask = (1.0 - mask.float()) * -1e9
                position_bias = position_bias + mask

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        # compute scores, equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        scores = torch.matmul(query_states, key_states.transpose(-1, -2)) # same thing: torch.matmul(query_states, key_states.transpose(3, 2))
        # Below scaling not included in the T5 version but we added it to make implementation closer to original attention by Vaswani et al.
        scores = scores / math.sqrt(self.key_value_proj_dim)# [batch_size, n_heads, n_patch, n_patch]
        scores += position_bias_masked

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = torch.matmul(attn_weights, value_states)

        outputs = (attn_output, past_key_value, position_bias)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs, None
    
class T5InfiniAttention(T5Attention):
    def __init__(self,
        config: T5Config,
        has_relative_attention_bias=False,
        layer_idx: Optional[int] = None,
    ):
        super().__init__(config, has_relative_attention_bias, layer_idx)
        
        self.use_rope = config.use_rope
        self.elu = nn.ELU()

        if config.infini_channel_exclusion:
            self._update_memory_matrix = self._update_memory_matrix_channelexl
            self._retrieve_from_memory = self._retrieve_from_memory_channelexl
        else:
            self._update_memory_matrix = self._update_memory_matrix_allchannels
            self._retrieve_from_memory = self._retrieve_from_memory_allchannels

    def compute_bias(self, query_length, key_length, device=None, cache_position=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        if cache_position is None:
            context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        else:
            context_position = cache_position[:, None].to(device)
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0).unsqueeze(0)  # shape (1, 1, num_heads, query_length, key_length) --> NEW: added dimension=1 for n_channels
        return values
    
    def _update_memory_matrix_allchannels(self, key_states, value_states, n_channels):
        sigma_k = self.elu(key_states) + 1.0  # [batch_size, n_channels, n_heads, n_patch, dim]
        sigma_k_transposed = sigma_k.transpose(-2, -1) # [batch_size, n_channels, n_heads, dim, n_patch]

        memory_matrix = torch.matmul(sigma_k_transposed, value_states).sum(dim=1).unsqueeze(1) # [batch_size, 1, n_heads, dim, dim] sum over channels then unsqueeze to enable broadcasting over channels
        
        z = sigma_k.sum(dim=-2).unsqueeze(-1).sum(dim=1) # [batch_size, n_heads, dim, 1] sum over sequence length and channels
        z = z.unsqueeze(dim=1) # [batch_size, 1, n_heads, dim, 1]
        
        return memory_matrix, z
    
    def _retrieve_from_memory_allchannels(self, query_states, memory_matrix, z_excluded):
        sigma_q = self.elu(query_states) + 1.0  # [B, C, H, P, D]
        numerator = sigma_q @ memory_matrix         # [B, C, H, P, D]
        denominator = (sigma_q @ z_excluded) + 1e-6 # [B, C, H, P, 1]
        A_mem = numerator / denominator             # [B, C, H, P, D]
    
        return A_mem
    
    def _update_memory_matrix_channelexl(self, key_states, value_states, n_channels):
        # σ_k = elu(k) + 1
        sigma_k = self.elu(key_states) + 1.0  # [B, C, H, P, D]
        sigma_k_T = sigma_k.transpose(-2, -1)  # [B, C, H, D, P]

        # Compute per-channel memory matrices
        memory_matrix = torch.matmul(sigma_k_T, value_states)  # [B, C, H, D, D]

        # Build exclusion mask
        channel_mask = torch.ones(n_channels, n_channels, device=sigma_k.device)
        channel_mask.fill_diagonal_(0)  # 0 for the excluded channel
        channel_mask = channel_mask.view(1, n_channels, n_channels, 1, 1, 1)

        # Exclude each channel from memory_matrix
        memory_matrix_exp = memory_matrix.unsqueeze(1)  # [B, 1, C, H, D, D]
        memory_matrix_masked = memory_matrix_exp * channel_mask  # [B, C, C, H, D, D]
        memory_matrix_summed = memory_matrix_masked.sum(dim=2)   # [B, C, H, D, D]

        # ---- Compute z_excluded (matching the exclusion logic) ----
        sigma_k_sum = sigma_k.sum(dim=-2)  # sum over patch dim -> [B, C, H, D]
        sigma_k_exp = sigma_k_sum.unsqueeze(1)  # [B, 1, C, H, D]
        sigma_k_masked = sigma_k_exp * channel_mask.squeeze(-1)  # [B, C, C, H, D]
        z_excluded = sigma_k_masked.sum(dim=2).unsqueeze(-1)  # [B, C, H, D, 1]

        return memory_matrix_summed, z_excluded

    def _retrieve_from_memory_channelexl(self, query_states, memory_matrix, z_excluded):
        sigma_q = self.elu(query_states) + 1.0  # [B, C, H, P, D]
        numerator = sigma_q @ memory_matrix         # [B, C, H, P, D]
        denominator = (sigma_q @ z_excluded) + 1e-6 # [B, C, H, P, 1]
        A_mem = numerator / denominator             # [B, C, H, P, D]
    
        return A_mem

    def forward(
        self,
        n_channels,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, 1, 1, key_length) (non-causal encoder) or (batch_size, 1, n_patch, key_length) (causal decoder)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        # if key_value_states are provided this layer is used as a cross-attention layer for the decoder
        is_cross_attention = key_value_states is not None

        query_states = self.q(hidden_states)
        query_states = query_states.view(batch_size, 
                                         -1, 
                                         self.n_heads, 
                                         self.key_value_proj_dim).transpose(1, 2)  # [batch_size, n_heads, n_patch, dim]
        query_states = query_states.view(batch_size//n_channels, 
                                         n_channels, 
                                         self.n_heads, 
                                         seq_length,
                                         self.key_value_proj_dim) # [batch_size, n_channels, n_heads, n_patch, dim]

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                curr_past_key_value = past_key_value.cross_attention_cache
            else:
                curr_past_key_value = past_key_value.self_attention_cache

        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_value is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = curr_past_key_value.key_cache[self.layer_idx]
            value_states = curr_past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self.k(current_states)
            value_states = self.v(current_states)
            key_states = key_states.view(batch_size, 
                                         -1, 
                                         self.n_heads, 
                                         self.key_value_proj_dim).transpose(1, 2)
            key_states = key_states.view(batch_size//n_channels, 
                                         n_channels, 
                                         self.n_heads, 
                                         seq_length,
                                         self.key_value_proj_dim) # [batch_size, n_channels, n_heads, n_patch, dim]
            value_states = value_states.view(batch_size, 
                                             -1, 
                                             self.n_heads, 
                                             self.key_value_proj_dim).transpose(1, 2)
            value_states = value_states.view(batch_size//n_channels, 
                                             n_channels, 
                                             self.n_heads, 
                                             seq_length, 
                                             self.key_value_proj_dim) # [batch_size, n_channels, n_heads, n_patch, dim]

            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention:
                    past_key_value.is_updated[self.layer_idx] = True

        if position_bias is None:
            key_length = key_states.shape[-2]
            # cache position is 0-indexed so we add 1 to get the real length of queries (aka with past)
            real_seq_length = query_length if query_length is not None else cache_position[-1] + 1
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, 1, self.n_heads, seq_length, key_length), device=hidden_states.device, dtype=hidden_states.dtype
                ) # NEW: added dim(1) for n_channels
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=hidden_states.device, cache_position=cache_position
                )
                position_bias = position_bias[:, :, :, -seq_length:, :]

            if mask is not None:
                #causal_mask = mask[:, :, :, :, : key_states.shape[-2]]
                #position_bias = position_bias + causal_mask
                mask = mask.view(batch_size//n_channels, n_channels, 1, 1, key_states.shape[-2])
                mask = (1.0 - mask.float()) * -1e9
                position_bias = position_bias + mask

        if self.pruned_heads:
            head_mask = torch.ones(position_bias.shape[1])
            head_mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, head_mask.bool()]
        else:
            position_bias_masked = position_bias

        # compute scores, equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        scores = torch.matmul(query_states, key_states.transpose(-1, -2)) # [batch_size, n_channels, n_heads, n_patch, n_patch]
        # Below scaling not included in the T5 version but we added it to make implementation closer to original attention by Vaswani et al.
        scores = scores / math.sqrt(self.key_value_proj_dim)# [batch_size, n_channels, n_heads, n_patch, n_patch]
        scores += position_bias_masked # [batch_size, n_channels, n_heads, n_patch, n_patch]
    
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores) # [batch_size, n_channels, n_heads, n_patch, n_patch]
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training) # [batch_size, n_channels, n_heads, n_patch, n_patch]

        # Infini attention computation across channels
        memory_matrix, z = self._update_memory_matrix(key_states, value_states, n_channels)
        A_mem = self._retrieve_from_memory(query_states, memory_matrix, z)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = torch.matmul(attn_weights, value_states) # [batch_size, n_channels, n_heads, n_patch, dim]

        outputs = (attn_output, past_key_value, position_bias)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs, A_mem

class T5LayerSelfAttention(nn.Module):
    def __init__(self, 
                 config, 
                 has_relative_attention_bias=False, 
                 layer_idx: Optional[int] = None, 
                 beta: Optional[torch.tensor] = None, 
        ):
        super().__init__()

        if config.infini_mixer_type.lower() == 'betas':
            self.SelfAttention = T5InfiniAttention(
                config, has_relative_attention_bias=has_relative_attention_bias, layer_idx=layer_idx,
            )
            if beta is not None:
                self.beta = beta
            else:
                if config.channelwise_beta:
                    self.beta = nn.Parameter(torch.rand((1, config.n_channels, config.num_heads, 1, 1))*1e-2)
                else:
                    self.beta = nn.Parameter(torch.rand((1, 1, config.num_heads, 1, 1))*1e-2)
                # Adjust the values to ensure they sum to 0
                with torch.no_grad():
                    self.beta -= self.beta.mean(dim=2, keepdim=True)

        elif config.infini_mixer_type.lower() == 'mlp':
            self.SelfAttention = T5InfiniAttention(
                config, has_relative_attention_bias=has_relative_attention_bias, layer_idx=layer_idx,
            )
            self.mlp = MLP(
                in_features=config.d_kv * 2,
                out_features=config.d_kv,
                activation='ReLU',
                hidden_size=config.mlpmixer_hidden_size,
                num_layers=config.mlpmixer_num_layers,
                dropout=config.mlpmixer_dropout,
            )
        elif config.infini_mixer_type.lower() == 'none':
            self.SelfAttention = T5Attention(
                config, has_relative_attention_bias=has_relative_attention_bias, layer_idx=layer_idx
            )
        else:
            raise Exception(f"Channel mixing method: {config.infini_mixer_type} is not an option. Please use one of 'betas', 'mlp', or 'none'.")
    
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.inner_dim = config.num_heads * config.d_kv
        self.o = nn.Linear(self.inner_dim, config.d_model, bias=False)
        self.infini_mixer_type = config.infini_mixer_type

    def forward(
        self,
        n_channels,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_outputs, a_mem = self.SelfAttention(
            n_channels=n_channels,
            hidden_states=normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        if self.infini_mixer_type == 'mlp':
            # Concatenate memory and attention outputs
            attn_output = torch.cat([a_mem, attention_outputs[0]], dim=-1)  # [batch_size, n_channels, n_heads, n_patch, dim*2]
            attn_output = self.mlp(attn_output)  # [batch_size, n_channels, n_heads, n_patch, dim*2]
            attn_output = attn_output.transpose(2, 3).contiguous() # [batch_size, n_channels, n_patch, n_heads, dim]

        elif self.infini_mixer_type == 'betas':
            attn_output = torch.sigmoid(self.beta) * a_mem + (1 - torch.sigmoid(self.beta)) * attention_outputs[0] 
            attn_output = attn_output.transpose(2, 3).contiguous() # [batch_size, n_channels, n_patch, n_heads, dim]

        elif self.infini_mixer_type == 'none':
            attn_output = attention_outputs[0] # [batch_size, n_heads, n_patch, dim]
            attn_output = attn_output.transpose(1, 2).contiguous() # [batch_size, n_patch, n_heads, dim]

        else:
            raise Exception(f'infini_mixer_type {self.infini_mixer_type} not recognized.')
        
        attn_output = attn_output.view(hidden_states.shape[0], -1, self.inner_dim) # [batch_size*n_channels, n_patch, n_heads*dim]
        attn_output = self.o(attn_output) # [batch_size*n_channels, n_patch, n_heads*dim]

        hidden_states = hidden_states + self.dropout(attn_output)
        outputs = (hidden_states,) + attention_outputs[1:]  # add attentions if we output them
        return outputs
    
class T5LayerCrossAttention(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None, beta: Optional[torch.tensor] = None):
        super().__init__()

        if config.infini_mixer_type.lower() == 'betas':
            self.SelfAttention = T5InfiniAttention(
                config, has_relative_attention_bias=False, layer_idx=layer_idx,
            )
            if beta is not None:
                self.beta = beta
            else:
                if config.channelwise_beta:
                    self.beta = nn.Parameter(torch.rand((1, config.n_channels, config.num_heads, 1, 1))*1e-2)
                else:
                    self.beta = nn.Parameter(torch.rand((1, 1, config.num_heads, 1, 1))*1e-2)
                # Adjust the values to ensure they sum to 0
                with torch.no_grad():
                    self.beta -= self.beta.mean(dim=2, keepdim=True)

        elif config.infini_mixer_type.lower() == 'mlp':
            self.SelfAttention = T5InfiniAttention(
                config, has_relative_attention_bias=False, layer_idx=layer_idx,
            )
            self.mlp = MLP(
                in_features=config.d_kv * 2,
                out_features=config.d_kv,
                activation='ReLU',
                hidden_size=config.mlpmixer_hidden_size,
                num_layers=config.mlpmixer_num_layers,
                dropout=config.mlpmixer_dropout,
            )
        elif config.infini_mixer_type.lower() == 'none':
            self.SelfAttention = T5Attention(
                config, has_relative_attention_bias=False, layer_idx=layer_idx
            )
        else:
            raise Exception(f"Channel mixing method: {config.infini_mixer_type} is not an option. Please use one of 'betas', 'mlp', or 'none'.")
    
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.inner_dim = config.num_heads * config.d_kv
        self.o = nn.Linear(self.inner_dim, config.d_model, bias=False)
        self.infini_mixer_type = config.infini_mixer_type
    
    def forward(
        self,
        n_channels,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_values=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
        cache_position=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_outputs, a_mem = self.EncDecAttention(
            n_channels=n_channels,
            hidden_states=normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_values,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        if self.infini_mixer_type == 'mlp':
            # Concatenate memory and attention outputs
            attn_output = torch.cat([a_mem, attention_outputs[0]], dim=-1)  # [batch_size, n_channels, n_heads, n_patch, dim*2]
            attn_output = self.mlp(attn_output)  # [batch_size, n_channels, n_heads, n_patch, dim*2]
            attn_output = attn_output.transpose(2, 3).contiguous() # [batch_size, n_channels, n_patch, n_heads, dim]

        elif self.infini_mixer_type == 'betas':
            attn_output = torch.sigmoid(self.beta) * a_mem + (1 - torch.sigmoid(self.beta)) * attention_outputs[0] 
            attn_output = attn_output.transpose(2, 3).contiguous() # [batch_size, n_channels, n_patch, n_heads, dim]
        
        elif self.infini_mixer_type == 'none':
            attn_output = attention_outputs[0] # [batch_size, n_heads, n_patch, dim]
            attn_output = attn_output.transpose(1, 2).contiguous() # [batch_size, n_patch, n_heads, dim]
        
        else:
            raise Exception(f'infini_mixer_type {self.infini_mixer_type} not recognized.')
        
        attn_output = attn_output.view(hidden_states.shape[0], -1, self.inner_dim) # [batch_size*n_channels, n_patch, n_heads*dim]
        attn_output = self.o(attn_output) # [batch_size*n_channels, n_patch, n_heads*dim]

        layer_output = hidden_states + self.dropout(attn_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs
        
class T5Block(T5Block):
    def __init__(self, 
                 config, 
                 has_relative_attention_bias=False, 
                 layer_idx: Optional[int] = None, 
                 beta: Optional[torch.tensor] = None
                 ):
        super().__init__(config)
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(
            T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias, layer_idx=layer_idx, beta=beta)
        )
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config, layer_idx=layer_idx))

        self.layer.append(T5LayerFF(config))

    #@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        n_channels,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        cache_position=None,
    ):
        self_attention_outputs = self.layer[0](
            n_channels=n_channels,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states = self_attention_outputs[0]
        attention_outputs = self_attention_outputs[1:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            cross_attention_outputs = self.layer[1](
                n_channels=n_channels,
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_values=past_key_values,
                query_length=cache_position[-1] + 1,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[1:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        return (
            outputs + attention_outputs
        )  # hidden-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)

class T5Stack(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
    
        # check on beta initialization --> people have used zeros and random, which one is best?
        if config.layerwise_beta:
            beta = None
        else:
            n_channels = config.n_channels
            n_heads = config.num_heads
            # Create a layer-specific beta
            if config.channelwise_beta:
                beta = nn.Parameter(torch.rand((1, n_channels, n_heads, 1, 1))*1e-2)
            else:
                beta = nn.Parameter(torch.rand((1, 1, n_heads, 1, 1))*1e-2)
            # Adjust the values to ensure they sum to 0
            with torch.no_grad():
                beta -= beta.mean(dim=2, keepdim=True)

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0), layer_idx=i, beta=beta) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def forward(
        self,
        n_channels=None,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        if self.is_decoder:
            if use_cache and past_key_values is None:
                if self.config.is_encoder_decoder:
                    past_key_values = EncoderDecoderCache(
                        DynamicCache(config=self.config), DynamicCache(config=self.config)
                    )
                else:
                    past_key_values = DynamicCache(config=self.config)
        elif not self.is_decoder:
            # do not pass cache object down the line for encoder stack
            # it messes indexing later in decoder-stack because cache object is modified in-place
            past_key_values = None

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, device=inputs_embeds.device
            )

        if attention_mask is None and not is_torchdynamo_compiling():
            # required mask seq length can be calculated via length of past cache
            mask_seq_length = past_key_values_length + seq_length
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

        if self.config.is_decoder:
            causal_mask = self._update_causal_mask(
                attention_mask,
                inputs_embeds,
                cache_position,
                past_key_values.self_attention_cache
                if isinstance(past_key_values, EncoderDecoderCache)
                else past_key_values,
                output_attentions,
            )
        elif attention_mask is not None:
            causal_mask = attention_mask[:, None, None, :]
            causal_mask = causal_mask.to(dtype=inputs_embeds.dtype)
            causal_mask = (1.0 - causal_mask) * torch.finfo(inputs_embeds.dtype).min
        else:
            causal_mask = None

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device, dtype=torch.long
                )
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, layer_module in enumerate(self.block):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if causal_mask is not None:
                    causal_mask = causal_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                n_channels=n_channels,
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,  # as a positional argument for gradient checkpointing
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                return_dict=return_dict,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[1]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[3 if output_attentions else 2]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[4],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    past_key_values,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

class T5Model(T5Model):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
    
    #@auto_docstring
    def forward(
        self,
        n_channels, 
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[tuple[torch.FloatTensor], Seq2SeqModelOutput]:
        """
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for detail.

            [What are input IDs?](../glossary#input-ids)

            To know more on how to prepare `input_ids` for pretraining take a look a [T5 Training](./t5#training).
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            T5 uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

            To know more on how to prepare `decoder_input_ids` for pretraining take a look at [T5
            Training](./t5#training).
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        decoder_head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        cross_attn_head_mask (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
            `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, T5Model

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = T5Model.from_pretrained("google-t5/t5-small")

        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

        >>> # preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
        >>> # This is not needed for torch's T5ForConditionalGeneration as it does this internally using labels arg.
        >>> decoder_input_ids = model._shift_right(decoder_input_ids)

        >>> # forward pass
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                n_channels=n_channels,
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )