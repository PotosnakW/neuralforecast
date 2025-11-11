import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional

from transformers.models.t5.modeling_t5 import T5Stack, T5Block, T5LayerNorm, T5Model, T5Config, T5EncoderModel, T5LayerCrossAttention, T5LayerSelfAttention, T5LayerFF

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
)

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
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

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

        # compute scores, equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        if position_bias is None:
            key_length = key_states.shape[-2]
            # cache position is 0-indexed so we add 1 to get the real length of queries (aka with past)
            real_seq_length = query_length if query_length is not None else cache_position[-1] + 1
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=scores.device, cache_position=cache_position
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

        scores += position_bias_masked

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.inner_dim)
        attn_output = self.o(attn_output)

        outputs = (attn_output, past_key_value, position_bias)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs
    
class T5InfiniAttention(T5Attention):
    def __init__(self,
        config: T5Config,
        has_relative_attention_bias=False,
        layer_idx: Optional[int] = None,
        beta: Optional[torch.tensor] = None,
    ):
        super().__init__(config, has_relative_attention_bias, layer_idx)
        
        self.use_rope = config.use_rope
        self.elu = nn.ELU()
        self.n_channels = config.n_channels

        if beta is not None:
            self.beta = beta
        else:
            if config.channelwise_beta:
                self.beta = nn.Parameter(torch.rand((1, self.n_channels, self.n_heads, 1, 1))*1e-2)
            else:
                self.beta = nn.Parameter(torch.rand((1, 1, self.n_heads, 1, 1))*1e-2)
            # Adjust the values to ensure they sum to 0
            with torch.no_grad():
                self.beta -= self.beta.mean(dim=2, keepdim=True)

    def compute_bias(self, query_length, key_length, device=None, cache_position=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        if cache_position is None:
            context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        else:
            context_position = cache_position[:, None].to(device)
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]

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
    
    def _update_memory_matrix(self, key_states, value_states):
        sigma_k = self.elu(key_states) + 1.0  # [batch_size, n_channels, n_heads, n_patch, dim]
        sigma_k_transposed = sigma_k.transpose(-2, -1) # [batch_size, n_channels, n_heads, dim, n_patch]

        memory_matrix = torch.matmul(sigma_k_transposed, value_states).sum(dim=1).unsqueeze(1) # [batch_size, 1, n_heads, dim, dim] sum over channels then unsqueeze to enable broadcasting over channels
        
        z = sigma_k.sum(dim=-2).unsqueeze(-1).sum(dim=1) # [batch_size, n_heads, dim, 1] sum over sequence length and channels
        z = z.unsqueeze(dim=1) # [batch_size, 1, n_heads, dim, 1]
        
        return memory_matrix, z
    
    def _retrieve_from_memory(self, query_states, memory_matrix, z_excluded):
        sigma_q = self.elu(query_states) + 1.0  # [B, C, H, P, D]
        numerator = sigma_q @ memory_matrix         # [B, C, H, P, D]
        denominator = (sigma_q @ z_excluded) + 1e-6 # [B, C, H, P, 1]
        A_mem = numerator / denominator             # [B, C, H, P, D]
    
        return A_mem

    def forward(
        self,
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
        query_states = query_states.view(batch_size//self.n_channels, 
                                         self.n_channels, 
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
            key_states = key_states.view(batch_size//self.n_channels, 
                                         self.n_channels, 
                                         self.n_heads, 
                                         seq_length,
                                         self.key_value_proj_dim) # [batch_size, n_channels, n_heads, n_patch, dim]
            value_states = value_states.view(batch_size, 
                                             -1, 
                                             self.n_heads, 
                                             self.key_value_proj_dim).transpose(1, 2)
            value_states = value_states.view(batch_size//self.n_channels, 
                                             self.n_channels, 
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
                mask = mask.view(batch_size//self.n_channels, self.n_channels, 1, 1, key_states.shape[-2])
                mask = (1.0 - mask.float()) * -1e9
                position_bias = position_bias + mask

        if self.pruned_heads:
            head_mask = torch.ones(position_bias.shape[1])
            head_mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, head_mask.bool()]
        else:
            position_bias_masked = position_bias
        
        # Infini attention computation across channels
        memory_matrix, z = self._update_memory_matrix(key_states, value_states)
        A_mem = self._retrieve_from_memory(query_states, memory_matrix, z)

        scores = query_states @ key_states.transpose(-2, -1) # [batch_size, n_channels, n_heads, n_patch, n_patch]
        scores += position_bias_masked # [batch_size, n_channels, n_heads, n_patch, n_patch]
        
        scores = scores / torch.sqrt(torch.tensor(self.key_value_proj_dim, 
                                                  device=hidden_states.device, 
                                                  dtype=torch.float16
                                                  )
        ) # [batch_size, n_channels, n_heads, n_patch, n_patch]

        attn_weights = F.softmax(scores, dim=-1) # [batch_size, n_channels, n_heads, n_patch, n_patch]
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training) # [batch_size, n_channels, n_heads, n_patch, n_patch]

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = attn_weights @ value_states # [batch_size, n_channels, n_heads, n_patch, dim]

        attn_output = F.sigmoid(self.beta) * A_mem + (1 - F.sigmoid(self.beta)) * attn_output # [batch_size, n_channels, n_heads, n_patch, dim]

        attn_output = attn_output.transpose(2, 3).contiguous() # [batch_size, n_channels, n_patch, n_heads, dim]
        attn_output = attn_output.view(batch_size, -1, self.inner_dim) # [batch_size*n_channels, n_patch, n_heads*dim]
        attn_output = self.o(attn_output) # [batch_size*n_channels, n_patch, n_heads*dim]

        outputs = (attn_output, past_key_value, position_bias)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

class T5InfiniChannelExclusionAttention(T5Attention):
    def __init__(self,
        config: T5Config,
        has_relative_attention_bias=False,
        layer_idx: Optional[int] = None,
        beta: Optional[torch.tensor] = None,
    ):
        super().__init__(config, has_relative_attention_bias, layer_idx)
        
        self.use_rope = config.use_rope
        self.elu = nn.ELU()
        self.n_channels = config.n_channels

        if beta is not None:
            self.beta = beta
        else:
            if config.channelwise_beta:
                self.beta = nn.Parameter(torch.rand((1, self.n_channels, self.n_heads, 1, 1))*1e-2)
            else:
                self.beta = nn.Parameter(torch.rand((1, 1, self.n_heads, 1, 1))*1e-2)
            # Adjust the values to ensure they sum to 0
            with torch.no_grad():
                self.beta -= self.beta.mean(dim=2, keepdim=True)

    def compute_bias(self, query_length, key_length, device=None, cache_position=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        if cache_position is None:
            context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        else:
            context_position = cache_position[:, None].to(device)
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]

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
    
    def _update_memory_matrix(self, key_states, value_states):
        # σ_k = elu(k) + 1
        sigma_k = self.elu(key_states) + 1.0  # [B, C, H, P, D]
        sigma_k_T = sigma_k.transpose(-2, -1)  # [B, C, H, D, P]

        # Compute per-channel memory matrices
        memory_matrix = torch.matmul(sigma_k_T, value_states)  # [B, C, H, D, D]

        # Build exclusion mask
        channel_mask = torch.ones(self.n_channels, self.n_channels, device=sigma_k.device)
        channel_mask.fill_diagonal_(0)  # 0 for the excluded channel
        channel_mask = channel_mask.view(1, self.n_channels, self.n_channels, 1, 1, 1)

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

    def _retrieve_from_memory(self, query_states, memory_matrix, z_excluded):
        sigma_q = self.elu(query_states) + 1.0  # [B, C, H, P, D]
        numerator = sigma_q @ memory_matrix         # [B, C, H, P, D]
        denominator = (sigma_q @ z_excluded) + 1e-6 # [B, C, H, P, 1]
        A_mem = numerator / denominator             # [B, C, H, P, D]
    
        return A_mem

    def forward(
        self,
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
        query_states = query_states.view(batch_size//self.n_channels, 
                                         self.n_channels, 
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
            key_states = key_states.view(batch_size//self.n_channels, 
                                         self.n_channels, 
                                         self.n_heads, 
                                         seq_length,
                                         self.key_value_proj_dim) # [batch_size, n_channels, n_heads, n_patch, dim]
            value_states = value_states.view(batch_size, 
                                             -1, 
                                             self.n_heads, 
                                             self.key_value_proj_dim).transpose(1, 2)
            value_states = value_states.view(batch_size//self.n_channels, 
                                             self.n_channels, 
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
                mask = mask.view(batch_size//self.n_channels, self.n_channels, 1, 1, key_states.shape[-2])
                mask = (1.0 - mask.float()) * -1e9
                position_bias = position_bias + mask

        if self.pruned_heads:
            head_mask = torch.ones(position_bias.shape[1])
            head_mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, head_mask.bool()]
        else:
            position_bias_masked = position_bias
        
        # Infini attention computation across channels
        memory_matrix, z = self._update_memory_matrix(key_states, value_states)
        A_mem = self._retrieve_from_memory(query_states, memory_matrix, z)

        scores = query_states @ key_states.transpose(-2, -1) # [batch_size, n_channels, n_heads, n_patch, n_patch]
        scores += position_bias_masked # [batch_size, n_channels, n_heads, n_patch, n_patch]
        
        scores = scores / torch.sqrt(torch.tensor(self.key_value_proj_dim, 
                                                  device=hidden_states.device, 
                                                  dtype=torch.float16
                                                  )
        ) # [batch_size, n_channels, n_heads, n_patch, n_patch]

        attn_weights = F.softmax(scores, dim=-1) # [batch_size, n_channels, n_heads, n_patch, n_patch]
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training) # [batch_size, n_channels, n_heads, n_patch, n_patch]

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = attn_weights @ value_states # [batch_size, n_channels, n_heads, n_patch, dim]

        attn_output = F.sigmoid(self.beta) * A_mem + (1 - F.sigmoid(self.beta)) * attn_output # [batch_size, n_channels, n_heads, n_patch, dim]

        attn_output = attn_output.transpose(2, 3).contiguous() # [batch_size, n_channels, n_patch, n_heads, dim]
        attn_output = attn_output.view(batch_size, -1, self.inner_dim) # [batch_size*n_channels, n_patch, n_heads*dim]
        attn_output = self.o(attn_output) # [batch_size*n_channels, n_patch, n_heads*dim]

        outputs = (attn_output, past_key_value, position_bias)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs
    
class MLP(nn.Module):
    """Multi-Layer Perceptron Class

    Args:
        in_features (int): Dimension of input.
        out_features (int): Dimension of output.
        activation (str): Activation function to use.
        hidden_size (int): Dimension of hidden layers.
        num_layers (int): Number of hidden layers.
        dropout (float): Dropout rate.
    """

    def __init__(
        self, in_features, out_features, activation, hidden_size, num_layers, dropout
    ):
        super().__init__()
        ACTIVATIONS = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"]
        assert activation in ACTIVATIONS, f"{activation} is not in {ACTIVATIONS}"

        self.activation = getattr(nn, activation)()

        # MultiLayer Perceptron
        # Input layer
        layers = [
            nn.Linear(in_features=in_features, out_features=hidden_size),
            self.activation,
            nn.Dropout(dropout),
        ]
        # Hidden layers
        for i in range(num_layers - 2):
            layers += [
                nn.Linear(in_features=hidden_size, out_features=hidden_size),
                self.activation,
                nn.Dropout(dropout),
            ]
        # Output layer
        layers += [nn.Linear(in_features=hidden_size, out_features=out_features)]

        # Store in layers as ModuleList
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
class T5MLPMixerAttention(T5Attention):
    def __init__(self,
        config: T5Config,
        has_relative_attention_bias=False,
        layer_idx: Optional[int] = None,
    ):
        super().__init__(config, has_relative_attention_bias, layer_idx)

        self.use_rope = config.use_rope
        self.elu = nn.ELU()
        self.n_channels = config.n_channels
        self.mlp = MLP(
            in_features=config.d_kv * 2,
            out_features=config.d_kv,
            activation='ReLU',
            hidden_size=config.mlpmixer_hidden_size,
            num_layers=config.mlpmixer_num_layers,
            dropout=config.mlpmixer_dropout,
        )

    def compute_bias(self, query_length, key_length, device=None, cache_position=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        if cache_position is None:
            context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        else:
            context_position = cache_position[:, None].to(device)
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]

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
    
    def _update_memory_matrix(self, key_states, value_states):
        # σ_k = elu(k) + 1
        sigma_k = self.elu(key_states) + 1.0  # [B, C, H, P, D]
        sigma_k_T = sigma_k.transpose(-2, -1)  # [B, C, H, D, P]

        # Compute per-channel memory matrices
        memory_matrix = torch.matmul(sigma_k_T, value_states)  # [B, C, H, D, D]

        # Build exclusion mask
        channel_mask = torch.ones(self.n_channels, self.n_channels, device=sigma_k.device)
        channel_mask.fill_diagonal_(0)  # 0 for the excluded channel
        channel_mask = channel_mask.view(1, self.n_channels, self.n_channels, 1, 1, 1)

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

    def _retrieve_from_memory(self, query_states, memory_matrix, z_excluded):
        sigma_q = self.elu(query_states) + 1.0  # [B, C, H, P, D]
        numerator = sigma_q @ memory_matrix         # [B, C, H, P, D]
        denominator = (sigma_q @ z_excluded) + 1e-6 # [B, C, H, P, 1]
        A_mem = numerator / denominator             # [B, C, H, P, D]
    
        return A_mem

    def forward(
        self,
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
        query_states = query_states.view(batch_size//self.n_channels, 
                                         self.n_channels, 
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
            key_states = key_states.view(batch_size//self.n_channels, 
                                         self.n_channels, 
                                         self.n_heads, 
                                         seq_length,
                                         self.key_value_proj_dim) # [batch_size, n_channels, n_heads, n_patch, dim]
            value_states = value_states.view(batch_size, 
                                             -1, 
                                             self.n_heads, 
                                             self.key_value_proj_dim).transpose(1, 2)
            value_states = value_states.view(batch_size//self.n_channels, 
                                             self.n_channels, 
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
                mask = mask.view(batch_size//self.n_channels, self.n_channels, 1, 1, key_states.shape[-2])
                mask = (1.0 - mask.float()) * -1e9
                position_bias = position_bias + mask

        if self.pruned_heads:
            head_mask = torch.ones(position_bias.shape[1])
            head_mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, head_mask.bool()]
        else:
            position_bias_masked = position_bias
        
        # Infini attention computation across channels
        memory_matrix, z = self._update_memory_matrix(key_states, value_states)
        A_mem = self._retrieve_from_memory(query_states, memory_matrix, z)

        scores = query_states @ key_states.transpose(-2, -1) # [batch_size, n_channels, n_heads, n_patch, n_patch]
        scores += position_bias_masked # [batch_size, n_channels, n_heads, n_patch, n_patch]
        
        scores = scores / torch.sqrt(torch.tensor(self.key_value_proj_dim, 
                                                  device=hidden_states.device, 
                                                  dtype=torch.float16
                                                  )
        ) # [batch_size, n_channels, n_heads, n_patch, n_patch]

        attn_weights = F.softmax(scores, dim=-1) # [batch_size, n_channels, n_heads, n_patch, n_patch]
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training) # [batch_size, n_channels, n_heads, n_patch, n_patch]

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = attn_weights @ value_states # [batch_size, n_channels, n_heads, n_patch, dim]
        attn_output = torch.concat((A_mem, attn_output), dim=-1)

        attn_output = self.mlp(attn_output) # [batch_size, n_channels, n_heads, n_patch, dim*2]

        attn_output = attn_output.transpose(2, 3).contiguous() # [batch_size, n_channels, n_patch, n_heads, dim]
        attn_output = attn_output.view(batch_size, -1, self.inner_dim) # [batch_size*n_channels, n_patch, n_heads*dim]
        attn_output = self.o(attn_output) # [batch_size*n_channels, n_patch, n_heads*dim]

        outputs = (attn_output, past_key_value, position_bias)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False, layer_idx: Optional[int] = None, beta: Optional[torch.tensor] = None):
        super().__init__()

        if config.channel_mixing_method.lower() == 'infini':
            self.SelfAttention = T5InfiniAttention(
                config, has_relative_attention_bias=has_relative_attention_bias, layer_idx=layer_idx, beta=beta,
            )
        elif config.channel_mixing_method.lower() == 'infini_ci_exclusion':
            self.SelfAttention = T5InfiniChannelExclusionAttention(
                config, has_relative_attention_bias=has_relative_attention_bias, layer_idx=layer_idx, beta=beta,
            )
        elif config.channel_mixing_method.lower() == 'mlp_mixer':
            self.SelfAttention = T5MLPMixerAttention(
                config, has_relative_attention_bias=has_relative_attention_bias, layer_idx=layer_idx
            )
        elif config.channel_mixing_method.lower() == 'none':
            self.SelfAttention = T5Attention(
                config, has_relative_attention_bias=has_relative_attention_bias, layer_idx=layer_idx
            )
        else:
            raise Exception(f"Channel mixing method: {config.channel_mixing_method} is not an option. Please use one of 'infini' 'infini_ci_exclusion', 'mlp_mixer', 'none'.")
        
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs
    
class T5LayerCrossAttention(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None, beta: Optional[torch.tensor] = None):
        super().__init__()

        if config.channel_mixing_method.lower() == 'infini':
            self.EncDecAttention = T5InfiniAttention(
                config, has_relative_attention_bias=False, layer_idx=layer_idx, beta=beta,
            )
        elif config.channel_mixing_method.lower() == 'infini_ci_exclusion':
            self.EncDecAttention = T5InfiniChannelExclusionAttention(
                config, has_relative_attention_bias=False, layer_idx=layer_idx, beta=beta,
            )
        elif config.channel_mixing_method.lower() == 'mlp_mixer':
            self.EncDecAttention = T5MLPMixerAttention(
                config, has_relative_attention_bias=False, layer_idx=layer_idx
            )
        elif config.channel_mixing_method.lower() == 'none':
            self.EncDecAttention = T5Attention(
                config, has_relative_attention_bias=False, layer_idx=layer_idx
            )
        else:
            raise Exception(f"Channel mixing method: {config.channel_mixing_method} is not an option. Please use one of 'infini' 'infini_ci_exclusion', 'mlp_mixer', 'none'.")
    
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
        cache_position=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs
        
class T5Block(T5Block):
    def __init__(self, config, has_relative_attention_bias=False, layer_idx: Optional[int] = None, beta: Optional[torch.tensor] = None):
        super().__init__(config)
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(
            T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias, layer_idx=layer_idx, beta=beta)
        )
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config, layer_idx=layer_idx))

        self.layer.append(T5LayerFF(config))

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

class T5EncoderModel(T5EncoderModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
