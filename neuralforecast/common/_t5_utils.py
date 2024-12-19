import torch
import torch.nn as nn
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Attention, T5Stack, T5Block, T5LayerNorm, T5LayerSelfAttention, T5LayerCrossAttention, T5LayerFF
from typing import Optional
from ._positional_encodings import PositionalEncoding

     
class CustomT5LayerSelfAttention(T5LayerSelfAttention):
    def __init__(self, config, pe, has_relative_attention_bias=False, layer_idx: Optional[int] = None):
        super().__init__(config, has_relative_attention_bias, layer_idx)
        self.SelfAttention = CustomT5Attention(
           config, pe=pe, has_relative_attention_bias=has_relative_attention_bias, layer_idx=layer_idx, 
        )
        
class CustomT5Stack(T5Stack):
    def __init__(self, config, pe="zeros", embed_tokens=None):
        super().__init__(config, embed_tokens)

        # Modify the blocks based on the 'pe' parameter
        if pe == "relative":
            for i, block in enumerate(self.block):
                block.layer[0] = CustomT5LayerSelfAttention(
                    config, pe=pe, has_relative_attention_bias=bool(i == 0), layer_idx=i,
                )
        if pe == "rope":
            for i, block in enumerate(self.block):
                if i == 0:  # Apply RoPE only for the first layer
                    block.layer[0] = CustomT5LayerSelfAttention(
                        config, pe="rope", has_relative_attention_bias=False, layer_idx=i
                    )
                else:  # For other layers, use default or other configurations
                    block.layer[0] = T5LayerSelfAttention(
                        config, has_relative_attention_bias=False, layer_idx=i
                    )
        else:
            for i, block in enumerate(self.block):
                block.layer[0] = CustomT5LayerSelfAttention(
                    config, pe=pe, has_relative_attention_bias=False, layer_idx=i,
                )

class CustomT5Attention(T5Attention):
    def __init__(
        self,
        config: T5Config,
        pe='zeros',  # Added option for RoPE
        has_relative_attention_bias=False,
        layer_idx: Optional[int] = None,
    ):
        super().__init__(config, has_relative_attention_bias, layer_idx)
        
        self.pe = pe  # RoPE flag
        #print('has_relative_attention_bias:', has_relative_attention_bias)

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
           
        #NEW
        if self.pe == 'rope':
            query_states = PositionalEncoding().RotaryPositionalEmbedding(query_states.clone())
            key_states = PositionalEncoding().RotaryPositionalEmbedding(key_states.clone())

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
                causal_mask = mask[:, :, :, : key_states.shape[-2]]
                position_bias = position_bias + causal_mask

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
