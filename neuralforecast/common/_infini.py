import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from ..common._modules import MLP

class _ScaledDotProductAttention(nn.Module):
    """
    Vanilla Scaled Dot-Product Attention.
    Based on "Attention is All You Need" (Vaswani et al., 2017).
    """
    
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        attn_dropout: float = 0.0,
        res_attention=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.scale = self.head_dim ** -0.5
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        prev: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Scaled Dot-Product Attention.
        
        Input shape:
            q: [bs x n_heads x seq_len x d_k]
            k: [bs x n_heads x d_k x seq_len]  (transposed)
            v: [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
            
        Output shape:
            output: [bs x n_heads x seq_len x d_v]
            attn_weights: [bs x n_heads x seq_len x seq_len]
        """
        
        # Scaled MatMul (q, k) - compute attention scores
        attn_scores = torch.matmul(q, k) * self.scale  # Vaswani et al. scaling

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None:
            attn_scores = attn_scores + prev
        
        # Apply attention mask (optional)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -float('inf'))
            else:
                attn_scores += attn_mask
        
        # Apply key padding mask (optional)
        if key_padding_mask is not None:
            attn_scores.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2), -float('inf')
            )
        
        # Normalize attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Compute attention output
        output = torch.matmul(attn_weights, v)
        
        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights

class _InfiniScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention with Infini-attention memory mechanism.
    Based on "Attention is All You Need" (Vaswani et al., 2017) and 
    "Leave No Context Behind" (Munkhdalai et al., 2024).
    """
    
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        attn_dropout: float = 0.0,
        res_attention=False,
        infini_channel_exclusion: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.scale = self.head_dim ** -0.5
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        self.elu = nn.ELU()
        
        # Select memory update/retrieval methods based on channel exclusion
        if infini_channel_exclusion:
            self._update_memory_matrix = self._update_memory_matrix_channelexl
            self._retrieve_from_memory = self._retrieve_from_memory_channelexl
        else:
            self._update_memory_matrix = self._update_memory_matrix_allchannels
            self._retrieve_from_memory = self._retrieve_from_memory_allchannels
    
    def _update_memory_matrix_allchannels(self, key_states, value_states):
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
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        n_channels: int,
        prev: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Scaled Dot-Product Attention with memory mechanism.
        
        Input shape (with channels):
            q: [bs x n_channels x n_heads x seq_len x d_k]
            k: [bs x n_channels x n_heads x d_k x seq_len]  (transposed)
            v: [bs x n_channels x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
            
        Output shape:
            output: [bs x n_channels x n_heads x seq_len x d_v]
            A_mem: [bs x n_channels x n_heads x seq_len x d_v]
            attn_weights: [bs x n_channels x n_heads x seq_len x seq_len]
        """
        
        # Scaled MatMul (q, k) - compute attention scores
        attn_scores = torch.matmul(q, k) * self.scale  # Vaswani et al. scaling

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None:
            attn_scores = attn_scores + prev
        
        # Apply attention mask (optional)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -float('inf'))
            else:
                attn_scores += attn_mask
        
        # Apply key padding mask (optional)
        if key_padding_mask is not None:
            attn_scores.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3), -float('inf')
            )
        
        # Normalize attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Compute attention output (v is [B, C, H, P, D])
        output = torch.matmul(attn_weights, v)
        
        # Infini-attention: retrieve from memory
        # k_for_memory should be [B, C, H, P, D] (not transposed)
        k_for_memory = k.transpose(-2, -1)
        memory_matrix, z = self._update_memory_matrix(k_for_memory, v)
        A_mem = self._retrieve_from_memory(q, memory_matrix, z)

        if self.res_attention:
            return output, A_mem, attn_weights, attn_scores
        else:
            return output, A_mem, attn_weights
        
class _MultiheadAttention(nn.Module):
    """
    Multi-Head Attention with optional Infini-attention memory mechanism.
    Traditional format similar to standard Transformer implementations.
    """
    
    def __init__(
        self,
        n_channels: int,
        hidden_size: int,
        n_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        res_attention=False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        qkv_bias: bool = True,
        infini_mixer_type: str = 'none',
        infini_channel_exclusion: bool = False,
        beta: Optional[torch.tensor] = None,
        channelwise_beta: bool = False,
        mlpmixer_hidden_size: int = 128,
        mlpmixer_num_layers: int = 3,
        mlpmixer_dropout: float = 0.1,
    ):
        """
        Multi-Head Attention with optional Infini memory mechanism.
        
        Args:
            hidden_size: Model dimension
            n_heads: Number of attention heads
            d_k: Dimension per head for keys/queries (default: hidden_size // n_heads)
            d_v: Dimension per head for values (default: hidden_size // n_heads)
            attn_dropout: Dropout rate for attention weights
            proj_dropout: Dropout rate for output projection
            qkv_bias: Whether to use bias in Q/K/V projections
            infini_mixer_type: Type of mixer ('none', 'mlp', 'betas')
            infini_channel_exclusion: Whether to exclude self-channel in memory
        """
        super().__init__()
        
        d_k = hidden_size // n_heads if d_k is None else d_k
        d_v = hidden_size // n_heads if d_v is None else d_v
        
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.infini_mixer_type = infini_mixer_type.lower()
        self.res_attention = res_attention
        
        # Q, K, V projections
        self.W_Q = nn.Linear(hidden_size, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(hidden_size, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(hidden_size, d_v * n_heads, bias=qkv_bias)
        
        # Scaled Dot-Product Attention (vanilla or infini)
        if self.infini_mixer_type == 'none':
            self.sdp_attn = _ScaledDotProductAttention(
                hidden_size=hidden_size,
                n_heads=n_heads,
                attn_dropout=attn_dropout,
                res_attention=res_attention,
            )
        elif self.infini_mixer_type in ['mlp', 'betas']:
            self.sdp_attn = _InfiniScaledDotProductAttention(
                hidden_size=hidden_size,
                n_heads=n_heads,
                attn_dropout=attn_dropout,
                res_attention=res_attention,
                infini_channel_exclusion=infini_channel_exclusion,
            )

            if self.infini_mixer_type == 'mlp':
                self.mlp = MLP(
                    in_features=d_v * 2,
                    out_features=d_v,
                    activation='ReLU',
                    hidden_size=mlpmixer_hidden_size,
                    num_layers=mlpmixer_num_layers,
                    dropout=mlpmixer_dropout,
                )
                
            elif self.infini_mixer_type == 'betas':
                # Beta parameter for gated mixing
                if beta is not None:
                    self.beta = beta
                else:
                    if channelwise_beta:
                        self.beta = nn.Parameter(torch.rand((1, n_channels, n_heads, 1, 1)) * 1e-2)
                    else:
                        self.beta = nn.Parameter(torch.rand((1, 1, n_heads, 1, 1)) * 1e-2)
                    # Adjust the values to ensure they sum to 0 across heads
                    with torch.no_grad():
                        self.beta -= self.beta.mean(dim=2, keepdim=True)
        else:
            raise ValueError(f"infini_mixer_type must be 'none', 'mlp', or 'betas', got '{infini_mixer_type}'")
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_v, hidden_size),
            nn.Dropout(proj_dropout)
        )
    
    def forward(
        self,
        n_channels: int,
        Q: torch.Tensor,
        K: Optional[torch.Tensor] = None,
        V: Optional[torch.Tensor] = None,
        prev: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for multi-head attention.
        
        Input shape (without channels):
            Q: [bs x seq_len x hidden_size]
            K: [bs x seq_len x hidden_size] (optional, defaults to Q)
            V: [bs x seq_len x hidden_size] (optional, defaults to Q)
            
        Input shape (with channels):
            Q: [bs*n_channels x seq_len x hidden_size]
            (internally reshaped to [bs x n_channels x seq_len x hidden_size])
            
        Output shape:
            output: [bs*n_channels x seq_len x hidden_size]
            A_mem: [bs x n_channels x n_heads x seq_len x d_v] (if infini_mixer_type != 'none')
            attn_weights: [bs x (n_channels) x n_heads x seq_len x seq_len]
        """
        
        bs = Q.size(0)
        if K is None:
            K = Q
        if V is None:
            V = Q
        
        use_channels = n_channels > 1
        print(f"{use_channels}")
        print(f"{n_channels}")
        
        # Linear projections and split into multiple heads
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k)  # [bs x seq_len x n_heads x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k)  # [bs x seq_len x n_heads x d_k]
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v)  # [bs x seq_len x n_heads x d_v]
        
        if use_channels and self.infini_mixer_type != 'none':
            print('using infini')
            # Reshape for multi-channel processing (Infini-attention)
            seq_len = q_s.size(1)
            bs_orig = bs // n_channels
            
            q_s = q_s.view(bs_orig, n_channels, seq_len, self.n_heads, self.d_k)
            q_s = q_s.transpose(2, 3).contiguous()  # [bs x n_channels x n_heads x seq_len x d_k]
            
            k_s = k_s.view(bs_orig, n_channels, seq_len, self.n_heads, self.d_k)
            k_s = k_s.permute(0, 1, 3, 4, 2).contiguous()  # [bs x n_channels x n_heads x d_k x seq_len]
            
            v_s = v_s.view(bs_orig, n_channels, seq_len, self.n_heads, self.d_v)
            v_s = v_s.transpose(2, 3).contiguous()  # [bs x n_channels x n_heads x seq_len x d_v]
            
            # Apply Scaled Dot-Product Attention (multiple heads)
            if self.res_attention:
                output, A_mem, attn_weights, attn_scores = self.sdp_attn(
                    q=q_s,
                    k=k_s,
                    v=v_s,
                    n_channels=n_channels,
                    prev=prev,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                )
            else:
                output, A_mem, attn_weights = self.sdp_attn(
                    q=q_s, 
                    k=k_s, 
                    v=v_s, 
                    n_channels=n_channels,
                    key_padding_mask=key_padding_mask, 
                    attn_mask=attn_mask
                )
            
            # Mix attention and memory
            if self.infini_mixer_type == 'mlp':
                # Concatenate memory and attention outputs
                mixed = torch.cat([A_mem, output], dim=-1)  # [bs x n_channels x n_heads x seq_len x d_v*2]
                output = self.mlp(mixed)  # [bs x n_channels x n_heads x seq_len x d_v]
            elif self.infini_mixer_type == 'betas':
                output = torch.sigmoid(self.beta) * A_mem + (1 - torch.sigmoid(self.beta)) * output
            
            # Reshape back
            output = output.transpose(2, 3).contiguous()  # [bs x n_channels x seq_len x n_heads x d_v]
            output = output.view(bs, -1, self.n_heads * self.d_v)  # [bs*n_channels x seq_len x n_heads*d_v]
            
        else:
            print('using none')
            # Standard transformer format (vanilla attention or no channels)
            q_s = q_s.transpose(1, 2)  # [bs x n_heads x seq_len x d_k]
            k_s = k_s.permute(0, 2, 3, 1)  # [bs x n_heads x d_k x seq_len]
            v_s = v_s.transpose(1, 2)  # [bs x n_heads x seq_len x d_v]
            
            # Apply Scaled Dot-Product Attention (multiple heads)
            if self.res_attention:
                output, attn_weights, attn_scores = self.sdp_attn(
                    q=q_s,
                    k=k_s,
                    v=v_s,
                    prev=prev,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                )
            else:
                output, attn_weights = self.sdp_attn(
                    q=q_s, 
                    k=k_s, 
                    v=v_s, 
                    key_padding_mask=key_padding_mask, 
                    attn_mask=attn_mask
                )
            
            # Reshape back
            output = output.transpose(1, 2).contiguous()  # [bs x seq_len x n_heads x d_v]
            output = output.view(bs, -1, self.n_heads * self.d_v)  # [bs x seq_len x n_heads*d_v]
        
        # Final output projection
        output = self.to_out(output)
        
        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights
