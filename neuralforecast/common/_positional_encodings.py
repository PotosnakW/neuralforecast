import torch
import torch.nn as nn
import math


class PositionalEncoding():
    
    def __init__(
        self, pe=None,
    ):
        super(PositionalEncoding, self).__init__()

        self.pe = pe

    # %% ../../nbs/models.patchtst.ipynb 11
    def SinCosPosEncoding(self, q_len, hidden_size, normalize=True):
        pe = torch.zeros(q_len, hidden_size)
        position = torch.arange(0, q_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if normalize:
            pe = pe - pe.mean()
            pe = pe / (pe.std() * 10)
        return pe

    def Coord2dPosEncoding(self, q_len, hidden_size, exponential=False, normalize=True, eps=1e-3):
        x = 0.5 if exponential else 1
        i = 0
        for i in range(100):
            cpe = (
                2
                * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x)
                * (torch.linspace(0, 1, hidden_size).reshape(1, -1) ** x)
                - 1
            )
            if abs(cpe.mean()) <= eps:
                break
            elif cpe.mean() > eps:
                x += 0.001
            else:
                x -= 0.001
            i += 1
        if normalize:
            cpe = cpe - cpe.mean()
            cpe = cpe / (cpe.std() * 10)
        return cpe

    def Coord1dPosEncoding(self, q_len, exponential=False, normalize=True):
        cpe = (
            2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** (0.5 if exponential else 1))
            - 1
        )
        if normalize:
            cpe = cpe - cpe.mean()
            cpe = cpe / (cpe.std() * 10)
        return cpe
    
    def RotaryPositionalEmbedding(self, q, base=10000.0):
        """Compute Rotary Position Embedding (RoPE)."""

        bs, n_heads, q_len, d_k = q.shape

        theta = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        pos = torch.arange(0, q_len, dtype=torch.float) 
        idx_theta = torch.einsum("i , j -> i j", pos, theta)
        cos_remb = idx_theta.cos().unsqueeze(0).unsqueeze(0).to(q.device)
        sin_remb = idx_theta.sin().unsqueeze(0).unsqueeze(0).to(q.device)

        # Reshape tensor to apply rotation and apply rotation matrix
#         q_reshaped = q.view(bs, n_heads, q_len, d_k // 2, 2).clone()
#         rotated_q = torch.stack([q_reshaped[..., 0] * cos_remb - q_reshaped[..., 1] * sin_remb,
#                                  q_reshaped[..., 0] * sin_remb + q_reshaped[..., 1] * cos_remb
#                                 ], 
#                                  dim=-1
#                                 )
#         # Reshape back to original tensor shape
#         rpe = rotated_q.view(bs, n_heads, q_len, d_k)
        
        q1, q2 = q[..., ::2], q[..., 1::2]  # Split into even and odd dimensions
        rpe = torch.cat([q1 * cos_remb - q2 * sin_remb,
                               q1 * sin_remb + q2 * cos_remb], dim=-1)

        return rpe

    def output(self, learn_pe, q_len, hidden_size):
        # Positional encoding
        if self.pe == None:
            W_pos = torch.empty(
                (q_len, hidden_size)
            )  # pe = None and learn_pe = False can be used to measure impact of pe
            nn.init.uniform_(W_pos, -0.02, 0.02)
            learn_pe = False
        elif self.pe == "zero":
            W_pos = torch.empty((q_len, 1))
            nn.init.uniform_(W_pos, -0.02, 0.02)
        elif self.pe == "zeros":
            W_pos = torch.empty((q_len, hidden_size))
            nn.init.uniform_(W_pos, -0.02, 0.02)
        elif self.pe == "normal" or self.pe == "gauss":
            W_pos = torch.zeros((q_len, 1))
            torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
        elif self.pe == "uniform":
            W_pos = torch.zeros((q_len, 1))
            nn.init.uniform_(W_pos, a=0.0, b=0.1)
        elif self.pe == "lin1d":
            W_pos = self.Coord1dPosEncoding(q_len, exponential=False, normalize=True)
        elif self.pe == "exp1d":
            W_pos = self.Coord1dPosEncoding(q_len, exponential=True, normalize=True)
        elif self.pe == "lin2d":
            W_pos = self.Coord2dPosEncoding(
                q_len, hidden_size, exponential=False, normalize=True
            )
        elif self.pe == "exp2d":
            W_pos = self.Coord2dPosEncoding(q_len, hidden_size, exponential=True, normalize=True)
        elif self.pe == "sincos":
            W_pos = self.SinCosPosEncoding(q_len, hidden_size, normalize=True)
        elif self.pe == "sincos_relative":
            W_pos = self.SinCosPosEncoding(q_len, hidden_size, normalize=True)
        elif self.pe == "rope":
            W_pos = torch.empty((q_len, hidden_size))
            nn.init.uniform_(W_pos, -0.02, 0.02)
        elif self.pe == "relative":
            W_pos = torch.empty((q_len, hidden_size))
            nn.init.uniform_(W_pos, -0.02, 0.02)
        else:
            raise ValueError(
                f"{self.pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
            'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', rope None.)"
            )
        
        return nn.Parameter(W_pos, requires_grad=learn_pe)