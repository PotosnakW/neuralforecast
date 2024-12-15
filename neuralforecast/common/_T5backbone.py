import torch
import torch.nn as nn

from transformers import T5Config, T5Model

from ..common._positional_encodings import PositionalEncoding
from ..common._t5_utils import CustomT5Stack, CustomT5Attention


class T5backbone(nn.Module):  # i means channel-independent
    """
    TSTEncoder
    """

    def __init__(
        self,
        config,
    ):

        super().__init__()

        self.attn_mask = config.attn_mask

        # Input encoding
        self.W_P = nn.Linear(
            config.token_len, config.d_model
        )  # Eq 1: projection of feature vectors onto a d-dim vector space

        self.decoder = None
        self.W_pos_encoder = PositionalEncoding(pe=config.pe).output(config.learn_pe, 
                                                                     config.token_num, 
                                                                     config.d_model
                                                                    )
        
        # Residual dropout
        self.dropout = nn.Dropout(config.dropout)

        # Encoder
        model_config = T5Config.from_dict(config)
        transformer_backbone = T5Model(model_config)
        transformer_backbone.encoder = CustomT5Stack(model_config, pe=config.pe)
        self.encoder = transformer_backbone.get_encoder()
        
        if model_config.num_decoder_layers > 0:
            decoder_config = copy.deepcopy(model_config)
            decoder_config.is_decoder = True
            #decoder_config.is_encoder_decoder = True
            decoder_config.num_layers = model_config.num_decoder_layers
            transformer_backbone.decoder = CustomT5Stack(decoder_config, pe=config.pe)
            self.decoder = transformer_backbone.get_decoder() 
            self.W_pos_encoder = PositionalEncoding(pe=config.pe).output(config.learn_pe, 
                                                                         config.token_num-1, 
                                                                         config.d_model
                                                                        )
            self.W_pos_decoder = PositionalEncoding(pe=config.pe).output(config.learn_pe, 
                                                                         1, 
                                                                         config.d_model
                                                                        )           
        
        if config["enable_gradient_checkpointing"]==True:
            transformer_backbone.gradient_checkpointing_enable()
            
    def forward(self, x) -> torch.Tensor:  # x: [bs x nvars x patch_len x patch_num]
        n_vars = x.shape[1]
        if self.decoder:
            xe = x[:, :, :, :-1].clone()
            xd = x[:, :, :, -1].clone()
            xd = xd.unsqueeze(-1)
        else:
            xe = x.clone()

        # Mask [bs x token_num]
        if (self.attn_mask=='bidirectional') & (self.decoder==None):
            attn_mask = torch.ones(xe.shape[0], xe.shape[3], dtype=torch.long).to(x.device)
        elif (self.attn_mask=='bidirectional') & (self.decoder!=None):
            attn_mask = torch.ones(xe.shape[0], xe.shape[3], dtype=torch.long).to(x.device)
        elif self.attn_mask=='causal':
            attn_mask = torch.tril(torch.ones(xe.shape[0], xe.shape[3], dtype=torch.long, device=x.device))
           
        # Encoder
        xe = xe.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
        xe = self.W_P(xe)  # x: [bs x nvars x patch_num x hidden_size]
        ue = torch.reshape(
            xe, (xe.shape[0] * xe.shape[1], xe.shape[2], xe.shape[3])
        )  # u: [bs * nvars x patch_num x hidden_size]
        ue = self.dropout(ue + self.W_pos_encoder)  
        z = self.encoder(inputs_embeds=ue, attention_mask=attn_mask)
        
        # Decoder
        if self.decoder:
            causal_attn_mask = torch.tril(torch.ones(xd.shape[0], 
                                                     xd.shape[3], 
                                                     dtype=torch.long, 
                                                     device=x.device
                                                    )
                                         )
            xd = xd.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
            xd = self.W_P(xd)  # x: [bs x nvars x patch_num x hidden_size]
            ud = torch.reshape(
                xd, (xd.shape[0] * xd.shape[1], xd.shape[2], xd.shape[3])
            )  # u: [bs * nvars x patch_num x hidden_size]
            ud = self.dropout(ud + self.W_pos_decoder)  
        
            z = self.decoder(
                attention_mask=causal_attn_mask,
                inputs_embeds=ud,
                encoder_hidden_states=z.last_hidden_state, #output of the last layer of the model
                encoder_attention_mask=attn_mask,
                )

        z = z.last_hidden_state
        z = torch.reshape(
            z, (-1, n_vars, z.shape[-2], z.shape[-1])
        )  # z: [bs x nvars x patch_num x hidden_size]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x hidden_size x patch_num]

        return z
