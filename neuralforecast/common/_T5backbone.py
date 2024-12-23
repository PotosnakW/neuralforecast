import copy
import torch
import torch.nn as nn

from transformers import T5Config, T5Model, AutoConfig

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

        self.num_vars = config['c_in']
        self.attn_mask = config['attn_mask']
        self.decoder = None # Assume encoder-only first
        self.W_pos_encoder = PositionalEncoding(pe=config['pe']).output(config['learn_pe'], 
                                                                     config['token_num'], 
                                                                     config['d_model']
                                                                    )
        
        # Residual dropout
        self.dropout = nn.Dropout(config['dropout'])

        # Encoder
        if 'google' in config['backbone_type']:
            model_config = T5Config.from_pretrained(config['backbone_type'])
        else:
            model_config = T5Config.from_dict(config)

        model_config.num_decoder_layers = config["num_decoder_layers"]
        model_config.is_decoder = False
        model_config.is_encoder_decoder = False
        model_config.enable_gradient_checkpointing = False
        model_config.use_cache = False

        transformer_backbone = T5Model(model_config)
        transformer_backbone.encoder = CustomT5Stack(model_config, pe=config['pe'])
        self.encoder = transformer_backbone.get_encoder()
        
        if model_config.num_decoder_layers > 0:
            decoder_config = copy.deepcopy(model_config)
            decoder_config.is_decoder = True 
            decoder_config.is_encoder_decoder = False # does this do something?
            decoder_config.num_layers = model_config.num_decoder_layers
            transformer_backbone.decoder = CustomT5Stack(decoder_config, pe=config['pe'])
            self.decoder = transformer_backbone.get_decoder() 

    def forward(self, x) -> torch.Tensor:  # x:  [bs x nvars x patch_num x hidden_size]
        attn_mask = torch.ones(x.shape[0], x.shape[2], dtype=torch.long).to(x.device)
        if self.attn_mask=='causal':
            attn_mask = torch.tril(attn_mask)
        
        u = torch.reshape(
            x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        )  # u: [bs * nvars x patch_num x hidden_size]
        u = self.dropout(u + self.W_pos_encoder)  
        
        if self.decoder:
            ue = u[:, :-1, :].clone()
            ud = u[:, -1:, :].clone()
            attn_mask = torch.ones(x.shape[0], x.shape[2]-1, dtype=torch.long).to(x.device)
            causal_attn_mask = torch.tril(torch.ones(x.shape[0], 
                                                     1, 
                                                     dtype=torch.long, 
                                                     device=x.device
                                                    )
                                         )
        else:
            ue = u.clone()
            ud = None

        z = self.encoder(inputs_embeds=ue, attention_mask=attn_mask)

        #Decoder
        if self.decoder:
            z = self.decoder(
                attention_mask=causal_attn_mask,
                inputs_embeds=ud,
                encoder_hidden_states=z.last_hidden_state, #output of the last layer of the model
                encoder_attention_mask=attn_mask,
                )
        
        z = z.last_hidden_state
        z = torch.reshape(
            z, (-1, self.num_vars, z.shape[-2], z.shape[-1])
        )  # z: [bs x nvars x patch_num x hidden_size]

        return z
