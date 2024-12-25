import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.token_num = config['token_num']
        self.d_model = config['d_model']     
        self.decoder = None # Assume encoder-only first
        self.W_pos = PositionalEncoding(pe=config['pe']).output(config['learn_pe'], 
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
        model_config.enable_gradient_checkpointing = False
        model_config.use_cache = False

        self.shared = nn.Embedding(model_config.vocab_size, model_config.d_model)

        encoder_config = copy.deepcopy(model_config)
        if config['attn_mask'] == 'bidirectional':
            encoder_config.is_decoder = False
        elif config['attn_mask'] == 'causal':
            encoder_config.is_decoder = True
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = CustomT5Stack(encoder_config, pe=config['pe'], embed_tokens=self.shared)

        if model_config.num_decoder_layers > 0: 
            decoder_config = copy.deepcopy(model_config)
            decoder_config.is_decoder = True
            decoder_config.is_encoder_decoder = True
            decoder_config.num_layers = decoder_config.num_decoder_layers
            self.decoder = CustomT5Stack(decoder_config, pe=config['pe'], embed_tokens=self.shared)
            self.num_tokens_decoder = config['token_num_decoder']


    def forward(self, x) -> torch.Tensor:  # x:  [bs x nvars x patch_num x hidden_size]
        bs = x.shape[0]
        attn_mask = torch.ones(bs, 
                               self.token_num, 
                               dtype=torch.long
                              ).to(x.device)
        
        u = torch.reshape(
            x, (bs * self.num_vars, self.token_num, self.d_model)
        )  # u: [bs * nvars x patch_num x hidden_size]
        u = self.dropout(u + self.W_pos)  
        
        if self.decoder:
            ue = u[:, :self.token_num-1, :].clone()
            ud = u[:, self.token_num-1:, :].clone()

            if ud.shape[1] != self.num_tokens_decoder:
                patch_num_pad = self.num_tokens_decoder - ud.shape[1]
                ud = F.pad(ud, (0, 0, 0, patch_num_pad), mode='constant', value=0)

            attn_mask = attn_mask[:, :-1]
            decoder_attn_mask = torch.ones(x.shape[0], 
                                           self.num_tokens_decoder, #1
                                           dtype=torch.long, 
                                           device=x.device
                                          )
            decoder_attn_mask[:, ud.shape[1]:] = 0
            
        else:
            ue = u.clone()
            ud = None

        z = self.encoder(inputs_embeds=ue, attention_mask=attn_mask)
        #print(z.last_hidden_state.shape)

        #Decoder
        if self.decoder:
            z = self.decoder(
                attention_mask=decoder_attn_mask,
                inputs_embeds=ud,
                encoder_hidden_states=z.last_hidden_state, #output of the last layer of the model
                encoder_attention_mask=attn_mask,
                )
        
        z = z.last_hidden_state
        z = torch.reshape(
            z, (-1, self.num_vars, z.shape[-2], z.shape[-1])
        )  # z: [bs x nvars x patch_num x hidden_size]

        return z
