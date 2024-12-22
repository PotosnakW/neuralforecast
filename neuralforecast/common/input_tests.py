def check_input_validity(config):
    assert config['backbone_type'] in [
        "ttm", 
        "t5", 
        "google/t5-efficient-tiny", 
        "google/t5-efficient-mini", 
        "google/t5-efficient-small", 
        "google/t5-efficient-base", 
        ], \
        "Backbone not included. Must be one of 'ttm',\
                        't5',\
                        'google/t5-efficient-tiny',\
                        'google/t5-efficient-mini',\
                        'google/t5-efficient-small',\
                        'google/t5-efficient-base'."
        
      
    # Check number of decoder layers
    d_model_check = {"t5": config['d_model'],
             "ttm": config['d_model'],
             "google/t5-efficient-tiny": 256,
             "google/t5-efficient-mini": 384,
             "google/t5-efficient-small": 512,
             "google/t5-efficient-base": 768
            }
    config['d_model'] = d_model_check[config['backbone_type']]
    
    # Check number of decoder layers
    num_decoder_layers_check = {"t5": config['num_decoder_layers'],
             "ttm": config['num_decoder_layers'],
             "google/t5-efficient-tiny": 4,
             "google/t5-efficient-mini": 4,
             "google/t5-efficient-small": 6,
             "google/t5-efficient-base": 12,
            }
    if config['num_decoder_layers'] > 0:
        config['num_decoder_layers'] = num_decoder_layers_check[config['backbone_type']]

    # Enforce correct patch_len, regardless of user input
    if (config['tokenizer_type'] == 'lags')|(config['tokenizer_type'] == 'bins'):
        config['input_token_len']=1
        config['stride']=1
        config['padding_patch']=None

    elif 'patch' in config['tokenizer_type']:
        config['input_token_len'] = min(config['context_len'] + config['stride'], 
                                        config['input_token_len'])
        config['output_token_len'] = min(config['h'], config['output_token_len'])
    
    return config