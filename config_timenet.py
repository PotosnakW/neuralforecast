from neuralforecast.models import *
from neuralforecast.losses.pytorch import MAE, MQLoss, HuberMQLoss, DistributionLoss

# GLOBAL parameters

LOSS = HuberMQLoss(quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
LOSS_PROBA = DistributionLoss(distribution="StudentT", level=[80, 90], return_params=False)

MODELS = ['nhits_30_1024',
          'tft_1024',
          'lstm_512_4'
          'deepar_128_4']

FREQUENCY = ['minutely',
             '10minutely',
             '15minutely',
             '30minutely',
             'hourly',
             'daily',
             'weekly',
             'monthly',
             'quarterly',
             'yearly']

HORIZON_DICT = {'yearly': 1,
                'quarterly': 4,
                'monthly': 12,
                'weekly': 1,
                'daily': 7,
                'hourly': 24,
                '30minutely': 48,
                '15minutely': 96,
                '10minutely': 144,
                'minutely': 60}

def load_model(model_name):
    model = None
    frequency = model_name.split('_')[-1]
    horizon = HORIZON_DICT[frequency]

    ########################################################## NHITS ##########################################################
    if model_name == 'nhits_30_1024_minutely':
        model = NHITS(h=horizon,
                        input_size=5*horizon,
                        dropout_prob_theta=0.2,
                        stack_types=3*['identity'],
                        mlp_units=3 * [[1024, 1024, 1024, 1024]],
                        n_blocks=3*[10],
                        n_pool_kernel_size=3*[1],
                        n_freq_downsample=[4, 2, 1],
                        loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=5,
                        val_check_steps=100,
                        scaler_type='minmax1',
                        max_steps=1000,
                        batch_size=32,
                        windows_batch_size=128,
                        random_seed=1)
    if model_name == 'nhits_30_1024_10minutely':
        model = NHITS(h=horizon,
                        input_size=5*horizon,
                        dropout_prob_theta=0.2,
                        stack_types=3*['identity'],
                        mlp_units=3 * [[1024, 1024, 1024, 1024]],
                        n_blocks=3*[10],
                        n_pool_kernel_size=3*[1],
                        n_freq_downsample=[24, 2, 1],
                        loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=5,
                        val_check_steps=100,
                        scaler_type='minmax1',
                        max_steps=1000,
                        batch_size=32,
                        windows_batch_size=128,
                        random_seed=1)
    if model_name == 'nhits_30_1024_15minutely':
        model = NHITS(h=horizon,
                        input_size=5*horizon,
                        dropout_prob_theta=0.2,
                        stack_types=3*['identity'],
                        mlp_units=3 * [[1024, 1024, 1024, 1024]],
                        n_blocks=3*[10],
                        n_pool_kernel_size=3*[1],
                        n_freq_downsample=[24, 2, 1],
                        loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=5,
                        val_check_steps=100,
                        scaler_type='minmax1',
                        max_steps=1000,
                        batch_size=32,
                        windows_batch_size=128,
                        random_seed=1)
    if model_name == 'nhits_30_1024_30minutely':
        model = NHITS(h=horizon,
                        input_size=5*horizon,
                        dropout_prob_theta=0.2,
                        stack_types=3*['identity'],
                        mlp_units=3 * [[1024, 1024, 1024, 1024]],
                        n_blocks=3*[10],
                        n_pool_kernel_size=3*[1],
                        n_freq_downsample=[24, 2, 1],
                        loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=5,
                        val_check_steps=100,
                        scaler_type='minmax1',
                        max_steps=1000,
                        batch_size=32,
                        windows_batch_size=128,
                        random_seed=1)
    if model_name == 'nhits_30_1024_hourly':
        model = NHITS(h=horizon,
                        input_size=3*horizon,
                        dropout_prob_theta=0.2,
                        stack_types=3*['identity'],
                        mlp_units=3 * [[1024, 1024, 1024, 1024]],
                        n_blocks=3*[10],
                        n_pool_kernel_size=3*[1],
                        n_freq_downsample=[12, 2, 1],
                        loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=5,
                        val_check_steps=200,
                        scaler_type='minmax1',
                        max_steps=5000,
                        batch_size=128,
                        windows_batch_size=1024,
                        random_seed=1)
    if model_name == 'nhits_30_1024_daily':
        model = NHITS(h=horizon,
                        input_size=3*horizon,
                        dropout_prob_theta=0.2,
                        stack_types=3*['identity'],
                        mlp_units=3 * [[1024, 1024, 1024, 1024]],
                        n_blocks=3*[10],
                        n_pool_kernel_size=3*[1],
                        n_freq_downsample=[4, 2, 1],
                        loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=-1,
                        val_check_steps=1000,
                        scaler_type='minmax1',
                        max_steps=200,
                        batch_size=128,
                        windows_batch_size=1024,
                        random_seed=1)
    if model_name == 'nhits_30_1024_weekly':
        model = NHITS(h=horizon,
                        input_size=52*horizon,
                        dropout_prob_theta=0.2,
                        stack_types=3*['identity'],
                        mlp_units=3 * [[1024, 1024, 1024, 1024]],
                        n_blocks=3*[10],
                        n_pool_kernel_size=3*[1],
                        n_freq_downsample=[1, 1, 1],
                        loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=-1,
                        val_check_steps=1000,
                        scaler_type='minmax1',
                        max_steps=200,
                        batch_size=128,
                        windows_batch_size=1024,
                        random_seed=1)
    if model_name == 'nhits_30_1024_monthly':
        model = NHITS(h=horizon,
                        input_size=3*horizon,
                        dropout_prob_theta=0.2,
                        stack_types=3*['identity'],
                        mlp_units=3 * [[1024, 1024, 1024, 1024]],
                        n_blocks=3*[10],
                        n_pool_kernel_size=3*[1],
                        n_freq_downsample=[6, 2, 1],
                        loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=-1,
                        val_check_steps=1000,
                        scaler_type='minmax1',
                        max_steps=200,
                        batch_size=128,
                        windows_batch_size=1024,
                        random_seed=1)
    if model_name == 'nhits_30_1024_quarterly':
        model = NHITS(h=horizon,
                        input_size=3*horizon,
                        dropout_prob_theta=0.2,
                        stack_types=3*['identity'],
                        mlp_units=3 * [[1024, 1024, 1024, 1024]],
                        n_blocks=3*[10],
                        n_pool_kernel_size=3*[1],
                        n_freq_downsample=[1, 1, 1],
                        loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=5,
                        val_check_steps=200,
                        scaler_type='minmax1',
                        max_steps=5000,
                        batch_size=128,
                        windows_batch_size=1024,
                        random_seed=1)
    if model_name == 'nhits_30_1024_yearly':
        model = NHITS(h=horizon,
                        input_size=3*horizon,
                        dropout_prob_theta=0.2,
                        stack_types=3*['identity'],
                        mlp_units=3 * [[1024, 1024, 1024, 1024]],
                        n_blocks=3*[10],
                        n_pool_kernel_size=3*[1],
                        n_freq_downsample=[1, 1, 1],
                        loss=LOSS,
                        learning_rate=1e-4,
                        early_stop_patience_steps=5,
                        val_check_steps=200,
                        scaler_type='minmax1',
                        max_steps=5000,
                        batch_size=128,
                        windows_batch_size=1024,
                        random_seed=1)

        
    ########################################################## TFT ##########################################################
    if model_name == 'tft_1024_minutely':
        model = TFT(h=horizon,
                    input_size=5*horizon,
                    hidden_size=1024,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=5,
                    val_check_steps=100,
                    scaler_type='minmax1',
                    max_steps=1000,
                    batch_size=32,
                    windows_batch_size=128,
                    random_seed=1)
    if model_name == 'tft_1024_10minutely':
        model = TFT(h=horizon,
                    input_size=3*horizon,
                    hidden_size=1024,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=5,
                    val_check_steps=100,
                    scaler_type='minmax1',
                    max_steps=1000,
                    batch_size=32,
                    windows_batch_size=128,
                    random_seed=1)
    if model_name == 'tft_1024_15minutely':
        model = TFT(h=horizon,
                    input_size=5*horizon,
                    hidden_size=1024,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=5,
                    val_check_steps=100,
                    scaler_type='minmax1',
                    max_steps=1000,
                    batch_size=32,
                    windows_batch_size=128,
                    random_seed=1)
    if model_name == 'tft_1024_30minutely':
        model = TFT(h=horizon,
                    input_size=5*horizon,
                    hidden_size=1024,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=5,
                    val_check_steps=100,
                    scaler_type='minmax1',
                    max_steps=1000,
                    batch_size=32,
                    windows_batch_size=128,
                    random_seed=1)
    if model_name == 'tft_1024_hourly':
        horizon = 24
        model = TFT(h=horizon,
                    input_size=3*horizon,
                    hidden_size=1024,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=5,
                    val_check_steps=200,
                    scaler_type='minmax1',
                    max_steps=5000,
                    batch_size=128,
                    windows_batch_size=1024,
                    random_seed=1)
    if model_name == 'tft_1024_daily':
        model = TFT(h=horizon,
                    input_size=3*horizon,
                    hidden_size=1024,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=-1,
                    val_check_steps=1000,
                    scaler_type='minmax1',
                    max_steps=200,
                    batch_size=128,
                    windows_batch_size=1024,
                    random_seed=1)
    if model_name == 'tft_1024_weekly':
        model = TFT(h=horizon,
                    input_size=52*horizon,
                    hidden_size=1024,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=-1,
                    val_check_steps=1000,
                    scaler_type='minmax1',
                    max_steps=200,
                    batch_size=128,
                    windows_batch_size=1024,
                    random_seed=1)
    if model_name == 'tft_1024_monthly':
        model = TFT(h=horizon,
                    input_size=3*horizon,
                    hidden_size=1024,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=-1,
                    val_check_steps=1000,
                    scaler_type='minmax1',
                    max_steps=200,
                    batch_size=128,
                    windows_batch_size=1024,
                    random_seed=1)
    if model_name == 'tft_1024_quarterly':
        model = TFT(h=horizon,
                    input_size=3*horizon,
                    hidden_size=1024,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=5,
                    val_check_steps=200,
                    scaler_type='minmax1',
                    max_steps=5000,
                    batch_size=128,
                    windows_batch_size=1024,
                    random_seed=1)
    if model_name == 'tft_1024_yearly':
        model = TFT(h=horizon,
                    input_size=3*horizon,
                    hidden_size=1024,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=5,
                    val_check_steps=200,
                    scaler_type='minmax1',
                    max_steps=5000,
                    batch_size=128,
                    windows_batch_size=1024,
                    random_seed=1)

    ########################################################## LSTM ##########################################################
    if model_name == 'lstm_512_4_minutely':
        model = LSTM(h=horizon,
                    input_size=5*horizon,
                    inference_input_size=5*horizon,
                    encoder_n_layers=4,
                    encoder_hidden_size=512,
                    context_size=16,
                    decoder_hidden_size=128,
                    decoder_layers=2,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=5,
                    val_check_steps=200,
                    scaler_type='minmax1',
                    max_steps=3000,
                    batch_size=128,
                    random_seed=1)
    if model_name == 'lstm_512_4_10minutely':
        model = LSTM(h=horizon,
                    input_size=3*horizon,
                    inference_input_size=3*horizon,
                    encoder_n_layers=4,
                    encoder_hidden_size=512,
                    context_size=16,
                    decoder_hidden_size=128,
                    decoder_layers=2,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=5,
                    val_check_steps=200,
                    scaler_type='minmax1',
                    max_steps=3000,
                    batch_size=128,
                    random_seed=1)
    if model_name == 'lstm_512_4_15minutely':
        model = LSTM(h=horizon,
                    input_size=5*horizon,
                    inference_input_size=5*horizon,
                    encoder_n_layers=4,
                    encoder_hidden_size=512,
                    context_size=16,
                    decoder_hidden_size=128,
                    decoder_layers=2,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=5,
                    val_check_steps=200,
                    scaler_type='minmax1',
                    max_steps=3000,
                    batch_size=128,
                    random_seed=1)
    if model_name == 'lstm_512_4_30minutely':
        model = LSTM(h=horizon,
                    input_size=5*horizon,
                    inference_input_size=5*horizon,
                    encoder_n_layers=4,
                    encoder_hidden_size=512,
                    context_size=16,
                    decoder_hidden_size=128,
                    decoder_layers=2,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=5,
                    val_check_steps=200,
                    scaler_type='minmax1',
                    max_steps=3000,
                    batch_size=128,
                    random_seed=1)
    if model_name == 'lstm_512_4_hourly':
        model = LSTM(h=horizon,
                    input_size=5*horizon,
                    inference_input_size=5*horizon,
                    encoder_n_layers=4,
                    encoder_hidden_size=512,
                    context_size=16,
                    decoder_hidden_size=128,
                    decoder_layers=2,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=5,
                    val_check_steps=200,
                    scaler_type='minmax1',
                    max_steps=5000,
                    batch_size=128,
                    random_seed=1)
    if model_name == 'lstm_512_4_daily':
        model = LSTM(h=horizon,
                    input_size=5*horizon,
                    inference_input_size=5*horizon,
                    encoder_n_layers=4,
                    encoder_hidden_size=512,
                    context_size=16,
                    decoder_hidden_size=128,
                    decoder_layers=2,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=-1,
                    val_check_steps=1000,
                    scaler_type='minmax1',
                    max_steps=200,
                    batch_size=256,
                    random_seed=1)
    if model_name == 'lstm_512_4_weekly':
        model = LSTM(h=horizon,
                    input_size=52*horizon,
                    inference_input_size=52*horizon,
                    encoder_n_layers=4,
                    encoder_hidden_size=512,
                    context_size=16,
                    decoder_hidden_size=128,
                    decoder_layers=2,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=-1,
                    val_check_steps=1000,
                    scaler_type='minmax1',
                    max_steps=200,
                    batch_size=256,
                    random_seed=1)
    if model_name == 'lstm_512_4_monthly':
        model = LSTM(h=horizon,
                    input_size=3*horizon,
                    inference_input_size=3*horizon,
                    encoder_n_layers=4,
                    encoder_hidden_size=512,
                    context_size=16,
                    decoder_hidden_size=128,
                    decoder_layers=2,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=-1,
                    val_check_steps=1000,
                    scaler_type='minmax1',
                    max_steps=200,
                    batch_size=256,
                    random_seed=1)
    if model_name == 'lstm_512_4_quarterly':
        model = LSTM(h=horizon,
                    input_size=3*horizon,
                    inference_input_size=3*horizon,
                    encoder_n_layers=4,
                    encoder_hidden_size=512,
                    context_size=16,
                    decoder_hidden_size=128,
                    decoder_layers=2,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=5,
                    val_check_steps=200,
                    scaler_type='minmax1',
                    max_steps=5000,
                    batch_size=128,
                    random_seed=1)
    if model_name == 'lstm_512_4_yearly':
        model = LSTM(h=horizon,
                    input_size=5*horizon,
                    inference_input_size=5*horizon,
                    encoder_n_layers=4,
                    encoder_hidden_size=512,
                    context_size=16,
                    decoder_hidden_size=128,
                    decoder_layers=2,
                    loss=LOSS,
                    learning_rate=1e-4,
                    early_stop_patience_steps=5,
                    val_check_steps=200,
                    scaler_type='minmax1',
                    max_steps=5000,
                    batch_size=128,
                    random_seed=1)

    ########################################################## DeepAR ##########################################################
    if model_name == 'deepar_128_4_minutely':
        model = DeepAR(h=horizon,
                       input_size=5*horizon,
                       lstm_n_layers=4,
                       lstm_hidden_size=128,
                       lstm_dropout=0.1,
                       loss=LOSS_PROBA,
                       learning_rate=1e-4,
                       early_stop_patience_steps=5,
                       val_check_steps=200,
                       scaler_type='minmax1',
                       max_steps=3000,
                       batch_size=32,
                       windows_batch_size=128,
                       random_seed=1)
    if model_name == 'deepar_128_4_10minutely':
        model = DeepAR(h=horizon,
                        input_size=3*horizon,
                        lstm_n_layers=4,
                        lstm_hidden_size=128,
                        lstm_dropout=0.1,
                        loss=LOSS_PROBA,
                        learning_rate=1e-4,
                        early_stop_patience_steps=5,
                        val_check_steps=200,
                        scaler_type='minmax1',
                        max_steps=3000,
                        batch_size=32,
                        windows_batch_size=128,
                        random_seed=1)
    if model_name == 'deepar_128_4_15minutely':
        model = DeepAR(h=horizon,
                        input_size=5*horizon,
                        lstm_n_layers=4,
                        lstm_hidden_size=128,
                        lstm_dropout=0.1,
                        loss=LOSS_PROBA,
                        learning_rate=1e-4,
                        early_stop_patience_steps=5,
                        val_check_steps=200,
                        scaler_type='minmax1',
                        max_steps=3000,
                        batch_size=32,
                        windows_batch_size=128,
                        random_seed=1)
    if model_name == 'deepar_128_4_30minutely':
        model = DeepAR(h=horizon,
                        input_size=5*horizon,
                        lstm_n_layers=4,
                        lstm_hidden_size=128,
                        lstm_dropout=0.1,
                        loss=LOSS_PROBA,
                        learning_rate=1e-4,
                        early_stop_patience_steps=5,
                        val_check_steps=200,
                        scaler_type='minmax1',
                        max_steps=3000,
                        batch_size=32,
                        windows_batch_size=128,
                        random_seed=1)
    if model_name == 'deepar_128_4_hourly':
        model = DeepAR(h=horizon,
                        input_size=5*horizon,
                        lstm_n_layers=4,
                        lstm_hidden_size=128,
                        lstm_dropout=0.1,
                        loss=LOSS_PROBA,
                        learning_rate=1e-4,
                        early_stop_patience_steps=5,
                        val_check_steps=200,
                        scaler_type='minmax1',
                        max_steps=5000,
                        batch_size=128,
                        windows_batch_size=1024,
                        random_seed=1)
    if model_name == 'deepar_128_4_daily':
        model = DeepAR(h=horizon,
                        input_size=5*horizon,
                        lstm_n_layers=4,
                        lstm_hidden_size=128,
                        lstm_dropout=0.1,
                        loss=LOSS_PROBA,
                        learning_rate=1e-4,
                        early_stop_patience_steps=-1,
                        val_check_steps=1000,
                        scaler_type='minmax1',
                        max_steps=200,
                        batch_size=128,
                        windows_batch_size=1024,
                        random_seed=1)
    if model_name == 'deepar_128_4_weekly':
        model = DeepAR(h=horizon,
                        input_size=52*horizon,
                        lstm_n_layers=4,
                        lstm_hidden_size=128,
                        lstm_dropout=0.1,
                        loss=LOSS_PROBA,
                        learning_rate=1e-4,
                        early_stop_patience_steps=-1,
                        val_check_steps=1000,
                        scaler_type='minmax1',
                        max_steps=200,
                        batch_size=128,
                        windows_batch_size=1024,
                        random_seed=1)
    if model_name == 'deepar_128_4_monthly':
        model = DeepAR(h=horizon,
                        input_size=3*horizon,
                        lstm_n_layers=4,
                        lstm_hidden_size=128,
                        lstm_dropout=0.1,
                        loss=LOSS_PROBA,
                        learning_rate=1e-4,
                        early_stop_patience_steps=-1,
                        val_check_steps=1000,
                        scaler_type='minmax1',
                        max_steps=200,
                        batch_size=128,
                        windows_batch_size=1024,
                        random_seed=1)
    if model_name == 'deepar_128_4_quarterly':
        model = DeepAR(h=horizon,
                        input_size=3*horizon,
                        lstm_n_layers=4,
                        lstm_hidden_size=128,
                        lstm_dropout=0.1,
                        loss=LOSS_PROBA,
                        learning_rate=1e-4,
                        early_stop_patience_steps=5,
                        val_check_steps=200,
                        scaler_type='minmax1',
                        max_steps=5000,
                        batch_size=128,
                        windows_batch_size=1024,
                        random_seed=1)
    if model_name == 'deepar_128_4_yearly':
        model = DeepAR(h=horizon,
                        input_size=3*horizon,
                        lstm_n_layers=4,
                        lstm_hidden_size=128,
                        lstm_dropout=0.1,
                        loss=LOSS_PROBA,
                        learning_rate=1e-4,
                        early_stop_patience_steps=5,
                        val_check_steps=200,
                        scaler_type='minmax1',
                        max_steps=5000,
                        batch_size=128,
                        windows_batch_size=1024,
                        random_seed=1)
    if model is None:
        raise ValueError("Model name not found.")
    return model