import sys
import pandas as pd
import numpy as np

sys.path.append('../preprocessing/')
from generate_synthetic_datasets import SyntheticDataset_Composition


def get_datasets(args):
    if args.dataset_name == 'synthetic_sinusoid_composition':
        sdc = SyntheticDataset_Composition(n_periods=1,
                                           n_series=args.n_series,
                                           seq_len=args.seq_len,
                                           n_compositions=args.n_compositions,
                                           ood_ratio=args.ood_ratio,
                                           noise_mean=0,
                                           noise_std=0,
                                           freq_range=(3, 32), 
                                           amplitude_range=(1, 32),
                                           include_freq=True, 
                                           include_amp=True, 
                                           include_phase=False, 
                                           include_baseline=False, 
                                           include_trend=False,
                                          )
        aggregate_df = sdc.harmonic_composition()
        num_bases = int(args.n_compositions*2) # after data is generated account for 2x bases

    elif args.dataset_name == 'ettm2_100_series':
        aggregate_df = pd.read_csv('../data/ETTm2_100_series.csv')
        num_bases = args.n_compositions

    elif args.dataset_name == 'ecl_100_series':
        aggregate_df = pd.read_csv('../data/ECL_100_series.csv')
        num_bases = args.n_compositions
    
    elif args.dataset_name == 'solar1h_100_series':
        aggregate_df = pd.read_csv('../data/solar1h_100_series.csv')
        num_bases = args.n_compositions

    elif args.dataset_name == 'subseasonal_100_series':
        aggregate_df = pd.read_csv('../data/subseasonal_100_series.csv')
        num_bases = args.n_compositions

    elif args.dataset_name == 'loopseattle_100_series':
        aggregate_df = pd.read_csv('../data/loopseattle_100_series.csv')
        num_bases = args.n_compositions

    # get component dataset
    component_df = _create_component_df(aggregate_df,
                                        args.seq_len,
                                        num_bases,
                                        args.freq, 
                                        dft_coefficients=False,
                                       )

    return component_df, aggregate_df



def _create_component_df(aggregate_df, seq_len, top_k, freq, dft_coefficients):

    ds = pd.date_range(start=aggregate_df.ds.min(),
                       periods=seq_len, 
                       freq=freq,
                      )
    
    n_series = aggregate_df.shape[0] // seq_len
    signals = aggregate_df.y.values.reshape(n_series, seq_len)
    basis_functions = get_fft_bases(signals, top_k, dft_coefficients, seq_len)

    num_bases = basis_functions.shape[0]
    names = [f'basis{i}' for i in range(1, num_bases+1)]
    names = np.concatenate([np.repeat(i, seq_len) for i in names])
    
    df = pd.DataFrame(basis_functions.flatten(),
                     index=names,
                     columns=['y']
                     )
    df.index.name = 'unique_id'
    df['ds'] = np.concatenate([ds for _ in range(num_bases)])
    df.reset_index(drop=False, inplace=True)

    return df

def get_fft_bases(x, top_k, dft_coefficients, seq_len):
    seq_len = x.shape[-1]
    ts = 1.0 / seq_len
    t = np.arange(1e-5, 1, ts)

    # Compute the DFT of the input signals
    dft = np.fft.fft(x, n=seq_len, axis=-1, norm='ortho')
    ks = np.arange(seq_len)

    # Compute magnitudes and sort by magnitude
    dft_magnitudes = np.abs(dft)  # Compute the absolute value
    sorted_indices = np.argsort(dft_magnitudes, axis=-1)[:, ::-1]  # Sort in descending order
    top_kvals = sorted_indices[:, :top_k]

    top_dft = np.take_along_axis(dft, top_kvals, axis=1)
    top_dft = np.expand_dims(top_dft, axis=-1).repeat(seq_len, axis=-1)

    # # Prepare for basis function computation
    if dft_coefficients==False:
        top_kvals = np.unique(top_kvals) # if we don't care about dft coefficients
        top_kvals = np.expand_dims(top_kvals, 0)

    top_kvals = np.expand_dims(top_kvals, axis=-1).repeat(seq_len, axis=-1)

    # Compute the basis functions
    t = np.expand_dims(t, axis=0)  # Match dimensions with x
    bs = np.exp(-2j * np.pi * top_kvals * t)  # [num_signals, top_k, seq_len]
    bs = np.flip(bs, axis=-1)  # Flip along the sequence dimension

    if dft_coefficients==False:
        basis_functions = bs / np.sqrt(seq_len)  # Normalize
    else:
        basis_functions = (top_dft*bs) / np.sqrt(seq_len)  # Normalize

    # correct for shift of 1
    basis_functions = np.pad(basis_functions, ((0, 0), (0, 0), (1, 0)), mode='edge')
    basis_functions = basis_functions[:, :, :seq_len]
        
    basis_functions = basis_functions.reshape(-1, basis_functions.shape[-1])

    return basis_functions
