from generate_synthetic_data_MOMENT import SyntheticDataset
import numpy as np
import pandas as pd
import re
import itertools

class SyntheticDataset_Composition(SyntheticDataset):
    def __init__(self, n_periods=1, ood_ratio=0, include_freq=True, include_amp=False, 
                 include_phase=False, include_baseline=False, include_trend=False, 
                 n_compositions=1, *args, **kwargs):
        super(SyntheticDataset_Composition, self).__init__(*args, **kwargs)

        self.dataset_generator = SyntheticDataset(
            n_periods=self.n_periods,
            n_series=self.n_series,
            seq_len=self.seq_len,
            freq=self.freq,
            freq_range=self.freq_range,
            amplitude_range=self.amplitude_range,
            trend_range=self.trend_range,
            baseline_range=self.baseline_range,
            phase_range=self.phase_range,
            noise_mean=self.noise_mean,
            noise_std=self.noise_std,
            random_seed=self.random_seed
        )

        self.include_corr = include_phase
        self.include_freq = include_freq
        self.include_amp = include_amp
        self.include_baseline = include_baseline
        self.include_phase = include_phase
        self.include_trend = include_trend
        self.ood_ratio = ood_ratio
        self.n_compositions = n_compositions
        self.ds = pd.date_range(start='2024-01-01', 
                                periods=self.seq_len, 
                                freq='min'
                               )

        self.rng = np.random.RandomState(self.random_seed)

    def get_random_function_parameters(self):
        if self.include_amp==True:
            amp_range = np.linspace(self.amplitude_range[0],
                                    self.amplitude_range[1],
                                    30)
                                    #self.amplitude_range[1]-self.amplitude_range[0]+1)
        else:
            amp_range = 1

        if self.include_baseline==True:
            baseline_range = np.linspace(self.baseline_range[0],
                                         self.baseline_range[1],
                                         30)
                                    #self.baseline_range[1]-self.baseline_range[0]+1)
        else:
            baseline_range = 0

        if self.include_phase==True:
            phase_range = np.linspace(self.phase_range[0],
                                      self.phase_range[1],
                                      30)
                                    #self.phase_range[1]-self.phase_range[0]+1)
        else:
            phase_range = 0

        if self.include_freq==True:
            freq_range = np.linspace(self.freq_range[0],
                                     self.freq_range[1],
                                     30)
                                    #self.freq_range[1]-self.freq_range[0]+1)
        else:
            freq_range = 1

        if self.include_trend==True:
            trend_range = np.linspace(self.trend_range[0],
                                     self.trend_range[1],
                                      30)
                                    #self.trend_range[1]-self.trend_range[0]+1)
        else:
            trend_range = 0

        parameter_combinations = np.array(np.meshgrid(amp_range, 
                                                      freq_range, 
                                                      phase_range, 
                                                      baseline_range, 
                                                      trend_range)).T.reshape(-1, 5)

        l = parameter_combinations.shape[0]
        N = int(self.n_series)
        n = int(self.n_compositions)

        sampled_parameters = parameter_combinations[self.rng.choice(l,
                                                                    N*n, 
                                                                    replace=False,
                                                                   )
                                                   ]
        sampled_parameters = sampled_parameters.reshape(N, n, 5)
        sampled_functions = np.random.choice([np.sin, np.cos], size=(N, n))

        return sampled_parameters, sampled_functions

    def fsc_function(self, parameters, functions, n_series):
        n = int(self.n_compositions)
        N = n_series
        signals = np.zeros((N, 1, self.seq_len))
        for i in range(n):
            amp = parameters[:, i][:, 0]
            amp = np.tile(amp, (self.seq_len, 1)).T
            
            freq = parameters[:, i][:, 1]
            freq = np.tile(freq, (self.seq_len, 1)).T
               
            phase = parameters[:, i][:, 2]
            phase = np.tile(phase, (self.seq_len, 1)).T

            baseline = parameters[:, i][:, 3]
            baseline = np.tile(baseline, (self.seq_len, 1)).T
    
            trend = parameters[:, i][:, 4]
            trend = np.tile(trend, (self.seq_len, 1)).T

            func = functions[:, i]

            new_term = self.generate_signal(amp, freq, phase, baseline, trend, func)
            signals += new_term

        return signals

    def harmonic_composition(self):
        sampled_parameters, sampled_functions = self.get_random_function_parameters()
        agg_signals = self.fsc_function(sampled_parameters, sampled_functions, n_series=self.n_series)

        aggregate_df = pd.DataFrame(agg_signals.flatten(), columns=['y'])
        names = [f'composite{i}' for i in range(1, self.n_series+1)]
        aggregate_df['unique_id'] = np.concatenate([np.repeat(name, self.seq_len) for name in names])
        aggregate_df['ds'] = np.concatenate([self.ds for _ in range(self.n_series)])
       
        return aggregate_df

