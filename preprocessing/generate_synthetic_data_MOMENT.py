from typing import Tuple

import random
import os
import numpy as np

#from momentfm.utils.utils import control_randomness

def control_randomness(seed: int = 13):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

class SyntheticDataset:
    def __init__(
        self,
        n_periods = 1,
        n_series: int = 1024,
        seq_len: int = 512,
        freq: int = 1,
        freq_range: Tuple[int, int] = (1, 32),
        amplitude_range: Tuple[int, int] = (1, 32),
        trend_range: Tuple[int, int] = (1, 32),
        baseline_range: Tuple[int, int] = (1, 32),
        phase_range: Tuple[int, int] = (1, 32),
        noise_mean: float = 0.0,
        noise_std: float = 0.1,
        random_seed: int = 42,
    ):
        """
        Class to generate synthetic time series data.

        Parameters 
        ----------
        n_series : int
            Number of samples to generate.
        seq_len : int
            Length of the time series.
        freq : int
            Frequency of the sinusoidal wave.
        freq_range : Tuple[int, int]
            Range of frequencies to generate.
        amplitude_range : Tuple[int, int]
            Range of amplitudes to generate.
        trend_range : Tuple[int, int]
            Range of trends to generate.
        baseline_range : Tuple[int, int]
            Range of baselines to generate.
        phase_range : Tuple[int, int]
            Range of phases to generate.
        noise_mean : float
            Mean of the noise.
        noise_std : float
            Standard deviation of the noise.
        random_seed : int
            Random seed to control randomness.        
        """
        self.n_periods = n_periods
        self.n_series = n_series
        self.seq_len = seq_len
        self.freq = freq
        self.freq_range = freq_range
        self.amplitude_range = amplitude_range
        self.trend_range = trend_range
        self.baseline_range = baseline_range
        self.phase_range = phase_range
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.random_seed = random_seed

        control_randomness(self.random_seed)

    def __repr__(self):
        return (
            f"SyntheticDataset(n_series={self.n_series},"
            + f"seq_len={self.seq_len},"
            + f"freq={self.freq},"
            + f"freq_range={self.freq_range},"
            + f"amplitude_range={self.amplitude_range},"
            + f"trend_range={self.trend_range},"
            + f"baseline_range={self.baseline_range},"
            + f"noise_mean={self.noise_mean},"
            + f"noise_std={self.noise_std},"
        )

    def _generate_noise(self):
        epsilon = np.random.normal(
            loc=self.noise_mean,
            scale=self.noise_std,
            size=(self.n_series, self.seq_len),
        )
        return epsilon

    def _generate_x(self):
        t = np.linspace(start=0, stop=self.n_periods, num=self.seq_len)
        t = np.tile(t, (self.n_series, 1))
        x = 2 * self.freq * np.pi * t
        return x, t

    def gen_sinusoids_with_varying_freq(self):
        c = np.linspace(
            start=self.freq_range[0], stop=self.freq_range[1], num=self.n_series
        )
        c = np.tile(c[:, np.newaxis], (1, self.seq_len))
        x, _ = self._generate_x()
        epsilon = self._generate_noise()

        y = np.sin(c * x) + epsilon
        y = y[:, np.newaxis, :]
        return y, c

    def gen_sinusoids_with_varying_correlation(self):
        c = np.linspace(start=0, stop=2 * np.pi, num=self.n_series)
        c = np.tile(c[:, np.newaxis], (1, self.seq_len))
        x, _ = self._generate_x()
        epsilon = self._generate_noise()

        y = np.sin(x + c) + epsilon
        y = y[:, np.newaxis, :]
        return y, c

    def gen_sinusoids_with_varying_amplitude(self):
        c = np.linspace(
            start=self.amplitude_range[0],
            stop=self.amplitude_range[1],
            num=self.n_series,
        )
        c = np.tile(c[:, np.newaxis], (1, self.seq_len))
        x, _ = self._generate_x()
        epsilon = self._generate_noise()

        y = c * np.sin(x) + epsilon
        y = y[:, np.newaxis, :]
        return y, c

    def gen_sinusoids_with_varying_trend(self):
        c = np.linspace(
            start=self.trend_range[0], stop=self.trend_range[1], num=self.n_series
        )
        c = np.tile(c[:, np.newaxis], (1, self.seq_len))
        x, t = self._generate_x()
        epsilon = self._generate_noise()

        y = np.sin(x) + t*c + epsilon  # was originally t**c but this result in exponential growth/decay
        y = y[:, np.newaxis, :]
        return y, c

    def gen_sinusoids_with_varying_baseline(self):
        c = np.linspace(
            start=self.baseline_range[0],
            stop=self.baseline_range[1],
            num=self.n_series,
        )
        c = np.tile(c[:, np.newaxis], (1, self.seq_len))
        x, _ = self._generate_x()
        epsilon = self._generate_noise()

        y = np.sin(x) + c + epsilon
        y = y[:, np.newaxis, :]
        return y, c

    def generate_signal(self, amp, freq, phase, baseline, trend, epsilon, func=np.sin):
        """
        Generate a signal based on the given parameters, with the option to switch between sin and cos.
        
        Parameters:
        - x: time array or input values
        - amp: amplitude
        - freq: frequency
        - corr: phase shift (correlation)
        - trend: trend term to add
        - baseline: baseline offset
        - epsilon: random noise added to the signal
        - func: function to apply (np.sin or np.cos)
        
        Returns:
        - y: generated signal
        """

        l = amp.shape[0]
        x, t = self._generate_x()
        x = x[:l, :]
        t = t[:l, :]
        epsilon = self._generate_noise()
        epsilon = epsilon[:l, :]
        
        y = amp * func(freq * x + phase) + baseline + trend * t + epsilon
        y = y[:, np.newaxis, :]

        return y
