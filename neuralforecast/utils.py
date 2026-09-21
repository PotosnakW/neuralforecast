__all__ = ['AirPassengers', 'AirPassengersDF', 'unique_id', 'ds', 'y', 'AirPassengersPanel', 'snaive', 'airline1_dummy',
           'airline2_dummy', 'AirPassengersStatic', 'generate_series', 'TimeFeature', 'SecondOfMinute', 'MinuteOfHour',
           'HourOfDay', 'DayOfWeek', 'DayOfMonth', 'DayOfYear', 'MonthOfYear', 'WeekOfYear',
           'time_features_from_frequency_str', 'augment_calendar_df', 'get_indexer_raise_missing',
           'PredictionIntervals', 'add_conformal_distribution_intervals', 'add_conformal_error_intervals',
           'get_prediction_interval_method', 'level_to_quantiles', 'quantiles_to_level',
           'estimate_ar1_rho', 'interp_2d', 'gaussian_copula_sample',
           'schaake_shuffle_sample', 'sample_from_quantiles',
           'DEFAULT_QUANTILE_GRID', 'VALID_SIMULATION_METHODS',
           'extract_y_hist', 'ensemble_forecast_windows']


import math
import random
import warnings
from itertools import chain
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from utilsforecast.compat import DFType

from neuralforecast.losses.numpy import _reshape_windows_by_date


def generate_series(
    n_series: int,
    freq: str = "D",
    min_length: int = 50,
    max_length: int = 500,
    n_temporal_features: int = 0,
    n_static_features: int = 0,
    equal_ends: bool = False,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate Synthetic Panel Series.

    Generates `n_series` of frequency `freq` of different lengths in the interval [`min_length`, `max_length`].
    If `n_temporal_features > 0`, then each serie gets temporal features with random values.
    If `n_static_features > 0`, then a static dataframe is returned along the temporal dataframe.
    If `equal_ends == True` then all series end at the same date.

    Args:
        n_series (int): Number of series for synthetic panel.
        freq (str, optional): Frequency of the data, panda's available frequencies. Defaults to "D".
        min_length (int, optional): Minimal length of synthetic panel's series. Defaults to 50.
        max_length (int, optional): Maximal length of synthetic panel's series. Defaults to 500.
        n_temporal_features (int, optional): Number of temporal exogenous variables for synthetic panel's series. Defaults to 0.
        n_static_features (int, optional): Number of static exogenous variables for synthetic panel's series. Defaults to 0.
        equal_ends (bool, optional): If True, series finish in the same date stamp `ds`. Defaults to False.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.

    Returns:
        pd.DataFrame: Synthetic panel with columns [`unique_id`, `ds`, `y`] and exogenous.
    """
    seasonalities = {"D": 7, "M": 12}
    season = seasonalities[freq]

    rng = np.random.RandomState(seed)
    series_lengths = rng.randint(min_length, max_length + 1, n_series)
    total_length = series_lengths.sum()

    dates = pd.date_range("2000-01-01", periods=max_length, freq=freq).values
    uids = [np.repeat(i, serie_length) for i, serie_length in enumerate(series_lengths)]
    if equal_ends:
        ds = [dates[-serie_length:] for serie_length in series_lengths]
    else:
        ds = [dates[:serie_length] for serie_length in series_lengths]

    y = np.arange(total_length) % season + rng.rand(total_length) * 0.5
    temporal_df = pd.DataFrame(
        dict(unique_id=chain.from_iterable(uids), ds=chain.from_iterable(ds), y=y)
    )

    random.seed(seed)
    for i in range(n_temporal_features):
        random.seed(seed)
        temporal_values = [
            [random.randint(0, 100)] * serie_length for serie_length in series_lengths
        ]
        temporal_df[f"temporal_{i}"] = np.hstack(temporal_values)
        temporal_df[f"temporal_{i}"] = temporal_df[f"temporal_{i}"].astype("category")
        if i == 0:
            temporal_df["y"] = temporal_df["y"] * (
                1 + temporal_df[f"temporal_{i}"].cat.codes
            )

    temporal_df["unique_id"] = temporal_df["unique_id"].astype("category")
    temporal_df["unique_id"] = temporal_df["unique_id"].cat.as_ordered()

    if n_static_features > 0:
        static_features = np.random.uniform(
            low=0.0, high=1.0, size=(n_series, n_static_features)
        )
        static_df = pd.DataFrame.from_records(
            static_features, columns=[f"static_{i}" for i in range(n_static_features)]
        )

        static_df["unique_id"] = np.arange(n_series)
        static_df["unique_id"] = static_df["unique_id"].astype("category")
        static_df["unique_id"] = static_df["unique_id"].cat.as_ordered()

        return temporal_df, static_df

    return temporal_df


AirPassengers = np.array(
    [
        112.0,
        118.0,
        132.0,
        129.0,
        121.0,
        135.0,
        148.0,
        148.0,
        136.0,
        119.0,
        104.0,
        118.0,
        115.0,
        126.0,
        141.0,
        135.0,
        125.0,
        149.0,
        170.0,
        170.0,
        158.0,
        133.0,
        114.0,
        140.0,
        145.0,
        150.0,
        178.0,
        163.0,
        172.0,
        178.0,
        199.0,
        199.0,
        184.0,
        162.0,
        146.0,
        166.0,
        171.0,
        180.0,
        193.0,
        181.0,
        183.0,
        218.0,
        230.0,
        242.0,
        209.0,
        191.0,
        172.0,
        194.0,
        196.0,
        196.0,
        236.0,
        235.0,
        229.0,
        243.0,
        264.0,
        272.0,
        237.0,
        211.0,
        180.0,
        201.0,
        204.0,
        188.0,
        235.0,
        227.0,
        234.0,
        264.0,
        302.0,
        293.0,
        259.0,
        229.0,
        203.0,
        229.0,
        242.0,
        233.0,
        267.0,
        269.0,
        270.0,
        315.0,
        364.0,
        347.0,
        312.0,
        274.0,
        237.0,
        278.0,
        284.0,
        277.0,
        317.0,
        313.0,
        318.0,
        374.0,
        413.0,
        405.0,
        355.0,
        306.0,
        271.0,
        306.0,
        315.0,
        301.0,
        356.0,
        348.0,
        355.0,
        422.0,
        465.0,
        467.0,
        404.0,
        347.0,
        305.0,
        336.0,
        340.0,
        318.0,
        362.0,
        348.0,
        363.0,
        435.0,
        491.0,
        505.0,
        404.0,
        359.0,
        310.0,
        337.0,
        360.0,
        342.0,
        406.0,
        396.0,
        420.0,
        472.0,
        548.0,
        559.0,
        463.0,
        407.0,
        362.0,
        405.0,
        417.0,
        391.0,
        419.0,
        461.0,
        472.0,
        535.0,
        622.0,
        606.0,
        508.0,
        461.0,
        390.0,
        432.0,
    ],
    dtype=np.float32,
)


AirPassengersDF = pd.DataFrame(
    {
        "unique_id": np.ones(len(AirPassengers)),
        "ds": pd.date_range(
            start="1949-01-01", periods=len(AirPassengers), freq=pd.offsets.MonthEnd()
        ),
        "y": AirPassengers,
    }
)


# Declare Panel Data
unique_id = np.concatenate(
    [["Airline1"] * len(AirPassengers), ["Airline2"] * len(AirPassengers)]
)
ds = np.tile(
    pd.date_range(
        start="1949-01-01", periods=len(AirPassengers), freq=pd.offsets.MonthEnd()
    ).to_numpy(),
    2,
)
y = np.concatenate([AirPassengers, AirPassengers + 300])

AirPassengersPanel = pd.DataFrame({"unique_id": unique_id, "ds": ds, "y": y})

# For future exogenous variables
# Declare SeasonalNaive12 and fill first 12 values with y
snaive = (
    AirPassengersPanel.groupby("unique_id")["y"]
    .shift(periods=12)
    .reset_index(drop=True)
)
AirPassengersPanel["trend"] = range(len(AirPassengersPanel))
AirPassengersPanel["y_[lag12]"] = snaive.fillna(AirPassengersPanel["y"])

# Declare Static Data
unique_id = np.array(["Airline1", "Airline2"])
airline1_dummy = [0, 1]
airline2_dummy = [1, 0]
AirPassengersStatic = pd.DataFrame(
    {"unique_id": unique_id, "airline1": airline1_dummy, "airline2": airline2_dummy}
)


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex):
        return print("Overwrite with corresponding feature")

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Second of minute encoded as value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Day of week encoded as value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """Returns a list of time features that will be appropriate for the given frequency string.

    Args:
        freq_str (str): Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.

    Returns:
        List[TimeFeature]: List of time features appropriate for the frequency.
    """

    if freq_str not in ["Q", "M", "MS", "W", "D", "B", "H", "T", "S"]:
        raise Exception("Frequency not supported")

    if freq_str in ["Q", "M", "MS"]:
        return [cls() for cls in [MonthOfYear]]
    elif freq_str == "W":
        return [cls() for cls in [DayOfMonth, WeekOfYear]]
    elif freq_str in ["D", "B"]:
        return [cls() for cls in [DayOfWeek, DayOfMonth, DayOfYear]]
    elif freq_str == "H":
        return [cls() for cls in [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]]
    elif freq_str == "T":
        return [
            cls() for cls in [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]
        ]
    else:
        return [
            cls()
            for cls in [
                SecondOfMinute,
                MinuteOfHour,
                HourOfDay,
                DayOfWeek,
                DayOfMonth,
                DayOfYear,
            ]
        ]


def augment_calendar_df(df, freq="H"):
    """Augment a dataframe with calendar features based on frequency.

    Frequency mappings:
    - Q - [month]
    - M - [month]
    - W - [Day of month, week of year]
    - D - [Day of week, day of month, day of year]
    - B - [Day of week, day of month, day of year]
    - H - [Hour of day, day of week, day of month, day of year]
    - T - [Minute of hour*, hour of day, day of week, day of month, day of year]
    - S - [Second of minute, minute of hour, hour of day, day of week, day of month, day of year]

    *minute returns a number from 0-3 corresponding to the 15 minute period it falls into.

    Args:
        df (pd.DataFrame): DataFrame to augment with calendar features.
        freq (str, optional): Frequency string for determining which features to add. Defaults to "H".

    Returns:
        Tuple[pd.DataFrame, List[str]]: Tuple of (augmented DataFrame, list of feature column names).
    """
    df = df.copy()

    freq_map = {
        "Q": ["month"],
        "M": ["month"],
        "MS": ["month"],
        "W": ["monthday", "yearweek"],
        "D": ["weekday", "monthday", "yearday"],
        "B": ["weekday", "monthday", "yearday"],
        "H": ["dayhour", "weekday", "monthday", "yearday"],
        "T": ["hourminute", "dayhour", "weekday", "monthday", "yearday"],
        "S": [
            "minutesecond",
            "hourminute",
            "dayhour",
            "weekday",
            "monthday",
            "yearday",
        ],
    }

    ds_col = pd.to_datetime(df.ds.values)
    ds_data = np.vstack(
        [feat(ds_col) for feat in time_features_from_frequency_str(freq)]
    ).transpose(1, 0)
    ds_data = pd.DataFrame(ds_data, columns=freq_map[freq])

    return pd.concat([df, ds_data], axis=1), freq_map[freq]


def get_indexer_raise_missing(idx: pd.Index, vals: List[str]) -> List[int]:
    """Get index positions for values, raising error if any are missing.

    Args:
        idx (pd.Index): Index to search in.
        vals (List[str]): Values to find indices for.

    Returns:
        List[int]: List of index positions.

    Raises:
        ValueError: If any values are missing from the index.
    """
    idxs = idx.get_indexer(vals)
    missing = [v for i, v in zip(idxs, vals) if i == -1]
    if missing:
        raise ValueError(f"The following values are missing from the index: {missing}")
    return idxs


class PredictionIntervals:
    """Class for storing prediction intervals metadata information."""

    def __init__(
        self,
        n_windows: int = 2,
        method: str = "conformal_distribution",
        step_size: int = 1,
    ):
        """Initialize PredictionIntervals.

        Args:
            n_windows (int, optional): Number of windows to evaluate. Defaults to 2.
            method (str, optional): One of the supported methods for the computation of prediction intervals:
                conformal_error or conformal_distribution. Defaults to "conformal_distribution".
            step_size (int, optional): Step size between each cross-validation window. Defaults to 1.
        """
        if n_windows < 2:
            raise ValueError(
                "You need at least two windows to compute conformal intervals"
            )
        allowed_methods = ["conformal_error", "conformal_distribution"]
        if method not in allowed_methods:
            raise ValueError(f"method must be one of {allowed_methods}")
        if step_size < 1:
            raise ValueError("step_size must be at least 1")
        self.n_windows = n_windows
        self.method = method
        self.step_size = step_size

    def __repr__(self):
        return (
            f"PredictionIntervals(n_windows={self.n_windows}, method='{self.method}', step_size={self.step_size})"
        )


def add_conformal_distribution_intervals(
    model_fcsts: np.array,
    cs_df: DFType,
    model: str,
    cs_n_windows: int,
    n_series: int,
    horizon: int,
    level: Optional[List[Union[int, float]]] = None,
    quantiles: Optional[List[float]] = None,
) -> Tuple[np.array, List[str]]:
    """Add conformal intervals based on conformal scores using distribution strategy.

    This strategy creates forecast paths based on errors and calculates quantiles using those paths.

    Args:
        model_fcsts (np.array): Model forecasts array.
        cs_df (DFType): DataFrame containing conformal scores.
        model (str): Model name.
        cs_n_windows (int): Number of conformal score windows.
        n_series (int): Number of series.
        horizon (int): Forecast horizon.
        level (Optional[List[Union[int, float]]], optional): Confidence levels for prediction intervals. Defaults to None.
        quantiles (Optional[List[float]], optional): Quantiles for prediction intervals. Defaults to None.

    Returns:
        Tuple[np.array, List[str]]: Tuple of (forecasts with intervals, column names).
    """
    assert (
        level is not None or quantiles is not None
    ), "Either level or quantiles must be provided"

    if quantiles is None and level is not None:
        alphas = [100 - lv for lv in level]
        cuts = [alpha / 200 for alpha in reversed(alphas)]
        cuts.extend(1 - alpha / 200 for alpha in alphas)
    elif quantiles is not None:
        cuts = quantiles

    scores = cs_df[model].to_numpy().reshape(n_series, cs_n_windows, horizon)
    scores = scores.transpose(1, 0, 2)
    # restrict scores to horizon
    scores = scores[:, :, :horizon]
    mean = model_fcsts.reshape(1, n_series, -1)
    scores = np.vstack([mean - scores, mean + scores])
    scores_quantiles = np.quantile(
        scores,
        cuts,
        axis=0,
    )
    scores_quantiles = scores_quantiles.reshape(len(cuts), -1).T
    if quantiles is None and level is not None:
        lo_cols = [f"{model}-lo-{lv}" for lv in reversed(level)]
        hi_cols = [f"{model}-hi-{lv}" for lv in level]
        out_cols = lo_cols + hi_cols
    elif quantiles is not None:
        out_cols = [f"{model}-ql{q}" for q in quantiles]

    fcsts_with_intervals = np.hstack([model_fcsts, scores_quantiles])

    return fcsts_with_intervals, out_cols


def add_conformal_error_intervals(
    model_fcsts: np.array,
    cs_df: DFType,
    model: str,
    cs_n_windows: int,
    n_series: int,
    horizon: int,
    level: Optional[List[Union[int, float]]] = None,
    quantiles: Optional[List[float]] = None,
) -> Tuple[np.array, List[str]]:
    """Add conformal intervals based on conformal scores using error strategy.

    This strategy creates prediction intervals based on absolute errors.

    Args:
        model_fcsts (np.array): Model forecasts array.
        cs_df (DFType): DataFrame containing conformal scores.
        model (str): Model name.
        cs_n_windows (int): Number of conformal score windows.
        n_series (int): Number of series.
        horizon (int): Forecast horizon.
        level (Optional[List[Union[int, float]]], optional): Confidence levels for prediction intervals. Defaults to None.
        quantiles (Optional[List[float]], optional): Quantiles for prediction intervals. Defaults to None.

    Returns:
        Tuple[np.array, List[str]]: Tuple of (forecasts with intervals, column names).
    """
    assert (
        level is not None or quantiles is not None
    ), "Either level or quantiles must be provided"

    if quantiles is None and level is not None:
        alphas = [100 - lv for lv in level]
        cuts = [alpha / 200 for alpha in reversed(alphas)]
        cuts.extend(1 - alpha / 200 for alpha in alphas)
    elif quantiles is not None:
        cuts = quantiles

    mean = model_fcsts.ravel()
    scores = cs_df[model].to_numpy().reshape(n_series, cs_n_windows, horizon)
    scores = scores.transpose(1, 0, 2)
    # restrict scores to horizon
    scores = scores[:, :, :horizon]
    scores_quantiles = np.quantile(
        scores,
        cuts,
        axis=0,
    )
    scores_quantiles = scores_quantiles.reshape(len(cuts), -1)

    if quantiles is None and level is not None:
        lo_cols = [f"{model}-lo-{lv}" for lv in reversed(level)]
        hi_cols = [f"{model}-hi-{lv}" for lv in level]
        out_cols = lo_cols + hi_cols
    else:
        out_cols = [f"{model}-ql{q}" for q in cuts]

    scores_quantiles_ls = []
    for i, q in enumerate(cuts):
        if q < 0.5:
            scores_quantiles_ls.append(mean - scores_quantiles[::-1][i])
        elif q > 0.5:
            scores_quantiles_ls.append(mean + scores_quantiles[i])
        else:
            scores_quantiles_ls.append(mean)
    scores_quantiles = np.vstack(scores_quantiles_ls).T

    fcsts_with_intervals = np.hstack([model_fcsts, scores_quantiles])

    return fcsts_with_intervals, out_cols


def get_prediction_interval_method(method: str):
    """Get the prediction interval method function by name.

    Args:
        method (str): Name of the prediction interval method.

    Returns:
        Callable: The corresponding method function.

    Raises:
        ValueError: If the method is not supported.
    """
    available_methods = {
        "conformal_distribution": add_conformal_distribution_intervals,
        "conformal_error": add_conformal_error_intervals,
    }
    if method not in available_methods.keys():
        raise ValueError(
            f"prediction intervals method {method} not supported "
            f'please choose one of {", ".join(available_methods.keys())}'
        )
    return available_methods[method]


def level_to_quantiles(level: List[Union[int, float]]) -> List[float]:
    """Convert a list of confidence levels to quantiles.

    Args:
        level (List[Union[int, float]]): List of confidence levels (e.g., [80, 90]).

    Returns:
        List[float]: List of corresponding quantiles.
    """
    level_set = set(level)
    return sorted(
        list(
            set(sum([[(50 - l / 2) / 100, (50 + l / 2) / 100] for l in level_set], []))
        )
    )


def quantiles_to_level(quantiles: List[float]) -> List[Union[int, float]]:
    """Convert a list of quantiles to confidence levels.

    Args:
        quantiles (List[float]): List of quantiles (e.g., [0.1, 0.5, 0.9]).

    Returns:
        List[Union[int, float]]: List of corresponding confidence levels.
    """
    quantiles_set = set(quantiles)
    return sorted(
        set(
            [
                int(round(100 - 200 * (q * (q < 0.5) + (1 - q) * (q >= 0.5)), 2))
                for q in quantiles_set
            ]
        )
    )


# ─── Simulation constants ─────────────────────────────────────────────────────
DEFAULT_QUANTILE_GRID = [round(q / 100, 2) for q in range(1, 100)]
VALID_SIMULATION_METHODS = {"gaussian_copula", "schaake_shuffle"}


def estimate_ar1_rho(y):
    """Estimate AR(1) correlation coefficient from differenced series.

    Args:
        y: 1-D tensor of historical values (may contain NaN).

    Returns:
        rho: scalar tensor, clipped to (-0.99, 0.99).
    """
    y = y[~torch.isnan(y)]
    if len(y) < 3:
        return torch.tensor(0.0, dtype=torch.float64)
    y_diff = y[1:] - y[:-1]
    x, z = y_diff[:-1], y_diff[1:]
    x = x - x.mean()
    z = z - z.mean()
    denom = torch.sqrt((x * x).sum() * (z * z).sum())
    if denom < 1e-15:
        return torch.tensor(0.0, dtype=torch.float64)
    rho = (x * z).sum() / denom
    return rho.clamp(-0.99, 0.99)


def interp_2d(x, xp, fp):
    """Vectorised 1-D linear interpolation across rows.

    Each row of ``x`` is interpolated against ``xp`` using the
    corresponding row of ``fp`` as knot values.

    Args:
        x: (N, n_paths) query points in (0, 1).
        xp: (Q,) shared sorted knot positions.
        fp: (N, Q) per-row knot values.

    Returns:
        result: (N, n_paths) interpolated values.
    """
    Q = xp.shape[0]
    idx = torch.searchsorted(xp.expand(x.shape[0], -1), x).clamp(1, Q - 1)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    f0 = fp.gather(1, idx - 1)
    f1 = fp.gather(1, idx)
    denom = x1 - x0
    denom = torch.where(denom.abs() < 1e-15, torch.ones_like(denom), denom)
    t = (x - x0) / denom
    return f0 + t * (f1 - f0)


def gaussian_copula_sample(
    quantile_positions,
    quantile_values,
    y_hist,
    n_paths,
    seed=None,
):
    """Gaussian copula with AR(1) Toeplitz correlation and IQF marginals.

    All computation in torch tensors. Cholesky decompositions and sampling
    are batched across series for efficiency.

    Args:
        quantile_positions: (Q,) sorted quantile levels in (0, 1), tensor.
        quantile_values: (n_series, H, Q) quantile forecast values, tensor.
        y_hist: list of 1-D tensors, historical values per series.
        n_paths: number of sample paths.
        seed: random seed.

    Returns:
        simulations: (n_series, n_paths, H) tensor, dtype of `quantile_values`.
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    n_series, H, Q = quantile_values.shape
    dtype = quantile_values.dtype

    lags = torch.arange(H, dtype=dtype)
    abs_diff = (lags.unsqueeze(1) - lags.unsqueeze(0)).abs()  # (H, H)

    q_lo = quantile_positions[0]
    q_hi = quantile_positions[-1]
    sqrt2 = math.sqrt(2.0)

    # Compute one rho per series (variable-length NaN filtering: kept as loop).
    rhos = torch.stack(
        [estimate_ar1_rho(y).to(dtype) for y in y_hist]
    )  # (n_series,)

    # Build batched Toeplitz correlation matrices: (n_series, H, H)
    R = rhos.abs().view(n_series, 1, 1) ** abs_diff.unsqueeze(0)

    # Batched Cholesky with progressive jitter for numerical stability.
    # torch.linalg.cholesky_ex returns (L, info) without raising; info[i] > 0
    # signals failure for series i. Series that fail all jitters fall back to
    # the identity (independent samples).
    eye = torch.eye(H, dtype=dtype)
    Ls = eye.unsqueeze(0).expand(n_series, -1, -1).clone()  # (n_series, H, H)
    remaining = torch.ones(n_series, dtype=torch.bool)

    for jitter in [1e-8, 1e-6, 1e-4, 1e-2]:
        if not remaining.any():
            break
        rem_idx = remaining.nonzero(as_tuple=True)[0]
        R_j = R[rem_idx] + eye * jitter          # (k, H, H)
        L_cand, info = torch.linalg.cholesky_ex(R_j)
        ok_mask = info == 0
        ok_idx = rem_idx[ok_mask]
        Ls[ok_idx] = L_cand[ok_mask]
        remaining[ok_idx] = False
    # Series still in `remaining` keep the identity Ls (independent samples).

    # Single batched randn draw + correlated transform: (n_series, H, n_paths)
    Z = torch.randn(n_series, H, n_paths, dtype=dtype, generator=gen)
    Y = torch.bmm(Ls, Z)                                  # (n_series, H, n_paths)
    U = 0.5 * (1.0 + torch.erf(Y / sqrt2))
    U = U.clamp(q_lo, q_hi)

    # Single batched interpolation: reshape to (n_series*H, n_paths), interpolate,
    # then reshape back to (n_series, n_paths, H).
    U_flat = U.reshape(n_series * H, n_paths)
    qv_flat = quantile_values.reshape(n_series * H, Q)
    samples_flat = interp_2d(U_flat, quantile_positions, qv_flat)  # (n_series*H, n_paths)
    return samples_flat.view(n_series, H, n_paths).permute(0, 2, 1).contiguous()


def schaake_shuffle_sample(
    quantile_positions,
    quantile_values,
    y_hist,
    n_paths,
    seed=None,
):
    """Independent marginal samples reordered by historical rank templates.

    Draws independent uniform samples per horizon step, maps them through
    each step's marginal CDF (via quantile interpolation), then reorders
    them to match the rank structure of historical trajectory templates.
    Ranks are compared across paths within each horizon step, so the reordering
    leaves every step's marginal distribution untouched and only induces the
    cross-step dependence observed in the history.

    Templates are the length-`H` rolling windows of each series' history. They
    are drawn without replacement when at least `n_paths` windows exist, and
    with replacement otherwise (a short history therefore yields duplicated,
    less diverse templates).

    The uniform draw and CDF inversion are batched across all series. The
    template extraction loop remains per-series because each series may have
    a different NaN-filtered history length.

    Args:
        quantile_positions: (Q,) sorted quantile levels in (0, 1), tensor.
        quantile_values: (n_series, H, Q) quantile forecast values, tensor.
        y_hist: list of 1-D tensors, historical values per series. NaNs are
            dropped per series before templates are extracted.
        n_paths: number of sample paths.
        seed: random seed.

    Returns:
        simulations: (n_series, n_paths, H) tensor, dtype of `quantile_values`.

    Raises:
        ValueError: if a series has fewer than `H` non-NaN historical values.
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    n_series, H, Q = quantile_values.shape
    dtype = quantile_values.dtype

    q_lo = quantile_positions[0]
    q_hi = quantile_positions[-1]

    # Draw all uniform samples at once for all series: (n_series, H, n_paths)
    U_all = torch.empty(n_series, H, n_paths, dtype=dtype)
    U_all.uniform_(q_lo.item(), q_hi.item(), generator=gen)

    # Batched CDF inversion: reshape to (n_series*H, n_paths), interpolate,
    # then restore to (n_series, H, n_paths).
    U_flat = U_all.view(n_series * H, n_paths)
    qv_flat = quantile_values.reshape(n_series * H, Q)
    raw_all = interp_2d(U_flat, quantile_positions, qv_flat).view(
        n_series, H, n_paths
    )  # (n_series, H, n_paths)

    simulations = torch.empty((n_series, n_paths, H), dtype=dtype)

    for i in range(n_series):
        raw_samples = raw_all[i]  # (H, n_paths)

        # Extract historical trajectory templates
        yi = y_hist[i]
        yi_clean = yi[~torch.isnan(yi)]
        T = len(yi_clean)
        if T < H:
            raise ValueError(
                f"Series {i}: history length {T} is shorter than horizon {H}. "
                "Cannot extract trajectory templates for Schaake shuffle."
            )

        # Select n_paths starting indices for templates
        max_start = T - H + 1
        if max_start < n_paths:
            start_indices = torch.randint(
                0, max_start, (n_paths,), generator=gen
            )
        else:
            perm = torch.randperm(max_start, generator=gen)
            start_indices = perm[:n_paths]

        # Build template matrix using unfold: (max_start, H) → (H, n_paths)
        all_windows = yi_clean.unfold(0, H, 1)  # (max_start, H)
        templates = all_windows[start_indices].T  # (H, n_paths)

        # Reorder raw samples to match the rank structure of each template.
        # Ranks are taken across paths within each horizon step (dim=1), so
        # each row of template_ranks is a permutation of 0..n_paths-1 (ties
        # broken by index). Gather, not scatter: the sample at rank r must land
        # on the path whose template rank is r, i.e.
        #     reordered[h, p] = sorted_samples[h, template_ranks[h, p]]
        # Scattering instead applies the inverse permutation, which preserves
        # each step's marginal but destroys the cross-step dependence.
        template_ranks = torch.argsort(
            torch.argsort(templates, dim=1), dim=1
        )  # (H, n_paths)
        sorted_samples = torch.sort(raw_samples, dim=1).values  # (H, n_paths)
        reordered = torch.gather(sorted_samples, dim=1, index=template_ranks)

        simulations[i] = reordered.T  # (n_paths, H)

    return simulations


def extract_y_hist(dataset):
    """Extract per-series historical y tensors from a dataset.

    Args:
        dataset: NeuralForecast TimeSeriesDataset with ``y_idx``,
            ``temporal``, ``n_groups``, and ``indptr`` attributes.

    Returns:
        list of 1-D tensors, one per series.
    """
    y_idx = dataset.y_idx
    temporal = dataset.temporal[:, y_idx]
    y_hist = []
    for i in range(dataset.n_groups):
        start = dataset.indptr[i]
        end = dataset.indptr[i + 1]
        y_hist.append(temporal[start:end])
    return y_hist


def sample_from_quantiles(
    quantile_positions,
    quantile_values,
    dataset,
    n_paths,
    seed=None,
    method="gaussian_copula",
):
    """Dispatch to a simulation sampler given quantile forecasts.

    Extracts historical data from *dataset*, converts inputs to tensors,
    and calls the appropriate sampling function.

    Args:
        quantile_positions: array-like (Q,) sorted quantile levels in (0, 1).
        quantile_values: array-like (n_series, H, Q) quantile forecast values.
        dataset: NeuralForecast TimeSeriesDataset.
        n_paths: number of sample paths.
        seed: random seed.
        method: ``"gaussian_copula"`` or ``"schaake_shuffle"``.

    Returns:
        samples (np.ndarray): Array of shape ``[n_series, n_paths, H]``.
    """
    if method not in VALID_SIMULATION_METHODS:
        raise ValueError(
            f"Unknown simulation method '{method}'. "
            f"Valid methods: {sorted(VALID_SIMULATION_METHODS)}"
        )

    y_hist = extract_y_hist(dataset)

    quantile_positions_t = torch.as_tensor(
        np.array(quantile_positions), dtype=torch.float64
    )
    quantile_values_t = torch.as_tensor(quantile_values, dtype=torch.float64)

    sampler = {
        "gaussian_copula": gaussian_copula_sample,
        "schaake_shuffle": schaake_shuffle_sample,
    }[method]

    samples = sampler(
        quantile_positions=quantile_positions_t,
        quantile_values=quantile_values_t,
        y_hist=y_hist,
        n_paths=n_paths,
        seed=seed,
    )

    if isinstance(samples, torch.Tensor):
        if samples.dtype == torch.bfloat16:
            return samples.float().numpy()
        return samples.numpy()
    return samples



def _validate_ensembling_stride(stride: int, h: int) -> None:
    """Reject strides for which ensembling is undefined or pointless.

    Args:
        stride (int): Step size between consecutive forecast windows.
        h (int): Forecast horizon.

    Raises:
        ValueError: If `stride` is not positive, or if `stride >= h`.
    """
    if stride < 1:
        raise ValueError(f"stride={stride} must be a positive integer.")
    if stride == h:
        raise ValueError(
            f"stride={stride} equals h={h}: windows are non-overlapping so each "
            "target date has exactly one forecast and ensembling has no effect. "
            "Use method='identity' instead."
        )
    if stride > h:
        raise ValueError(
            f"stride={stride} > h={h}: some target dates have no forecast coverage. "
            "Please review the step size used to generate the forecast windows."
        )


def _gather_windows_from_dates(
    x: np.ndarray, stride: int, n_windows: int
) -> np.ndarray:
    """Invert `_reshape_windows_by_date`, going from target dates back to windows.

    Args:
        x (np.ndarray): Array of shape `[B, S, H, C]` indexed by target date.
        stride (int): Step size between consecutive forecast windows.
        n_windows (int): Number of forecast windows `T` to recover.

    Returns:
        np.ndarray: Array of shape `[B, T, H, C]`.
    """
    horizon = x.shape[2]
    t_grid, h_grid = np.meshgrid(
        np.arange(n_windows), np.arange(horizon), indexing="ij"
    )
    d_grid = t_grid * stride + h_grid  # the same index grid used to scatter
    return x[:, d_grid, h_grid, :]


def _causal_mean(x: np.ndarray, window_size: Optional[int]) -> np.ndarray:
    """Mean over each position and the positions before it, ignoring NaNs.

    Args:
        x (np.ndarray): Array of shape `[N, H, C]`, pooled along axis 1.
        window_size (int, optional): If given, pool at most this many positions,
            ending at the current one. Defaults to every earlier position.

    Returns:
        np.ndarray: Array of shape `[N, H, C]`, `np.nan` where the pool is empty.
    """
    n_steps = x.shape[1]
    if window_size is None:
        total = np.nancumsum(x, axis=1)
        counts = np.cumsum(~np.isnan(x), axis=1)
    else:
        total = np.stack(
            [
                np.nansum(x[:, max(0, i - window_size + 1) : i + 1, :], axis=1)
                for i in range(n_steps)
            ],
            axis=1,
        )
        counts = np.stack(
            [
                np.sum(~np.isnan(x[:, max(0, i - window_size + 1) : i + 1, :]), axis=1)
                for i in range(n_steps)
            ],
            axis=1,
        )
    with np.errstate(invalid="ignore"):
        out = total / counts
    out[counts == 0] = np.nan
    return out


def _causal_median(x: np.ndarray, window_size: Optional[int]) -> np.ndarray:
    """Median over each position and the positions before it, ignoring NaNs.

    Args:
        x (np.ndarray): Array of shape `[N, H, C]`, pooled along axis 1.
        window_size (int, optional): If given, pool at most this many positions,
            ending at the current one. Defaults to every earlier position.

    Returns:
        np.ndarray: Array of shape `[N, H, C]`, `np.nan` where the pool is empty.
    """
    n_steps = x.shape[1]
    with warnings.catch_warnings():
        # a pool can be entirely NaN where no forecast reaches that date
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.stack(
            [
                np.nanmedian(
                    x[:, max(0, i - window_size + 1) if window_size else 0 : i + 1, :],
                    axis=1,
                )
                for i in range(n_steps)
            ],
            axis=1,
        )


def _causal_ewm(
    x: np.ndarray, window_size: Optional[int], decay: float
) -> np.ndarray:
    """Exponentially weighted mean, weighting the current position most heavily.

    A position `k` steps before the current one gets weight `decay ** k`. The weights
    are renormalised over the non-NaN positions.

    Args:
        x (np.ndarray): Array of shape `[N, H, C]`, pooled along axis 1.
        window_size (int, optional): If given, pool at most this many positions,
            ending at the current one. Defaults to every earlier position.
        decay (float): Weight ratio between a position and the one after it, in
            `[0, 1)`.

    Returns:
        np.ndarray: Array of shape `[N, H, C]`, `np.nan` where the pool is empty.
    """
    n_steps = x.shape[1]
    span = window_size if window_size is not None else n_steps

    positions = np.arange(n_steps)
    lag = positions[:, None] - positions[None, :]  # [current, pooled]
    weights = np.where((lag >= 0) & (lag < span), decay ** np.abs(lag), 0.0)

    valid = ~np.isnan(x)
    clean = np.nan_to_num(x)
    weighted = np.einsum("hi,bic->bhc", weights, valid * clean)
    normaliser = np.einsum("hi,bic->bhc", weights, valid.astype(float))
    with np.errstate(invalid="ignore"):
        return np.where(normaliser > 0, weighted / normaliser, np.nan)


def ensemble_forecast_windows(
    y_hat: np.ndarray,
    stride: int = 1,
    method: str = "mean",
    window_size: Optional[int] = None,
    alpha: float = 0.5,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""Causal Forecast Ensembling

    A multi-horizon system forecasts each target date several times, once from every
    forecast creation date (FCD) whose horizon reaches it. Forecast ensembling
    combines the overlapping forecasts that target the same date.

    The combination is **causal**. Indexing by series $b$, FCD $t$ and horizon step
    $h$, with windows `stride` steps apart, the forecasts of the same target date
    issued $k$ FCDs earlier are $\hat{y}_{b,t-k,h+k \cdot \mathrm{stride}}$. The
    ensembled forecast pools only these, never a forecast from a later FCD, so it can
    be produced in real time:

    ```math
    \tilde{y}_{b,t,h} = \frac{\sum_{k=0}^{K} w_k \, \hat{y}_{b,t-k,h+k \cdot \mathrm{stride}}}{\sum_{k=0}^{K} w_k}
    ```

    where $K$ is the number of earlier forecasts available, capped by the start of the
    sample, the horizon, and `window_size`. `'mean'` uses $w_k = 1$; `'ewm'` uses

    ```math
    w_k = (1-\alpha)^{k}
    ```

    the exponential smoothing weights, as in `pandas.DataFrame.ewm(alpha=alpha)`. As
    $\alpha \to 0$ this approaches the plain mean, and $\alpha = 1$ keeps only the
    latest forecast. `'median'` takes the median of the same pool.

    An FCD near the start of the sample has few earlier forecasts to draw on and is
    therefore barely changed; later FCDs pool more and are smoothed more.

    Args:
        y_hat (np.ndarray): Forecasts of shape `[B, T, H, C]` or `[B, T, H, C, Q]`, for
            `B` series, `T` forecast windows, horizon `H`, `C` target channels and `Q`
            quantiles. Quantiles are ensembled independently.
        stride (int): Step size between consecutive forecast windows. Must be smaller
            than `H`, otherwise windows do not overlap and there is nothing to
            ensemble. Defaults to 1.
        method (str): One of `'mean'`, `'median'`, `'ewm'` or `'identity'`. `'mean'` and
            `'median'` weight every available FCD equally; `'ewm'` weights the most
            recent FCD most heavily, which suits the fact that older forecasts of a
            date carry more uncertainty. `'identity'` returns the input unchanged, which
            is useful as a baseline. Defaults to `'mean'`.
        window_size (int, optional): If given, pool only the `window_size` most recent
            FCDs rather than every earlier one. Must be at least 1. Defaults to None.
        alpha (float): Smoothing factor in `(0, 1]` for `method='ewm'`; larger values
            concentrate weight on the most recent FCD. Defaults to 0.5.
        mask (np.ndarray, optional): Array of shape `[B, T, H, C]`, 1 for a real
            timestep and 0 for a padded or missing one. Masked positions are excluded
            from the pool. Defaults to None.

    Returns:
        np.ndarray: Ensembled forecasts, the same shape as `y_hat`.

    Raises:
        ValueError: If `method` is unknown, if `y_hat` is not 4- or 5-dimensional, if
            `stride >= H`, if `window_size < 1`, if `alpha` is not in `(0, 1]`, or if
            `mask` does not match the first four dimensions of `y_hat`.

    Example:
        >>> from neuralforecast.losses.numpy import (
        ...     cross_validation_to_windows,
        ...     excess_volatility,
        ... )
        >>> from neuralforecast.utils import ensemble_forecast_windows
        >>> w = cross_validation_to_windows(cv_df, model='NHITS')  # doctest: +SKIP
        >>> ensembled = ensemble_forecast_windows(  # doctest: +SKIP
        ...     w.y_hat, stride=w.stride, method='mean', mask=w.mask,
        ... )
        >>> excess_volatility(  # doctest: +SKIP
        ...     y=w.y, y_hat=ensembled, quantiles=w.quantiles,
        ...     stride=w.stride, mask=w.mask,
        ... )

    References:
        - [Willa Potosnak, Malcolm Wolff, Mengfei Cao, Ruijun Ma, Tatiana
          Konstantinova, Dmitry Efimov, Michael W. Mahoney, Boris Oreshkin, Kin G.
          Olivares, "Forking-Sequences: Statistically and Computationally Efficient
          Multi-Horizon Forecasting with Reduced Volatility". Transactions on
          Machine Learning Research (2026).](https://openreview.net/forum?id=dXdycy7WCX)
    """
    methods = ("mean", "median", "ewm", "identity")
    if method not in methods:
        raise ValueError(f"method must be one of {list(methods)}; got {method!r}.")
    if y_hat.ndim not in (4, 5):
        raise ValueError(
            f"y_hat must have shape [B, T, H, C] or [B, T, H, C, Q]; got "
            f"{y_hat.ndim} dimensions."
        )

    # identity is a no-op, so it stays valid even where ensembling is not, which is
    # what the stride error below tells callers to fall back to.
    if method == "identity":
        return y_hat

    if window_size is not None and window_size < 1:
        raise ValueError(f"window_size={window_size} must be a positive integer.")
    if method == "ewm" and not 0 < alpha <= 1:
        raise ValueError(f"alpha={alpha} must be in (0, 1].")

    had_quantile_axis = y_hat.ndim == 5
    if not had_quantile_axis:
        y_hat = y_hat[..., np.newaxis]
    if mask is not None and mask.shape != y_hat.shape[:4]:
        raise ValueError(
            f"mask.shape={mask.shape} must match y_hat's first four dimensions "
            f"{y_hat.shape[:4]}."
        )

    n_series, n_windows, horizon, n_channels, n_quantiles = y_hat.shape
    _validate_ensembling_stride(stride, horizon)

    # Fold the quantile axis into the channel axis so one pass handles every quantile.
    folded = y_hat.reshape(n_series, n_windows, horizon, n_channels * n_quantiles)
    folded_mask = np.repeat(mask, n_quantiles, axis=-1) if mask is not None else None

    by_date = _reshape_windows_by_date(folded, stride=stride, mask=folded_mask)
    n_dates = by_date.shape[1]
    flat = by_date.reshape(n_series * n_dates, horizon, n_channels * n_quantiles)

    # Each row holds one target date's forecasts, oldest FCD in the last column. The
    # _causal_* helpers pool each position with the positions to its left, so flip
    # the row to put older FCDs on the left: each FCD is then pooled only with FCDs
    # issued before it.
    flat = flat[:, ::-1, :]
    # Consecutive FCDs sit `stride` horizon steps apart, so a window of FCDs and a
    # per-FCD decay are converted to horizon steps before pooling.
    span = None if window_size is None else (window_size - 1) * stride + 1
    if method == "mean":
        pooled = _causal_mean(flat, span)
    elif method == "median":
        pooled = _causal_median(flat, span)
    else:
        pooled = _causal_ewm(flat, span, decay=(1 - alpha) ** (1 / stride))
    pooled = pooled[:, ::-1, :]

    pooled = pooled.reshape(n_series, n_dates, horizon, n_channels * n_quantiles)
    out = _gather_windows_from_dates(pooled, stride=stride, n_windows=n_windows)
    out = out.reshape(n_series, n_windows, horizon, n_channels, n_quantiles)
    return out if had_quantile_axis else out[..., 0]
