__all__ = ["ensemble_forecast_windows"]


import warnings
from typing import Optional

import numpy as np

from neuralforecast.losses.numpy import _reshape_windows_by_date


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


def _trailing_mean(x: np.ndarray, window_size: Optional[int]) -> np.ndarray:
    """Mean over each position and the positions before it, ignoring NaNs."""
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


def _trailing_median(x: np.ndarray, window_size: Optional[int]) -> np.ndarray:
    """Median over each position and the positions before it, ignoring NaNs."""
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


def _trailing_ewm(
    x: np.ndarray, window_size: Optional[int], alpha: float
) -> np.ndarray:
    """Exponentially weighted mean, weighting the current position most heavily."""
    n_steps = x.shape[1]
    beta = np.log(alpha / (1 - alpha))
    span = window_size if window_size is not None else n_steps

    positions = np.arange(n_steps)
    starts = np.maximum(0, positions - span + 1)
    in_window = (positions[None, :] >= starts[:, None]) & (
        positions[None, :] <= positions[:, None]
    )
    weights = np.where(
        in_window, np.exp(beta * (positions[None, :] - starts[:, None])), 0.0
    )

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
    r"""Ensemble the overlapping forecasts that target the same date.

    A multi-horizon system run on a schedule forecasts each target date several times,
    once from every forecast creation date (FCD) whose horizon reaches it. Averaging
    those repeated forecasts reduces the variance of the estimate, and so reduces
    forecast volatility: successive FCDs move the forecast around less.

    The combination is **causal**. The ensembled forecast issued at FCD $t$ for a given
    target date pools only the forecasts of that date available at FCD $t$, namely the
    one issued at $t$ itself and those issued at earlier FCDs. Forecasts from later
    FCDs are never used, so the output can be produced in real time. Concretely, the
    prediction at horizon step $h$ is combined with the predictions at horizon steps
    $h+1, h+2, \dots$ of the same target date, which are the ones made earlier.

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
            FCDs rather than every earlier one. Defaults to None.
        alpha (float): Smoothing factor in `(0, 1)` for `method='ewm'`; larger values
            concentrate weight on the most recent FCD. Defaults to 0.5.
        mask (np.ndarray, optional): Array of shape `[B, T, H, C]`, 1 for a real
            timestep and 0 for a padded or missing one. Masked positions are excluded
            from the pool. Defaults to None.

    Returns:
        np.ndarray: Ensembled forecasts, the same shape as `y_hat`.

    Raises:
        ValueError: If `method` is unknown, if `stride >= H`, or if `y_hat` is not 4- or
            5-dimensional.

    Example:
        >>> from neuralforecast.losses.numpy import (
        ...     cross_validation_to_windows,
        ...     excess_volatility,
        ... )
        >>> from neuralforecast.ensembling import ensemble_forecast_windows
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

    had_quantile_axis = y_hat.ndim == 5
    if not had_quantile_axis:
        y_hat = y_hat[..., np.newaxis]

    n_series, n_windows, horizon, n_channels, n_quantiles = y_hat.shape
    _validate_ensembling_stride(stride, horizon)

    # Fold the quantile axis into the channel axis so one pass handles every quantile.
    folded = y_hat.reshape(n_series, n_windows, horizon, n_channels * n_quantiles)
    folded_mask = np.repeat(mask, n_quantiles, axis=-1) if mask is not None else None

    by_date = _reshape_windows_by_date(folded, stride=stride, mask=folded_mask)
    n_dates = by_date.shape[1]
    flat = by_date.reshape(n_series * n_dates, horizon, n_channels * n_quantiles)

    # Larger horizon steps hold older forecasts, so reversing makes "everything before
    # this position" mean "this FCD and the ones before it", which is the causal pool.
    flat = flat[:, ::-1, :]
    if method == "mean":
        pooled = _trailing_mean(flat, window_size)
    elif method == "median":
        pooled = _trailing_median(flat, window_size)
    else:
        pooled = _trailing_ewm(flat, window_size, alpha)
    pooled = pooled[:, ::-1, :]

    pooled = pooled.reshape(n_series, n_dates, horizon, n_channels * n_quantiles)
    out = _gather_windows_from_dates(pooled, stride=stride, n_windows=n_windows)
    out = out.reshape(n_series, n_windows, horizon, n_channels, n_quantiles)
    return out if had_quantile_axis else out[..., 0]
