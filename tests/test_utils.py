import numpy as np
import pandas as pd
import pytest

from neuralforecast.losses.numpy import excess_volatility, forecast_percentage_change
from neuralforecast.utils import (
    PredictionIntervals,
    ensemble_forecast_windows,
    level_to_quantiles,
    quantiles_to_level,
)


# Test level_to_quantiles
def test_level_to_quantiles():
    level_base = [80, 90]
    quantiles_base = [0.05, 0.1, 0.9, 0.95]
    quantiles = level_to_quantiles(level_base)
    level = quantiles_to_level(quantiles_base)

    assert quantiles == quantiles_base
    assert level == level_base


@pytest.mark.parametrize("step_size", [0, -1])
def test_prediction_intervals_step_size_validation(step_size):
    with pytest.raises(ValueError, match="step_size must be at least 1"):
        PredictionIntervals(step_size=step_size)


# ---------------------------------------------------------------------------
# ensemble_forecast_windows
# ---------------------------------------------------------------------------


def _tagged(n_windows=5, horizon=5):
    """Every prediction carries its own (FCD, lead) as `10 * t + h`, in [1, T, H, 1, 1].

    The forecasts of a shared target date are then easy to identify by eye and the
    pool can be checked exactly.
    """
    t, h = np.meshgrid(np.arange(n_windows), np.arange(horizon), indexing="ij")
    return (10.0 * t + h)[None, :, :, None, None]


def _reference_pool(preds, t, h, stride, window_size):
    """The forecasts of the date that window `t`, lead `h` targets, newest first.

    Only window `t` itself and earlier windows are allowed, which is what makes the
    ensemble causal.
    """
    horizon = preds.shape[2]
    pool = []
    for k in range(t + 1):  # k FCDs back
        lead = h + k * stride
        if lead >= horizon or (window_size is not None and k >= window_size):
            break
        pool.append(preds[0, t - k, lead, 0, 0])
    return np.array(pool)


def _reference(pool, method, alpha):
    if method == "mean":
        return pool.mean()
    if method == "median":
        return np.median(pool)
    weights = (1 - alpha) ** np.arange(len(pool))
    return (weights * pool).sum() / weights.sum()


# ---------------------------------------------------------------------------
# Causality: an FCD may only pool forecasts it could already have seen
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["mean", "median", "ewm"])
@pytest.mark.parametrize("stride, window_size", [(1, None), (1, 2), (2, None), (2, 2)])
def test_each_window_pools_exactly_the_current_and_earlier_forecasts(
    method, stride, window_size
):
    preds = _tagged(6, 6)
    out = ensemble_forecast_windows(
        preds, stride=stride, method=method, window_size=window_size, alpha=0.3
    )

    for t in range(preds.shape[1]):
        for h in range(preds.shape[2]):
            pool = _reference_pool(preds, t, h, stride, window_size)
            assert out[0, t, h, 0, 0] == pytest.approx(
                _reference(pool, method, alpha=0.3)
            ), f"window {t}, lead {h} pooled the wrong set"


@pytest.mark.parametrize("method", ["mean", "median", "ewm"])
def test_ensemble_never_uses_a_later_window(method):
    """Changing only the last window must leave every earlier window untouched."""
    preds = _tagged()
    perturbed = preds.copy()
    perturbed[0, -1] += 1000.0

    before = ensemble_forecast_windows(preds, stride=1, method=method)
    after = ensemble_forecast_windows(perturbed, stride=1, method=method)
    np.testing.assert_allclose(before[0, :-1], after[0, :-1])


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.9, 1.0])
def test_ewm_matches_pandas(alpha):
    pool = [4.0, 13.0, 22.0, 31.0, 40.0]  # date 4, oldest first; 40 is the newest
    out = ensemble_forecast_windows(_tagged(), stride=1, method="ewm", alpha=alpha)
    expected = pd.Series(pool).ewm(alpha=alpha).mean().iloc[-1]
    assert out[0, 4, 0, 0, 0] == pytest.approx(expected)


def test_identity_returns_the_input_even_where_ensembling_is_not():
    """stride == H is the case the stride error tells callers to use identity for."""
    preds = _tagged()
    out = ensemble_forecast_windows(preds, stride=5, method="identity")
    np.testing.assert_allclose(out, preds)


@pytest.mark.parametrize("method", ["mean", "median", "ewm"])
def test_quantiles_are_ensembled_independently(method):
    point = _tagged()[..., 0]  # [B, T, H, C]
    quantiles = np.concatenate([point[..., None], 2 * point[..., None]], axis=-1)

    single = ensemble_forecast_windows(point, stride=1, method=method)
    both = ensemble_forecast_windows(quantiles, stride=1, method=method)

    assert single.shape == point.shape
    np.testing.assert_allclose(both[..., 0], single)
    np.testing.assert_allclose(both[..., 1], 2 * single)


def test_mask_excludes_positions_from_the_pool():
    mask = np.ones((1, 5, 5, 1))
    mask[0, 0, 4, 0] = 0  # drop the oldest forecast of date 4

    out = ensemble_forecast_windows(_tagged(), stride=1, method="mean", mask=mask)
    assert out[0, 4, 0, 0, 0] == pytest.approx(np.mean([13, 22, 31, 40]))


@pytest.mark.parametrize(
    "y_hat, kwargs, match",
    [
        (_tagged(), {"stride": 5}, "non-overlapping"),
        (_tagged(), {"stride": 6}, "no forecast coverage"),
        (_tagged(), {"stride": 0}, "stride=0 must be a positive integer"),
        (_tagged(), {"method": "average"}, "method must be one of"),
        (np.ones((2, 2, 2)), {}, r"\[B, T, H, C\]"),
        (_tagged(), {"window_size": 0}, "window_size=0 must be a positive integer"),
        (_tagged(), {"method": "ewm", "alpha": 0.0}, r"must be in \(0, 1\]"),
        (_tagged(), {"method": "ewm", "alpha": 1.5}, r"must be in \(0, 1\]"),
    ],
)
def test_invalid_inputs_raise(y_hat, kwargs, match):
    with pytest.raises(ValueError, match=match):
        ensemble_forecast_windows(y_hat, **kwargs)


# ---------------------------------------------------------------------------
# The point of the exercise: ensembling lowers volatility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["mean", "median", "ewm"])
def test_ensembling_reduces_volatility(method):
    """A forecaster that revises heavily between FCDs is steadier once ensembled."""
    rng = np.random.default_rng(0)
    n_windows, horizon = 8, 6
    truth = np.sin(np.arange(40) / 3.0) * 10 + 50
    y = np.stack([truth[s : s + horizon] for s in range(n_windows)])[None, :, :, None]
    medians = np.stack(
        [truth[s : s + horizon] + rng.normal(0, 3.0, horizon) for s in range(n_windows)]
    )
    y_hat = np.stack([medians - 2.0, medians, medians + 2.0], axis=-1)[None, :, :, None]
    quantiles = [0.1, 0.5, 0.9]

    ensembled = ensemble_forecast_windows(y_hat, stride=1, method=method)

    before = forecast_percentage_change(y_hat=y_hat[..., 1], stride=1)
    after = forecast_percentage_change(y_hat=ensembled[..., 1], stride=1)
    assert after < before, f"{method} did not reduce sFPC"

    sev_before = excess_volatility(y=y, y_hat=y_hat, quantiles=quantiles, stride=1)
    sev_after = excess_volatility(y=y, y_hat=ensembled, quantiles=quantiles, stride=1)
    assert sev_after < sev_before, f"{method} did not reduce sEV"
