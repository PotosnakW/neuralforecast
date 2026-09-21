import numpy as np
import pytest

from neuralforecast.ensembling import ensemble_forecast_windows
from neuralforecast.losses.numpy import excess_volatility, forecast_percentage_change


T, H, C = 10, 5, 2


def _tagged(n_windows=5, horizon=5):
    """Every prediction carries its own (FCD, lead) as `10 * t + h`.

    With stride 1 the target date is `t + h`, so the forecasts of a shared date are
    easy to identify by eye and the pool can be checked exactly.
    """
    preds = np.zeros((1, n_windows, horizon, 1, 1))
    for t in range(n_windows):
        for h in range(horizon):
            preds[0, t, h, 0, 0] = 10 * t + h
    return preds


def _rolled(quantiles):
    """A self-consistent rolling forecast, broadcast across channels and quantiles."""
    pattern = np.array([1, 2, 3, 4, 5], dtype=float)
    single = np.stack([np.roll(pattern, -t) for t in range(T)])  # [T, H]
    point = np.stack([single, single], axis=-1)[None]  # [1, T, H, C]
    return np.stack([point] * quantiles, axis=-1)  # [1, T, H, C, Q]


# ---------------------------------------------------------------------------
# Causality: an FCD may only pool forecasts it could already have seen
# ---------------------------------------------------------------------------


def test_ensemble_pools_only_the_current_and_earlier_windows():
    """Window `t` must combine only windows `0..t`, never a later one."""
    n_windows = horizon = 5
    out = ensemble_forecast_windows(
        _tagged(n_windows, horizon), stride=1, method="mean"
    )

    for t in range(n_windows):
        for h in range(horizon):
            date = t + h
            # every forecast of this date issued at window t or earlier
            available = [
                10 * past + (date - past)
                for past in range(t + 1)
                if 0 <= date - past < horizon
            ]
            assert out[0, t, h, 0, 0] == pytest.approx(
                np.mean(available)
            ), f"window {t}, lead {h} pooled the wrong set"


def test_oldest_window_is_unchanged_and_newest_pools_everything():
    n_windows = horizon = 5
    tagged = _tagged(n_windows, horizon)
    out = ensemble_forecast_windows(tagged, stride=1, method="mean")

    # the first window has no earlier forecast to draw on
    np.testing.assert_allclose(out[0, 0], tagged[0, 0])

    # target date 4 seen by all five windows; the last one pools all of them
    assert out[0, 4, 0, 0, 0] == pytest.approx(np.mean([4, 13, 22, 31, 40]))


def test_ensemble_never_uses_a_later_window():
    """Changing only the last window must leave every earlier window untouched."""
    tagged = _tagged()
    baseline = ensemble_forecast_windows(tagged, stride=1, method="mean")

    perturbed = tagged.copy()
    perturbed[0, -1] += 1000.0
    after = ensemble_forecast_windows(perturbed, stride=1, method="mean")

    np.testing.assert_allclose(baseline[0, :-1], after[0, :-1])


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quantiles", [1, 3])
@pytest.mark.parametrize("stride", [1, 2])
def test_identity_returns_the_input_unchanged(quantiles, stride):
    preds = _rolled(quantiles)
    out = ensemble_forecast_windows(preds, stride=stride, method="identity")
    assert out.shape == (1, T, H, C, quantiles)
    np.testing.assert_allclose(out, preds)


def test_identity_is_allowed_where_ensembling_is_not():
    """stride == H is the case the stride error tells callers to use identity for."""
    preds = _rolled(1)
    np.testing.assert_allclose(
        ensemble_forecast_windows(preds, stride=H, method="identity"), preds
    )


@pytest.mark.parametrize("method", ["mean", "median", "ewm"])
def test_quantiles_are_ensembled_independently(method):
    single = ensemble_forecast_windows(_rolled(1), stride=1, method=method)
    triple = ensemble_forecast_windows(_rolled(3), stride=1, method=method)
    assert triple.shape == (1, T, H, C, 3)
    for q in range(3):
        np.testing.assert_allclose(triple[..., q], single[..., 0])


def test_median_pools_the_same_set_as_mean():
    out = ensemble_forecast_windows(_tagged(), stride=1, method="median")
    # target date 4 at the last window pools all five forecasts
    assert out[0, 4, 0, 0, 0] == pytest.approx(np.median([4, 13, 22, 31, 40]))


def test_ewm_weights_the_newest_forecast_most():
    """The exponentially weighted mean must sit nearer the newest forecast than the
    equally weighted mean does."""
    tagged = _tagged()
    pool = [4, 13, 22, 31, 40]  # date 4, oldest first; 40 is the newest
    mean = ensemble_forecast_windows(tagged, stride=1, method="mean")[0, 4, 0, 0, 0]
    ewm = ensemble_forecast_windows(tagged, stride=1, method="ewm", alpha=0.8)[
        0, 4, 0, 0, 0
    ]

    assert mean == pytest.approx(np.mean(pool))
    assert ewm > mean  # pulled toward the newest value
    assert ewm < pool[-1]


def test_window_size_limits_how_far_back_the_pool_reaches():
    out = ensemble_forecast_windows(_tagged(), stride=1, method="mean", window_size=2)
    # date 4 at the last window now sees only the two most recent forecasts
    assert out[0, 4, 0, 0, 0] == pytest.approx(np.mean([31, 40]))


def test_larger_stride_pools_fewer_windows():
    out = ensemble_forecast_windows(_tagged(6, 6), stride=2, method="mean")
    # with stride 2, date 4 is reached by window 0 (lead 4), 1 (lead 2) and 2 (lead 0)
    assert out[0, 2, 0, 0, 0] == pytest.approx(np.mean([4, 12, 20]))


# ---------------------------------------------------------------------------
# Shapes, masking and validation
# ---------------------------------------------------------------------------


def test_point_forecasts_keep_their_shape():
    point = _rolled(1)[..., 0]  # [B, T, H, C]
    out = ensemble_forecast_windows(point, stride=1, method="mean")
    assert out.shape == point.shape


def test_mask_excludes_positions_from_the_pool():
    tagged = _tagged()
    mask = np.ones((1, 5, 5, 1))
    mask[0, 0, 4, 0] = 0  # drop the oldest forecast of date 4

    out = ensemble_forecast_windows(tagged, stride=1, method="mean", mask=mask)
    assert out[0, 4, 0, 0, 0] == pytest.approx(np.mean([13, 22, 31, 40]))


@pytest.mark.parametrize(
    "stride, match",
    [(H, "non-overlapping"), (H + 1, "no forecast coverage"), (0, "positive integer")],
)
def test_invalid_strides_raise(stride, match):
    with pytest.raises(ValueError, match=match):
        ensemble_forecast_windows(_rolled(1), stride=stride, method="mean")


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="method must be one of"):
        ensemble_forecast_windows(_rolled(1), stride=1, method="average")


def test_wrong_dimensionality_raises():
    with pytest.raises(ValueError, match=r"\[B, T, H, C\]"):
        ensemble_forecast_windows(np.ones((2, 2, 2)), stride=1, method="mean")


# ---------------------------------------------------------------------------
# The point of the exercise: ensembling lowers volatility
# ---------------------------------------------------------------------------


def _erratic_forecast(rng):
    """A forecaster that revises heavily between FCDs, in [B, T, H, C, Q] form."""
    n_windows, horizon = 8, 6
    truth = np.sin(np.arange(40) / 3.0) * 10 + 50
    y = np.stack([truth[s : s + horizon] for s in range(n_windows)])[None, :, :, None]
    medians = np.stack(
        [truth[s : s + horizon] + rng.normal(0, 3.0, horizon) for s in range(n_windows)]
    )
    y_hat = np.stack([medians - 2.0, medians, medians + 2.0], axis=-1)[None, :, :, None]
    return y, y_hat


@pytest.mark.parametrize("method", ["mean", "median", "ewm"])
def test_ensembling_reduces_volatility(method):
    rng = np.random.default_rng(0)
    y, y_hat = _erratic_forecast(rng)
    quantiles = [0.1, 0.5, 0.9]

    ensembled = ensemble_forecast_windows(y_hat, stride=1, method=method)

    before = forecast_percentage_change(y_hat=y_hat[..., 1], stride=1)
    after = forecast_percentage_change(y_hat=ensembled[..., 1], stride=1)
    assert after < before, f"{method} did not reduce sFPC"

    sev_before = excess_volatility(y=y, y_hat=y_hat, quantiles=quantiles, stride=1)
    sev_after = excess_volatility(y=y, y_hat=ensembled, quantiles=quantiles, stride=1)
    assert sev_after < sev_before, f"{method} did not reduce sEV"
