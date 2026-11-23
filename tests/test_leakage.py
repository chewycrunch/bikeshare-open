"""The tests that make the agent honest.

test_no_forbidden_features is the single most valuable test in this repo:
casual + registered == cnt exactly, so any model handed those columns
scores R^2 = 1.0 and looks brilliant while having learned nothing.
"""
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from src.config import FORBIDDEN, TARGET
from src.features import WEATHER_ORDER, build_features, feature_columns
from src.ingest import load_raw


@pytest.fixture(scope="module")
def feats():
    return build_features(load_raw())


def test_leakage_identity_holds(feats):
    """Documents *why* casual and registered are forbidden."""
    assert (feats.casual + feats.registered == feats[TARGET]).all()


def test_no_forbidden_features(feats):
    """Guards the gate, not the data: feature_columns() excludes FORBIDDEN by
    construction, so this cannot discover a *new* leaky column. What it
    catches is someone editing feature_columns() or FORBIDDEN -- which is
    exactly the fault the Step 7 hook demonstrates."""
    cols = set(feature_columns(feats))
    assert not (cols & FORBIDDEN), f"forbidden columns in feature set: {cols & FORBIDDEN}"


def test_no_feature_restates_a_forbidden_column(feats):
    """Check the values, not the names. No feature, alone or in a pair, may
    be an exact linear function of a forbidden column, so a leak under an
    innocent name (`riders = casual + registered`) goes red even though
    test_no_forbidden_features passes. Every forbidden column is
    integer-valued, so a fit within half a unit on every row recovers it
    exactly by rounding: a restatement, not a forecast."""
    cols = feature_columns(feats)
    ones = np.ones(len(feats))
    numeric = set(feats.select_dtypes(include="number").columns)
    for target in sorted(FORBIDDEN & numeric):
        y = feats[target].to_numpy(dtype=float)
        for subset in [*combinations(cols, 1), *combinations(cols, 2)]:
            X = np.column_stack([feats[list(subset)].to_numpy(dtype=float), ones])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            worst = float(np.abs(X @ coef - y).max())
            assert worst >= 0.5, f"{list(subset)} restates {target!r} (max residual {worst:.2g})"


def test_lags_are_true_calendar_lags(feats):
    """cnt_lag_1 for date d must equal cnt on d-1, in calendar days."""
    by_date = feats.set_index("dteday")[TARGET]
    checked = 0
    for date, lag in feats.set_index("dteday").cnt_lag_1.items():
        prev = date - pd.Timedelta(days=1)
        if prev in by_date.index:
            assert lag == pytest.approx(by_date[prev]), f"bad lag at {date.date()}"
            checked += 1
    assert checked > 600, "too few rows actually verified"


def test_lag_is_not_the_target(feats):
    assert not (feats.cnt_lag_1 == feats[TARGET]).all()


def test_rolling_mean_excludes_today(feats):
    """cnt_roll_7 must be the mean of the 7 calendar days *before* today."""
    by_date = feats.set_index("dteday")[TARGET]
    date = feats.dteday.iloc[200]
    window = [date - pd.Timedelta(days=k) for k in range(1, 8)]
    if all(d in by_date.index for d in window):
        expected = sum(by_date[d] for d in window) / 7
        assert feats.set_index("dteday").cnt_roll_7[date] == pytest.approx(expected)


def test_weather_lag_is_yesterdays_weather(feats):
    """weather_severity_lag_1 on date d is the weather of calendar day d-1.

    Hand-computed from data/raw/bike_daily.csv: 2011-10-29 was BAD (the
    October nor'easter, 627 rides) and 2011-10-30 was GOOD, so the lag on
    2011-10-30 must be WEATHER_ORDER["BAD"] == 2.0 -- not 0.0, today's own
    weather. Then the same check on every row whose previous day exists.
    """
    by_date = feats.set_index("dteday")
    assert by_date.weather_severity_lag_1[pd.Timestamp("2011-10-30")] == 2.0
    assert by_date.weather_severity[pd.Timestamp("2011-10-30")] == 0.0

    raw = load_raw().set_index("dteday").weather.map(WEATHER_ORDER)
    checked = 0
    for date, lag in by_date.weather_severity_lag_1.items():
        prev = date - pd.Timedelta(days=1)
        if prev in raw.index:
            assert lag == raw[prev], f"bad weather lag at {date.date()}"
            checked += 1
    assert checked > 600, "too few rows actually verified"


def test_unmapped_weather_label_fails_loudly():
    """An unseen weather label must raise, not become NaN.

    The real data only holds GOOD/MISTY/BAD, so nothing else exercises the
    guard in encode_weather. Without it, a relabelled BAD ("Bad") maps to
    NaN, build_features' dropna() deletes those days, and every other test
    still passes -- test_no_missing_values runs after the rows are gone.
    """
    raw = load_raw()
    assert (raw.weather == "BAD").any(), "perturbation would be a no-op"
    raw["weather"] = raw.weather.replace({"BAD": "Bad"})
    with pytest.raises(ValueError, match="unmapped weather labels"):
        build_features(raw)


def test_no_missing_values(feats):
    assert feats[feature_columns(feats)].isna().sum().sum() == 0


def test_dates_are_sorted_and_unique(feats):
    assert feats.dteday.is_monotonic_increasing
    assert feats.dteday.is_unique
