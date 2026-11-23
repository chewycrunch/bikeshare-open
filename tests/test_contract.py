"""Guard rails on the ingest contract and the trained model's behaviour."""
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import r2_score

from src.config import (
    ENCODED_COLS,
    FORBIDDEN,
    LAG_WARMUP,
    RAW,
    RAW_FEATURE_COLS,
    RAW_SHA256,
    REDUNDANT_COLS,
    TARGET,
)
from src.features import build_features, feature_columns, reindex_to_calendar
from src.ingest import load_raw, raw_sha256
from src.train import MODELS, temporal_split


@pytest.fixture(scope="module")
def raw():
    return load_raw().sort_values("dteday").reset_index(drop=True)


@pytest.fixture(scope="module")
def feats(raw):
    return build_features(raw)


def test_raw_data_matches_pinned_hash():
    """Raw data is immutable. Imputing the missing 2011-03-10 and copying the
    result over the raw file passed every other test while silently adding 8
    training rows and moving three of four validation scores
    (docs/SEEDED_FAULTS.md, fault 4). Checking the bytes catches any change,
    by any route."""
    assert raw_sha256() == RAW_SHA256, (
        "data/raw/bike_daily.csv changed; restore it with git checkout -- data/raw/"
    )


def test_calendar_is_complete_before_lagging(raw):
    """The reindexed frame must have no gaps, or the lags are wrong."""
    grid = reindex_to_calendar(raw)
    assert set(grid.dteday.diff().dropna().dt.days.unique()) == {1}


def test_only_expected_rows_are_dropped(raw, feats):
    """Rows may only vanish for the lag warm-up or downstream of a genuinely
    absent day -- never because an encoder produced a silent NaN.

    Accounting: the first LAG_WARMUP days have no complete history. Each
    absent calendar day m costs the LAG_WARMUP rows after it (cnt_lag_1 is
    NaN on m+1, cnt_roll_7 is NaN on m+1..m+7, cnt_lag_7 on m+7; their
    union is 7 rows). Day m itself was never in `raw`, so it is not a loss.
    The bound is exact for the current feature set; add a longer lag and
    LAG_WARMUP in src/config.py must grow with it.
    """
    full = pd.date_range(raw.dteday.min(), raw.dteday.max(), freq="D")
    missing_days = len(full) - len(raw)
    allowed_loss = LAG_WARMUP + LAG_WARMUP * missing_days
    lost = len(raw) - len(feats)
    assert lost <= allowed_loss, f"lost {lost} rows; expected at most {allowed_loss}"


def test_every_raw_column_is_classified(feats):
    """A new column in the raw data must not enter the model silently.

    feature_columns() is a denylist, so a column appended to the CSV, or
    `casual` renamed to something innocent, would become a feature with every
    other test still green. Here every raw column must be named in
    src/config.py, and a raw column may reach the model only through
    RAW_FEATURE_COLS. Reads the header directly so it fails even if
    load_raw() would not.
    """
    raw_cols = set(pd.read_csv(RAW, nrows=0).columns)
    classified = FORBIDDEN | RAW_FEATURE_COLS | ENCODED_COLS | REDUNDANT_COLS | {"dteday"}
    unclassified = raw_cols - classified
    assert not unclassified, (
        f"raw columns not classified in src/config.py: {sorted(unclassified)}"
    )
    admitted = (set(feature_columns(feats)) & raw_cols) - RAW_FEATURE_COLS
    assert not admitted, (
        f"raw columns reached the model without being in RAW_FEATURE_COLS: {sorted(admitted)}"
    )


def test_beats_both_baselines_on_validation(feats):
    """A model that cannot beat 'yesterday again' is not a result. Judged on
    validation: this runs on every `make test`, so scoring the test split
    here would make it a pass/fail gate in the development loop. (The
    starter version asserted test R^2 > 0.4, below persistence's 0.51.)"""
    cols = feature_columns(feats)
    train, valid, _ = temporal_split(feats)
    scores = {}
    for name in ("baseline_mean", "baseline_persistence", "ridge"):
        model = clone(MODELS[name]).fit(train[cols], train[TARGET])
        scores[name] = r2_score(valid[TARGET], model.predict(valid[cols]))
    assert scores["ridge"] > max(scores["baseline_mean"], scores["baseline_persistence"]), scores


def test_persistence_baseline_is_reported():
    """The baseline that matters for a lag-driven model is 'yesterday again'.
    It must be in the comparison, whatever its score."""
    assert "baseline_persistence" in MODELS
