"""Stage 3: temporal split, fit, evaluate, write metrics.

Two baselines and two models, on purpose.

  baseline_mean         predict the 2011 average for every day
  baseline_persistence  predict yesterday's count (cnt_lag_1)
  ridge                 standardised linear model on the audited features
  hgb                   gradient-boosted trees on the same features

The mean baseline shows what "the distribution moved" costs. Persistence
is the baseline that actually matters: ridge's features are dominated by
lagged demand, so if it cannot beat "yesterday again" it has learned
nothing worth reporting. The booster is there because it fails, and the
reason it fails is the headline result.

Run: python -m src.train

Exercise 4 (never part of `make all`; reports/metrics.json stays untransformed):
  python -m src.train --target-transform ratio --anchor cnt_roll_7
scores ridge and hgb on a target re-expressed relative to a lagged anchor.
Validation only, unless --test is passed, which happens once, at the end.
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import METRICS, PROCESSED, TARGET, TRAIN_END, VALID_END
from src.features import feature_columns


class PersistenceBaseline(BaseEstimator, RegressorMixin):
    """Predict a single lagged column unchanged. Nothing is fit."""

    def __init__(self, column: str = "cnt_lag_1"):
        self.column = column

    def fit(self, X, y=None):
        if self.column not in X.columns:
            raise ValueError(f"persistence baseline needs column {self.column!r}")
        return self

    def predict(self, X):
        return X[self.column].to_numpy(dtype=float)


class AnchoredTarget(BaseEstimator, RegressorMixin):
    """Fit `model` on the target relative to a lagged anchor, then undo it.

      kind="diff":  learn cnt - anchor          predict anchor + f(X)
      kind="ratio": learn log(cnt / anchor)     predict anchor * exp(f(X))

    The anchor is a lag column, known at prediction time, so nothing from
    the future enters. A tree can only output values it saw in training;
    re-expressed this way it predicts a relative change, and the anchor
    carries the level that grew 65% out of the training range.
    """

    def __init__(self, model=None, anchor: str = "cnt_roll_7", kind: str = "ratio"):
        self.model = model
        self.anchor = anchor
        self.kind = kind

    def fit(self, X, y):
        a = X[self.anchor].to_numpy(dtype=float)
        y = np.asarray(y, dtype=float)
        z = y - a if self.kind == "diff" else np.log(y / a)
        self.model_ = clone(self.model).fit(X, z)
        return self

    def predict(self, X):
        a = X[self.anchor].to_numpy(dtype=float)
        z = self.model_.predict(X)
        return a + z if self.kind == "diff" else a * np.exp(z)


MODELS = {
    "baseline_mean": DummyRegressor(strategy="mean"),
    "baseline_persistence": PersistenceBaseline("cnt_lag_1"),
    "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "hgb": HistGradientBoostingRegressor(random_state=0, max_iter=300),
}


def temporal_split(df: pd.DataFrame):
    """Split by date, never at random. A random split on a time series
    puts tomorrow in train and today in test."""
    train = df[df.dteday <= TRAIN_END]
    valid = df[(df.dteday > TRAIN_END) & (df.dteday <= VALID_END)]
    test = df[df.dteday > VALID_END]
    return train, valid, test


def fit_and_score(models: dict = MODELS, splits=("valid", "test")) -> dict:
    df = pd.read_parquet(PROCESSED)
    cols = feature_columns(df)
    train, valid, test = temporal_split(df)

    results = {
        "n_features": len(cols),
        "features": cols,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
    }
    for name, model in models.items():
        model.fit(train[cols], train[TARGET])
        results[name] = {
            split: {
                "r2": round(float(r2_score(part[TARGET], model.predict(part[cols]))), 4),
                "mae": round(float(mean_absolute_error(part[TARGET], model.predict(part[cols]))), 1),
            }
            for split, part in [("valid", valid), ("test", test)]
            if split in splits
        }
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-transform", choices=["diff", "ratio"])
    ap.add_argument("--anchor", choices=["cnt_lag_1", "cnt_roll_7"], default="cnt_roll_7")
    ap.add_argument("--test", action="store_true", help="also score test (Exercise 4: once)")
    ns = ap.parse_args()
    METRICS.parent.mkdir(parents=True, exist_ok=True)

    if ns.target_transform is None:
        res = fit_and_score()
        out = METRICS
    else:
        tag = f"{ns.target_transform}_{ns.anchor}"
        models = {f"{name}_{tag}": AnchoredTarget(MODELS[name], ns.anchor, ns.target_transform)
                  for name in ("ridge", "hgb")}
        res = fit_and_score(models, ("valid", "test") if ns.test else ("valid",))
        out = METRICS.parent / f"target_{tag}{'_test' if ns.test else ''}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "features"}, indent=2))
