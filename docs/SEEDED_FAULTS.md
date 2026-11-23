# Seeded faults (Exercise 2)

**Setup.** The agent was Claude Opus 5.5 in Claude Code, running in this repo
with `.claude/settings.json` loaded, so the starter's PostToolUse hook
(`.claude/hooks/check_leakage.sh`) was live. I seeded each fault with the
agent's `Edit` tool so the hook saw it, committed the broken state, then had
the agent fix it. Everything below is pasted output. The hook stderr is the
text Claude Code handed back to the agent as a blocking error.

**How turns are counted.** One turn is one agent tool call, counted from the
hook's stderr arriving to the suite being green again. The same session
seeded and fixed each fault, so it already knew the cause. These counts are a
lower bound on what a cold agent would need.

---

## Test-first: `weather_severity_lag_1` (Exercise 1)

I wrote the test before the feature (commit `b6b7631`). It failed for the
right reason: the column did not exist yet, so the named-date assertion could
not even be evaluated. `make test`, abridged to the failing frame and
summary:

```
python -m pytest tests/ -q
..........F.....                                                         [100%]
=================================== FAILURES ===================================
____________________ test_weather_lag_is_yesterdays_weather ____________________
...
        by_date = feats.set_index("dteday")
>       assert by_date.weather_severity_lag_1[pd.Timestamp("2011-10-30")] == 2.0
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_leakage.py:69:
...
E       AttributeError: 'DataFrame' object has no attribute 'weather_severity_lag_1'. Did you mean: 'weather_severity'?

.venv/lib/python3.12/site-packages/pandas/core/generic.py:6194: AttributeError
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_weather_lag_is_yesterdays_weather - Attrib...
1 failed, 15 passed in 2.59s
make: *** [test] Error 1
```

After the one-line feature (commit `8653a30`): `16 passed in 2.06s`.

---

## Fault 1: a forbidden column added to `feature_columns`

**Change** (commit `bd6f549`):

```diff
@@ -86,7 +86,7 @@ def build_features(df: pd.DataFrame) -> pd.DataFrame:
 def feature_columns(df: pd.DataFrame) -> list[str]:
     """Numeric columns that are safe to feed the model."""
     numeric = df.select_dtypes(include="number").columns
-    return [c for c in numeric if c not in FORBIDDEN]
+    return [c for c in numeric if c not in FORBIDDEN] + ["casual"]
```

**Hook stderr** (exit 2, delivered to the agent as soon as the Edit landed):

```
Leakage tests failed after editing /Users/chewy/Desktop/bikeshare-open/src/features.py:
feats =         dteday  instant  season  ...  cnt_lag_1 cnt_lag_7   cnt_roll_7
0   2011-01-10     10.0  WINTER  ...      822.0...    1787.0  1530.142857
713 2012-12-31    731.0  WINTER  ...     1796.0     920.0  1531.428571

[714 rows x 29 columns]

    def test_no_forbidden_features(feats):
        """Guards the gate, not the data: feature_columns() excludes FORBIDDEN by
        construction, so this cannot discover a *new* leaky column. What it
        catches is someone editing feature_columns() or FORBIDDEN -- which is
        exactly the fault the Step 7 hook demonstrates."""
        cols = set(feature_columns(feats))
>       assert not (cols & FORBIDDEN), f"forbidden columns in feature set: {cols & FORBIDDEN}"
E       AssertionError: forbidden columns in feature set: {'casual'}
E       assert not ({'atemp', 'casual', 'cnt_lag_1', 'cnt_lag_7', 'cnt_roll_7', 'dayofweek', ...} & {'casual', 'cnt', 'cnt_2d_bfr', 'days_since_2011', 'instant', 'registered', ...})

tests/test_leakage.py:31: AssertionError
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_no_forbidden_features - AssertionError: fo...
1 failed, 7 passed in 0.64s
```

**Red `make test`** (the `E`/summary lines):

```
E       AssertionError: raw columns reached the model without being in RAW_FEATURE_COLS: ['casual']
E       assert not {'casual'}
E       AssertionError: forbidden columns in feature set: {'casual'}
E       assert not ({'atemp', 'casual', 'cnt_lag_1', 'cnt_lag_7', 'cnt_roll_7', 'dayofweek', ...} & {'casual', 'cnt', 'cnt_2d_bfr', 'days_since_2011', 'instant', 'registered', ...})
FAILED tests/test_contract.py::test_every_raw_column_is_classified - Assertio...
FAILED tests/test_leakage.py::test_no_forbidden_features - AssertionError: fo...
2 failed, 14 passed in 2.40s
make: *** [test] Error 1
```

Two independent tests catch it: the starter's gate test and my
raw-column classification test. `test_beats_the_mean_baseline` stays green,
because a leaked model beats the mean easily. A "beats the baseline" test is
no defence against leakage.

**Turns to fix: 1.** One Edit removing `+ ["casual"]`. The hook re-ran
silently (exit 0), and `make test` gave `16 passed in 2.00s`.

---

## Fault 2: `.shift(1)` removed from `cnt_roll_7`

Without the shift, the 7-day rolling mean includes today's `cnt`, so the
feature contains a seventh of the target.

*This fault was run end to end by a fresh headless Claude Code session (Opus
5.5) invoking my skill, `claude -p '/seed-fault fault 2: delete the .shift(1)
from cnt_roll_7 in src/features.py'`. It took 13 turns in total (12 tool
calls), $0.36 and 65 s, and wrote this section itself. The hook fired inside
that agent's loop.*

**Starting commit:** `d6273b2`, with `16 passed in 2.56s`.

**Change** (commit `d5b4a07`):

```diff
@@ -78,7 +78,7 @@ def build_features(df: pd.DataFrame) -> pd.DataFrame:
     # If you add a longer lag, raise LAG_WARMUP in src/config.py.
     out["cnt_lag_1"] = out[TARGET].shift(1)
     out["cnt_lag_7"] = out[TARGET].shift(LAG_WARMUP)
-    out["cnt_roll_7"] = out[TARGET].shift(1).rolling(LAG_WARMUP).mean()
+    out["cnt_roll_7"] = out[TARGET].rolling(LAG_WARMUP).mean()
 
     return out.dropna().reset_index(drop=True)
```

**Hook stderr** (exit 2, delivered to the agent as soon as the Edit landed).
The hook keeps only the last 20 lines (`tail -20`), so the fixture repr is
cut down to its last line:

```
Leakage tests failed after editing /Users/chewy/Desktop/bikeshare-open/src/features.py:
[714 rows x 29 columns]

    def test_rolling_mean_excludes_today(feats):
        """cnt_roll_7 must be the mean of the 7 calendar days *before* today."""
        by_date = feats.set_index("dteday")[TARGET]
        date = feats.dteday.iloc[200]
        window = [date - pd.Timedelta(days=k) for k in range(1, 8)]
        if all(d in by_date.index for d in window):
            expected = sum(by_date[d] for d in window) / 7
>           assert feats.set_index("dteday").cnt_roll_7[date] == pytest.approx(expected)
E           assert np.float64(4389.0) == 4414.857142857143 ± 0.00441486
E             
E             comparison failed
E             Obtained: 4389.0
E             Expected: 4414.857142857143 ± 0.00441486

tests/test_leakage.py:57: AssertionError
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_rolling_mean_excludes_today - assert np.fl...
1 failed, 7 passed in 0.58s
```

**Red `make test`** (the `E`/summary lines):

```
E           assert np.float64(4389.0) == 4414.857142857143 ± 0.00441486
E             
E             comparison failed
E             Obtained: 4389.0
E             Expected: 4414.857142857143 ± 0.00441486
FAILED tests/test_leakage.py::test_rolling_mean_excludes_today - assert np.fl...
1 failed, 15 passed in 2.18s
```

Only one test catches this, and it checks a single date (`iloc[200]`). It
also sits behind an `if all(...)` guard: if that window ever touches a
missing day, the test passes without asserting anything. That does not
happen today, because `reindex_to_calendar` fills the 2011-03-10 gap and row
200 falls well after it. Still, this fault is guarded by one assertion on
one row. `test_beats_the_mean_baseline` stayed green again, because a leaked
feature makes the model look better, not worse.

**Turns to fix: 1.** One Edit putting `.shift(1)` back. The hook re-ran
silently (exit 0), `git diff d6273b2 -- src/` printed nothing, and
`make test` gave `16 passed in 1.94s`. Between the hook stderr and the fix,
the agent made two calls the procedure requires: recording the diff and red
output, then committing the broken state. Neither is counted.
