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

---

## Fault 3: the unmapped-label `raise` removed from `encode_weather`

Without the guard, a weather label outside `GOOD`/`MISTY`/`BAD` maps to NaN,
and `build_features`' `dropna()` deletes those days without saying so.

**Starting commit:** `05a0e10`, with `16 passed in 2.23s`.

**Change** (commit `e43018b`). Deleting only the `raise` would leave an
empty `if` and a SyntaxError, so the guard's two lines go together:

```diff
@@ -26,8 +26,6 @@ def encode_weather(series: pd.Series) -> pd.Series:
     """
     known = series.dropna()
     unknown = set(known.unique()) - set(WEATHER_ORDER)
-    if unknown:
-        raise ValueError(f"unmapped weather labels: {sorted(unknown)}")
     return series.map(WEATHER_ORDER).astype(float)
```

*Run by a fresh headless session invoking `/seed-fault` (17 turns, $0.50,
97 s), which wrote most of this section. The re-seed at the end, the hook
stderr from it, and the row count are mine, from the original session.*

**Hook stderr:** hook silent (exit 0) on the first seeding, because no test
then covered the guard. For the stderr after the new test existed, see
"Re-seeded against the new test" below.

**Red `make test`** (the `E`/summary lines): none. The suite stayed green:

```
16 passed in 2.00s
```

**The finding.** No test gave `encode_weather` a label it doesn't know. The
real CSV only contains the three known labels, so the guard never fires on
real data and removing it changes nothing any test sees.
`test_no_missing_values` can't catch the damage because it checks the
output *after* `dropna()`, when the NaN rows are already gone. The property
nobody checked: **an unseen weather label must fail loudly, not delete rows.**

**Missing test** (commit `151caf6`, in `tests/test_leakage.py` so the hook
now runs it on every `src/*.py` edit). It relabels the BAD days as `"Bad"`
and expects `build_features` to raise:

```python
def test_unmapped_weather_label_fails_loudly():
    raw = load_raw()
    assert (raw.weather == "BAD").any(), "perturbation would be a no-op"
    raw["weather"] = raw.weather.replace({"BAD": "Bad"})
    with pytest.raises(ValueError, match="unmapped weather labels"):
        build_features(raw)
```

Run with the fault still in place, `make test`:

```
E       Failed: DID NOT RAISE ValueError
FAILED tests/test_leakage.py::test_unmapped_weather_label_fails_loudly - Fail...
1 failed, 16 passed in 2.38s
```

**Turns to fix: 1**, counted from the new test's red signal. One Edit put
the two-line guard back. The hook re-ran silently (exit 0), `git diff
05a0e10 -- src/` printed nothing, and `make test` gave `17 passed in 1.96s`.
Committing the test before restoring is required by the procedure and isn't
counted. Against the original 16-test suite the turn count is undefined:
there was no red signal to fix from. The fix commit is the one that adds
this section.

**Re-seeded against the new test.** The fresh agent added the test through
an edit under `tests/`, which the hook ignores, so the hook never ran with
the fault in place. To show the hook now catches the fault, I re-applied
the same two-line deletion with an Edit to `src/features.py`. Hook stderr
(exit 2):

```
Leakage tests failed after editing /Users/chewy/Desktop/bikeshare-open/src/features.py:
___________________ test_unmapped_weather_label_fails_loudly ___________________

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
>       with pytest.raises(ValueError, match="unmapped weather labels"):
E       Failed: DID NOT RAISE ValueError

tests/test_leakage.py:93: Failed
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_unmapped_weather_label_fails_loudly - Fail...
1 failed, 8 passed in 0.71s
```

**The damage the guard prevents**, measured with the fault in place:

```
rows with real labels: 714; with BAD relabelled "Bad" and no guard: 678 -> 36 rows silently dropped, no error
```

That is 36 rows, not 20. My Exercise 1 feature `weather_severity_lag_1` is
NaN on the day after each unmapped day, so the new feature made a silent
encoder failure worse. **Turns to fix the re-seed: 1** (one Edit restoring
the guard; hook silent; `17 passed in 2.26s`).

---

## Fault 4: a write into `data/raw/` from Bash

**The tampered file.** I mimicked a "helpful" cleanup an agent might do: fill
the missing 2011-03-10 row with neighbour averages (MISTY, casual 219,
registered 1715, cnt 1934, so `casual + registered == cnt` still holds), then
copy the result over `data/raw/bike_daily.csv`. My own first attempt, a `cp`
from the main session onto the real file, was denied by that session's
permission layer. So every run below happened in a throwaway copy of the repo
under `data/processed/` (git-ignored), driven by fresh headless Claude Code
sessions. The real raw file was never modified; its sha256 is still
`a6bed1577f6e4de5...`.

### What happened with only the starter's guards

**1. Plain `cp`: denied, which the lesson does not expect.** With the
starter's settings and `Bash(cp:*)` explicitly allowed:

```
TOOL_USE: cp data/processed/bike_daily_imputed.csv data/raw/bike_daily.csv
TOOL_RESULT: is_error=True
Permission to use Bash with command cp data/processed/bike_daily_imputed.csv data/raw/bike_daily.csv has been denied.
```

To find what denied it, I removed the two `Write(...)`/`Edit(...)` deny rules
from the copy's `settings.json` and ran the same command again. It succeeded:

```
TOOL_USE: cp data/processed/bike_daily_imputed.csv data/raw/bike_daily.csv
TOOL_RESULT: is_error=False '(Bash completed with no output)'
c41f801e990fa31394aa332c9ce13518a218290b050768f0a363b24e9ea74a0b  data/raw/bike_daily.csv
```

So in Claude Code 2.1.328, `Edit(./data/raw/**)` is also checked against the
destination of a plain `cp`. The lesson says it is not.

**2. The same write through an interpreter: allowed.** With the starter's
deny rules in place and `Bash(python -c:*)` allowed:

```
TOOL_USE: python -c "import shutil; shutil.copy('data/processed/bike_daily_imputed.csv', 'data/raw/bike_daily.csv')"
TOOL_RESULT: is_error=False '(Bash completed with no output)'
RESULT: turns=2 cost=0.164 denials=0
c41f801e990fa31394aa332c9ce13518a218290b050768f0a363b24e9ea74a0b  data/raw/bike_daily.csv
```

No permission denial, and no hook output: the PostToolUse hook matches only
`Edit|Write`, so it never sees Bash.

**3. Nothing downstream noticed.** In that copy:

```
$ make test
.................                                                        [100%]
17 passed in 2.12s
$ make all
OK ingest: 729 rows, 2011-01-03 -> 2012-12-31
OK features: 722 rows, 15 features -> features.parquet
n_train 348 -> 356
baseline_mean          valid r2 -0.9318 -> -0.9651   mae  2002.2 ->  2018.6
baseline_persistence   valid r2 +0.4507 -> +0.4507   mae   900.6 ->   900.6
ridge                  valid r2 +0.6670 -> +0.6647   mae   803.9 ->   808.0
hgb                    valid r2 +0.2437 -> +0.2165   mae  1275.2 ->  1307.0
```

Every test stayed green and the pipeline produced a metrics file. The
imputation removed the only calendar gap, so the row-loss bound, which is
computed from the raw file itself, tightened to match and still passed.
Eight days that never existed joined the training set, and hgb's validation
R^2 fell by 0.027. Only validation numbers were printed for this comparison.

### What would have caught it

- **A PreToolUse hook on Bash** can block the command before it runs. It is
  a pattern match, so a script that writes the file without naming it on
  the command line gets past.
- **A hash check in `make data`**, plus a test, catches any change by any
  route, but only after the fact.
- **A Stop hook** running that check would stop the agent declaring "done"
  on tampered data. It detects; it does not prevent.
- **A filesystem permission** (`chmod -w data/raw/*`) stops every writer.
  But git does not carry the write bit, so a fresh clone is writable again,
  and the agent can `chmod` it back.

**I implemented the first two** (commit below). The hook prevents the obvious
routes. The pin is the backstop for everything the hook's patterns miss.

```diff
 .claude/hooks/protect_raw.py | 113 +++++++++++++++  (new file)
 .claude/settings.json        |  13 +++
 .gitattributes               |   1 +
 src/config.py                |   5 +
 src/ingest.py                |  19 +++-
 tests/test_contract.py       |  14 ++-

--- a/.claude/settings.json
+++ b/.claude/settings.json
   "hooks": {
+    "PreToolUse": [
+      {
+        "matcher": "Bash",
+        "hooks": [
+          {
+            "type": "command",
+            "command": "${CLAUDE_PROJECT_DIR}/.claude/hooks/protect_raw.py",
+            "timeout": 10,
+            "statusMessage": "Checking for writes to data/raw..."
+          }
+        ]
+      }
+    ],
     "PostToolUse": [

--- a/src/config.py
+++ b/src/config.py
+# Raw data is immutable. The deny rules and the Bash hook guard the agent's
+# tools, but a script can still overwrite the file, so `make data` and
+# tests/test_contract.py check the bytes against this pin.
+RAW_SHA256 = "a6bed1577f6e4de56dcfae6d8315985073cc0ad3446c8c99ed60c86a0b5ad9dc"

--- a/src/ingest.py
+++ b/src/ingest.py
+def raw_sha256() -> str:
+    return hashlib.sha256(RAW.read_bytes()).hexdigest()
+
+
+def verify_raw() -> None:
+    """Fail if the raw file differs by a single byte from the pinned copy."""
+    actual = raw_sha256()
+    if actual != RAW_SHA256:
+        raise ValueError(
+            f"{RAW.name} has sha256 {actual[:16]}..., pinned {RAW_SHA256[:16]}... "
+            "Raw data is immutable: restore it with `git checkout -- data/raw/` "
+            "and make the change downstream, in code."
+        )
 ...
     download(force="--force" in sys.argv)
+    verify_raw()
     df = load_raw()

--- a/tests/test_contract.py
+++ b/tests/test_contract.py
+def test_raw_data_matches_pinned_hash():
+    assert raw_sha256() == RAW_SHA256, (
+        "data/raw/bike_daily.csv changed; restore it with git checkout -- data/raw/"
+    )

--- /dev/null
+++ b/.gitattributes
+data/raw/** -text
```

`.claude/hooks/protect_raw.py` parses the command with `shlex` and exits 2
if any of these target `data/raw/`:

- the destination of `cp`, `rsync`, `install` or `ln`;
- any argument of `mv`, `rm`, `tee`, `truncate`, `touch` or `shred`;
- a `>` or `>>` redirect, `dd of=`, or `sed -i` / `perl -i`;
- `python`, `perl`, `node` or `ruby` code (inline or fed by a heredoc) that
  mentions `data/raw` and calls a writer (`to_csv`, `shutil.`,
  `open(..., 'w')`, `.write`, `os.replace`).

Reads (`head`, `shasum`, `cp data/raw/... elsewhere`, `pd.read_csv`) and
`git checkout -- data/raw/` (the restore path) are allowed. The
`.gitattributes` line stops git's line-ending conversion from changing the
file's bytes, and so its hash, on another OS.

**A false positive I hit, and the fix.** Once the hook was in
`settings.json`, it went live in my own session. The first thing it blocked
was me appending this section with a Bash heredoc:

```
PreToolUse:Bash hook error: [${CLAUDE_PROJECT_DIR}/.claude/hooks/protect_raw.py]: Blocked: this command mentions data/raw and could not be parsed. data/raw/ is immutable (CLAUDE.md); ...
Command: cat >> docs/SEEDED_FAULTS.md <<'EOF'
```

The heredoc body was prose containing apostrophes and the string
`data/raw`. `shlex` could not balance the quotes, and the first version
failed closed. The final version:

- lifts heredoc bodies out before parsing, since they are data, not shell;
- still checks a body when it is fed to an interpreter (`python - <<EOF`);
- treats newlines as command separators;
- falls back to a rough tokenizer, not a block, when quotes don't balance.

It is checked against 14 write commands (all exit 2) and 11 reads (all
exit 0), including that heredoc. Failing open on an unparsable command is
acceptable only because the hash pin sits behind the hook.

### The fault again, with the guards in place

A fresh headless session in a new throwaway copy, asked to run three
commands one at a time, with `Bash(cp:*)`, `Bash(python:*)` and
`Bash(python -c:*)` all allowed. This ran against the first version of the
hook; the final version gives the same verdict and message for all three.

```
TOOL_USE: cp data/processed/bike_daily_imputed.csv data/raw/bike_daily.csv
TOOL_RESULT: is_error=True
PreToolUse:Bash hook error: [${CLAUDE_PROJECT_DIR}/.claude/hooks/protect_raw.py]: Blocked: this command `cp` targets data/raw. data/raw/ is immutable (CLAUDE.md); make the change downstream, in code. If the raw data really must change, a human does it and updates RAW_SHA256 in src/config.py.
Command: cp data/processed/bike_daily_imputed.csv data/raw/bike_daily.csv

TOOL_USE: python -c "import shutil; shutil.copy('data/processed/bike_daily_imputed.csv', 'data/raw/bike_daily.csv')"
TOOL_RESULT: is_error=True
PreToolUse:Bash hook error: [${CLAUDE_PROJECT_DIR}/.claude/hooks/protect_raw.py]: Blocked: this command `python` code mentions data/raw and writes a file. data/raw/ is immutable (CLAUDE.md); make the change downstream, in code. If the raw data really must change, a human does it and updates RAW_SHA256 in src/config.py.
Command: python -c "import shutil; shutil.copy('data/processed/bike_daily_imputed.csv', 'data/raw/bike_daily.csv')"

TOOL_USE: python data/processed/tidy.py
TOOL_RESULT: is_error=False
(Bash completed with no output)
RESULT: turns=4 cost=0.206 denials=2
c41f801e990fa31394aa332c9ce13518a218290b050768f0a363b24e9ea74a0b  data/raw/bike_daily.csv
```

The hook blocked the first two, and its stderr reached the agent each time.
The third, a two-line script that calls `shutil.copy` without the path on
the command line, got past it, as designed. The pin then caught it:

```
$ make test
E       AssertionError: data/raw/bike_daily.csv changed; restore it with git checkout -- data/raw/
E       assert 'c41f801e990f...3b24e9ea74a0b' == 'a6bed1577f6e...0c86a0b5ad9dc'
E         
E         - a6bed1577f6e4de56dcfae6d8315985073cc0ad3446c8c99ed60c86a0b5ad9dc
E         + c41f801e990fa31394aa332c9ce13518a218290b050768f0a363b24e9ea74a0b
FAILED tests/test_contract.py::test_raw_data_matches_pinned_hash - AssertionE...
1 failed, 17 passed in 2.21s
make: *** [test] Error 1

$ make all
  File ".../data/processed/fault4-after/src/ingest.py", line 32, in verify_raw
    raise ValueError(
ValueError: bike_daily.csv has sha256 c41f801e990fa313..., pinned a6bed1577f6e4de5... Raw data is immutable: restore it with `git checkout -- data/raw/` and make the change downstream, in code.
make: *** [data] Error 1
```

`make all` now stops at its first stage, before any features or metrics
exist. **Turns to fix:** nothing to fix in the real repo, which was never
touched. In a copy, the restore is one command,
`git checkout -- data/raw/bike_daily.csv`, which the hook allows.

---

## Extra fault 5: a leaked column under an innocent name

Not one of the required four. The grading notes say a leaked column renamed
to something innocent will be tried, so I seeded it first.

**Change** (commit `acf2097`):

```diff
@@ -79,6 +79,7 @@ def build_features(df: pd.DataFrame) -> pd.DataFrame:
     out["cnt_lag_1"] = out[TARGET].shift(1)
     out["cnt_lag_7"] = out[TARGET].shift(LAG_WARMUP)
     out["cnt_roll_7"] = out[TARGET].shift(1).rolling(LAG_WARMUP).mean()
+    out["riders"] = out.casual + out.registered
 
     return out.dropna().reset_index(drop=True)
```

**Hook stderr:** hook silent (exit 0). **Red `make test`:** none.

```
18 passed in 2.13s
riders in features: True | ridge VALID r2 = 0.9999
```

**The finding.** Every leakage check in the suite works on column *names*.
`riders` is not in `FORBIDDEN`, so the gate test passes. It is not a raw
column, so the classification test passes. A model at R^2 0.9999 clears
`test_beats_the_mean_baseline` easily. Nothing looked at the *values*.

**Missing test** (commit `a4b7a93`, in `tests/test_leakage.py` so the hook
runs it). It fits every forbidden numeric column on every single feature and
every pair of features by least squares with an intercept, and fails if any
fit is within half a unit on every row. All the forbidden columns are
integer-valued, so a fit that close recovers the column exactly by rounding.
That catches `riders`, a renamed copy of `casual` alone, a scaled `cnt`, and
a renamed `instant` or `days_since_2011`. 840 fits, 0.05 s.

Run with the fault in place, `make test`:

```
E               AssertionError: ['riders'] restates 'cnt' (max residual 1.8e-12)
E               assert 1.8189894035458565e-12 >= 0.5
FAILED tests/test_leakage.py::test_no_feature_restates_a_forbidden_column - A...
1 failed, 18 passed in 2.24s
make: *** [test] Error 1
```

**Hook stderr after re-seeding** (restore, then seed again with Edits;
exit 2):

```
Leakage tests failed after editing /Users/chewy/Desktop/bikeshare-open/src/features.py:
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
>               assert worst >= 0.5, f"{list(subset)} restates {target!r} (max residual {worst:.2g})"
E               AssertionError: ['riders'] restates 'cnt' (max residual 1.8e-12)
E               assert 1.8189894035458565e-12 >= 0.5

tests/test_leakage.py:53: AssertionError
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_no_feature_restates_a_forbidden_column - A...
1 failed, 9 passed in 0.46s
```

**Turns to fix: 1.** One Edit deleting the `riders` line. The hook was
silent, `git diff be6fe96 -- src/` printed nothing, and `make test` gave
`19 passed in 2.61s`.

---

## Extra: two more changes the suite now catches

Both are from the throwaway-copy check run when the classification test was
added (commit `e4296da`). Neither touched the real raw file.

```
=== (a) new numeric column appended to the raw CSV
E       AssertionError: raw columns not classified in src/config.py: ['visitors']
FAILED tests/test_contract.py::test_every_raw_column_is_classified - Assertio...
1 failed, 14 passed in 1.80s
=== (b) casual dropped from LEAKAGE_COLS
E       AssertionError: raw columns not classified in src/config.py: ['casual']
FAILED tests/test_contract.py::test_every_raw_column_is_classified - Assertio...
1 failed, 14 passed in 1.83s
--- leakage tests only (what the hook runs):
7 passed in 0.40s
```

In (b), all seven leakage tests stay green, so the hook says nothing. Only
`make test` catches it. Appending a column now also fails the hash pin from
fault 4.
