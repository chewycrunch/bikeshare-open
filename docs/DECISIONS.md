# Decisions

One line per methodological decision: what you chose, why, and what it changed.
Agents append here via the `/new-feature` skill; you append here by hand.

- Exclude `casual`, `registered`: they sum to `cnt` exactly. Leaky model R^2 1.000 -> honest ridge 0.583.
- Exclude `cnt_2d_bfr`: a shipped 2-day lag we cannot test the alignment of. Build our own lags instead.
- Exclude `instant`, `yr`, `days_since_2011`: time indexes let a tree predict a flat constant past 2011 and a linear model extrapolate a trend we have no grounds for.
- Split 2011 / H1 2012 / H2 2012 by date. Test set touched once, after all feature work.
- Report mean and persistence baselines with every score. Persistence (test R^2 0.510, MAE 840) is within 0.07 R^2 of ridge and beats it on MAE; that caveat goes in the report.
- 2026-11-23 Keep the `FORBIDDEN` denylist but make it fail loudly: every raw column must be classified in `src/config.py` (`RAW_FEATURE_COLS`, `ENCODED_COLS`, `REDUNDANT_COLS`), enforced by `test_every_raw_column_is_classified`. A loud failure beats an allowlist's silent drop. Checked: an appended numeric column and `casual` removed from `LEAKAGE_COLS` both go red; the latter leaves all 7 leakage tests green.
- 2026-11-23 Add `weather_severity_lag_1`, yesterday's weather (via the `/new-feature` steps). Availability: it is the weather observed on d-1, known before d begins. Validation R^2 ridge 0.6475 -> 0.6670 (+0.0195), hgb 0.2185 -> 0.2437 (+0.0252); rows unchanged (714), so both baselines are identical. Kept: above the skill's 0.01 floor, though still within the noise of 182 validation days. Test numbers not consulted.
- 2026-11-23 Guard raw data in two layers. Prevention: a PreToolUse hook on Bash (`.claude/hooks/protect_raw.py`) blocks commands that write into `data/raw/`. Detection: the raw file's SHA-256 is pinned in `src/config.py` and checked by `make data` and `test_raw_data_matches_pinned_hash`. Why: `python -c "shutil.copy(...)"` got past the deny rules, and an imputed 2011-03-10 passed all 17 tests while adding 8 training rows (hgb valid R^2 -0.027). The hook fails open on unparsable commands because the pin backs it up.
