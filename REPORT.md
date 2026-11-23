# Forecasting daily bikeshare demand: what the honest numbers say

Daily Capital Bikeshare rides (`cnt`), Washington DC, 2011–2012, from the
Molnar mirror of the UCI data (728 rows, 2011-03-10 missing). Train on
2011 (348 rows after lag warm-up), validate on H1 2012 (182), test on
H2 2012 (184). Every number below is in `reports/metrics.json`, which
`make all` regenerates exactly from a clean checkout.

## Results

Validation and test R^2 and MAE (rides/day), before and after my
Exercise 1 feature, `weather_severity_lag_1` (yesterday's weather). The
"before" rows are the starter pipeline (commit `d85172a`). The "after"
rows are the final `reports/metrics.json`.

| Feature set | Model | Valid R^2 | Valid MAE | Test R^2 | Test MAE |
|---|---|---:|---:|---:|---:|
| before (starter, 14 features) | baseline_mean | -0.9318 | 2002.2 | -2.1441 | 2847.7 |
| before (starter, 14 features) | baseline_persistence | 0.4507 | 900.6 | 0.5099 | 840.1 |
| before (starter, 14 features) | ridge | 0.6475 | 832.9 | 0.5833 | 947.1 |
| before (starter, 14 features) | hgb | 0.2185 | 1296.6 | -0.4975 | 1972.6 |
| after (+ `weather_severity_lag_1`, 15) | baseline_mean | -0.9318 | 2002.2 | -2.1441 | 2847.7 |
| after (+ `weather_severity_lag_1`, 15) | baseline_persistence | 0.4507 | 900.6 | 0.5099 | 840.1 |
| after (+ `weather_severity_lag_1`, 15) | ridge | 0.6670 | 803.9 | 0.6124 | 910.6 |
| after (+ `weather_severity_lag_1`, 15) | hgb | 0.2437 | 1275.2 | -0.4687 | 1950.4 |

The feature drops no rows (714 before and after), so the two baselines are
identical across the two halves of the table. I chose the feature on
validation alone: ridge +0.0195, hgb +0.0252. The "after" test numbers
were read once, when this report was written. R^2 is computed against the
test-period mean, so 0 means "as good as a constant nobody could have known
in advance". For reference, the leaky model from the tutorial
(`python -m src.leakage_demo`, with `casual` and `registered`) scores test
R^2 = 1.000000.

## Headline finding

The naive evaluation scores R^2 = 1.000, and that number is invalid:
`casual + registered == cnt` on every row, so the model learned addition,
not demand. Under a temporal split the real picture is different:

- **Mean baseline:** predicting the 2011 average scores −2.14 on test,
  because the H2 2012 test mean (6,043 rides) is 73% above the 2011
  training mean (3,493).
- **Persistence:** "yesterday's count" scores +0.510.
- **Ridge:** +0.612. It beats persistence by 0.10 R^2 but loses to it on
  MAE (911 vs 840 rides/day), so the gain is in the large errors, not the
  typical day.
- **Gradient boosting:** −0.47. That is negative but far better than the
  mean baseline. It fails for a structural reason: a tree cannot predict
  above the largest value it saw in training, and 56.5% of test days
  exceed the 2011 maximum of 6,043 rides (July 4, 2011).

Adding yesterday's weather raised validation R^2 by 0.020 for ridge and
0.025 for hgb, with both baselines unchanged, so I kept it. Its ridge
coefficient (+121 rides per SD) has the predicted sign: a bad-weather
yesterday depresses `cnt_lag_1`, and the model adds part of it back.

## Exercise 4: target transformation

The tree fails because the level moved out of its training range, so I
re-expressed the target relative to a lagged anchor the model already
sees. It learns `cnt − anchor` (diff) or `log(cnt / anchor)` (ratio). It
predicts the anchor plus or times that change, which lets the anchor carry
the level. The code is `AnchoredTarget` in `src/train.py`, behind
`--target-transform`. `make all` and the committed `reports/metrics.json`
are untransformed.

I scored four candidates **on validation only** (commit `3fd81f1`):

| Transform | Ridge valid R^2 | hgb valid R^2 | hgb valid MAE |
|---|---:|---:|---:|
| none | 0.6670 | 0.2437 | 1275.2 |
| diff, `cnt_lag_1` | 0.6675 | 0.6254 | 804.3 |
| ratio, `cnt_lag_1` | 0.3328 | 0.5858 | 866.4 |
| **diff, `cnt_roll_7`** | **0.6753** | **0.7149** | **729.4** |
| ratio, `cnt_roll_7` | 0.5472 | 0.7161 | 724.5 |
| *(anchor alone: predict `cnt_roll_7`)* | | *0.5945* | *823.9* |

Ratio/`cnt_roll_7` was best for hgb by 0.001, which is noise. But it cuts
ridge from 0.667 to 0.547, because an exponentiated linear model
misbehaves. Picking it would be winning the comparison by choosing the
transform that suits one model. I chose diff on `cnt_roll_7`, the best
transform applied to both models, and recorded the choice before scoring
test.

**I then touched test once** (commit `376566b`, file
`reports/target_diff_cnt_roll_7_test.json`):

| Model | Test R^2 | Test MAE |
|---|---:|---:|
| hgb, untransformed | -0.4687 | 1950.4 |
| hgb, diff/`cnt_roll_7` | **0.6282** | **839.2** |
| ridge, untransformed | 0.6124 | 910.6 |
| ridge, diff/`cnt_roll_7` | 0.6191 | 898.2 |
| persistence | 0.5099 | 840.1 |
| anchor alone (`cnt_roll_7`) | 0.4796 | 871.6 |

**Is it defensible?** I think yes, as a forecasting choice, for three
reasons:

- The anchor is a past-only lag already in the feature set, so no future
  information enters.
- It encodes a real belief: tomorrow is last week's level plus a weather
  and calendar deviation.
- The identical transform also helps ridge, so it fixes the target's
  scale; it does not tilt the comparison toward trees.

What it does **not** support is "trees beat linear models". On test, the
anchored tree is 0.009 R^2 from ridge, inside the noise of 184 days. It
only ties persistence on MAE (839 vs 840). Choosing among four candidates
on 182 validation days also had a cost: hgb's validation R^2 of 0.715
shrank to 0.628 on test. The accurate conclusion is that the tree's
failure was a target-scale problem, not a model-class problem. Once the
anchor carries the level, model class barely matters.

## Exercise 5: ridge vs persistence

| (test, n = 184) | Ridge | Persistence |
|---|---:|---:|
| R^2 / RMSE | 0.612 / 1084 | 0.510 / 1219 |
| MAE / median abs. error | 911 / 878 | 840 / 507 |
| Mean error (bias) | −523 | +16 |
| Days closer to the truth | 41% | 59% |
| MAE, 17 days persistence misses by >2,000 | 1,844 | 3,019 |
| MAE, other 167 days | 816 | 618 |

**Ridge is not worth deploying over "yesterday again" as it stands.** Its
standardised coefficients show what it is: `cnt_roll_7` (481) and
`cnt_lag_1` (470) dominate, followed by today's weather severity (−288).
It is persistence with a weather adjustment. That adjustment is real. On
the 17 test days where persistence misses by more than 2,000 rides
(weather swings, the day after a storm), ridge's MAE is 1,844 against
3,019. Those days dominate squared error, which explains the entire R^2
edge (0.612 vs 0.510). On the other 167 days ridge is worse (816 vs 618).
It is closer to the truth on only 41% of test days, and its median error
is 878 rides against 507. The cause is a level bias of −523 rides/day:
fitted on 2011, ridge shrinks every forecast toward 2011's lower level,
while persistence is unbiased (+16). The same pattern was visible on
validation before I touched test (bias −479; ridge closer on 51% of days;
persistence better on unchanged-weather days, 669 vs 762). A planner who
sees the forecast run 500 rides low on an ordinary Tuesday will stop
trusting it. Ridge earns deployment only if the cost of error is
genuinely quadratic, such as staffing for storm days. Even then, the
better design keeps persistence's level and adds ridge's weather
correction. The anchored model in Exercise 4 is roughly that, and it
matches persistence on MAE (839) while beating it on R^2 (0.628). This
analysis only reads residuals of the final models; nothing was refit
after it.

## Attribution

**Model:** Claude Opus 5.5 (`claude-opus-5-5`, 1M-token context), hosted
by Anthropic, driven through Claude Code 2.1.328. Not the local Ollama
model the lesson describes: Ollama was not installed, so `make model` and
`make check-model` were not run. I did not switch models at any point. The
same model ran:

- the main session (VS Code extension);
- three fresh headless `claude -p` sessions: `/seed-fault` for faults 2
  and 3, and the fault-4 guard-rail probes;
- the batch auditor (`AUDIT_MODEL=claude-opus-5-5`).

Results from a different model size are not comparable with classmates
who used `qwen3.5:9b`.

> **Before submitting:** the agent proposed every decision listed below
> as well as writing the code. Read each one and its line in
> `docs/DECISIONS.md`. Keep it only if you agree and can defend it, then
> delete this note.

**Files and functions the agent generated:**

- **`src/features.py`:** the `weather_severity_lag_1` line.
- **`src/config.py`:** `RAW_FEATURE_COLS`, `ENCODED_COLS`,
  `REDUNDANT_COLS`, `RAW_SHA256`.
- **`src/ingest.py`:** `raw_sha256`, `verify_raw`.
- **`src/train.py`:** `AnchoredTarget`, the `--target-transform` /
  `--anchor` / `--test` flags, and `fit_and_score`'s `models` / `splits`
  arguments.
- **`tests/test_leakage.py`:**
  - `test_weather_lag_is_yesterdays_weather`;
  - `test_no_feature_restates_a_forbidden_column`;
  - `test_unmapped_weather_label_fails_loudly`, written by a headless
    `/seed-fault` session;
  - the assert in `test_rolling_mean_excludes_today`.
- **`tests/test_contract.py`:** `test_every_raw_column_is_classified`,
  `test_raw_data_matches_pinned_hash`.
- **`.claude/`:** `hooks/protect_raw.py`, the PreToolUse entry in
  `settings.json`, and `skills/seed-fault/SKILL.md`.
- **Documents:** `.gitattributes`; `docs/SEEDED_FAULTS.md`, with the
  fault 2 and 3 sections drafted by the headless sessions; the dated lines
  in `docs/DECISIONS.md`; the `CLAUDE.md` edits; and this report's prose
  and tables.
- **Generated outputs:** `reports/*.json` and `reports/audit.jsonl` were
  written by code, not by hand.

Everything in the first commit is the course starter.

**Decisions** (each traces to a dated line in `docs/DECISIONS.md`):

1. **What to predict:** daily total `cnt`, from calendar, same-day
   weather and lagged demand. This is the starter's framing, kept.
2. **What counts as leakage:** `casual`, `registered` and any arithmetic
   restatement of `cnt`. Also, a column nobody has classified is treated
   as a failure, not admitted silently (the denylist plus classification
   contract line).
3. **How to split:** 2011 / H1 2012 / H2 2012 by date. Test was scored
   once for the main table and once for the chosen Exercise 4 transform.
4. **Which baselines:** the mean and persistence, plus the anchor alone
   when judging Exercise 4.
5. **Whether to keep my feature:** kept, at +0.0195 ridge validation R^2
   (the feature line).
6. **Which target transform, and whether it is defensible:** diff on
   `cnt_roll_7`. Defensible, but a tie with ridge, not a win (the two
   Exercise 4 lines).
7. **Whether the answer is believable:** ridge's edge over persistence is
   real in R^2 but not on a typical day, so it is not deployed as it
   stands (the Exercise 5 line).
8. **Methodological guard rails:**
   - raw data is immutable by any route (the two-layer guard line);
   - a test is added wherever a seeded fault stayed green;
   - the `CLAUDE.md` changes.
