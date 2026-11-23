---
name: seed-fault
description: Deliberately break the pipeline to prove the tests catch it. Seeds one fault, records the hook stderr and red test output, restores the code, and logs the evidence in docs/SEEDED_FAULTS.md. Use when asked to seed, inject or simulate a bug, or to check whether the suite would catch a given mistake.
argument-hint: <the fault to seed, e.g. "delete the shift(1) from cnt_lag_7">
allowed-tools: Read, Edit, Write, Bash(make*), Bash(python -m pytest*), Bash(git diff*), Bash(git status*), Bash(git log*), Bash(git add*), Bash(git commit*)
---

# Starting state

!`git status --short; git log --oneline -1`

# Seeding a fault: $ARGUMENTS

A suite that stays green under a seeded fault is not a verification suite.
What you produce here is evidence, so paste real output, never a description
of it.

1. **Start clean.** The tree above must have no changes under `src/` or
   `tests/`, and `make test` must pass. If not, stop and say so: a fault
   seeded on a red suite proves nothing. Note the starting commit hash.

2. **Seed it with one Edit**, the smallest change that implements
   "$ARGUMENTS". Use the Edit tool, never `sed` or a Bash redirect, so the
   PostToolUse hook sees it. Never touch `data/raw/`. If the fault is about
   raw data, describe the change you would make, name what should catch it,
   and stop.

3. **Record before fixing anything:**
   - `git diff -- src/`: the exact change.
   - The hook's stderr, verbatim. If none arrived, write "hook silent
     (exit 0)". The hook runs only `tests/test_leakage.py`, and only after
     edits to `src/*.py`.
   - `make test 2>&1 | grep -E "^(E  |FAILED|[0-9]+ (passed|failed))"`
   - Commit the broken state: `Seed fault N: <what you changed>`.

4. **If everything stayed green, that is the finding.** Name the property
   no test checks and write the missing test. Put it in
   `tests/test_leakage.py`, the only file the hook runs, unless it needs a
   trained model. Run it with the fault still in place to show it goes red,
   then commit it before restoring. A test never seen failing against the
   real fault proves nothing.

5. **Restore** with one Edit that reverses step 2. Confirm
   `git diff <starting commit> -- src/` prints nothing and `make test`
   passes.

6. **Log it.** Add a `## Fault N` section to `docs/SEEDED_FAULTS.md` with
   the diff, hook stderr, red test lines, the commit hashes, and the turns
   to fix: agent tool calls from the red signal to green. Commit as
   `Fix fault N: <what you restored>`.

Finish with one line: which tests went red, whether the hook fired, and how
many turns the fix took.
