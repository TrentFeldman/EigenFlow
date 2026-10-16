# Task: small readability cleanup in `ode/`, with no behavior change

This repo is EigenFlow, at `/home/trent/Coding/Projects/ML/ODE`. The `ode/`
package was just refactored for readability, and that refactor is verified to
behave exactly like the original. Make five small follow-up fixes. Read
`AGENTS.md` and each file you touch before editing.

## Rules
- Do not change behavior. That includes outputs, file formats, CLI flags and
  JSON keys, error conditions, and RNG streams.
- Only touch the files named below. Do not add modules or dependencies, do not
  edit markdown, and do not commit.
- Keep the current docstring style: Google-style `Args:`/`Returns:`, with array
  shapes given using T, M, B and n.

## Fixes

1. **`ode/dataset.py:337-340`, `Normalizer.fit`.** Replace the
   `statistics_index` loop with two explicit calls, one for features and one
   for targets. Keep each one's running `(count, mean, squared_deviations)` as
   a tuple, like this:
   ```python
   feature_stats = _merge_moments(features, feature_mask, *feature_stats)
   target_stats = _merge_moments(target, mask, *target_stats)
   ```
   Also make `_merge_moments` (line 269) pure: it should return new arrays and
   never modify its inputs in place. The math must stay identical. It uses
   Chan's parallel variance merge, which is robust for constant columns, so do
   not swap it for a sum/sum-of-squares formula.

2. **`ode/dataset.py:78-79`, `encode`.** The chained assignment
   `features[:, layout["coefficients"]][:, :equation.order] = ...` only works
   because NumPy basic slicing returns a view. Rewrite it so a single indexing
   step names the target columns directly. For example, build the column slice
   from `layout["coefficients"].start` and `equation.order`. Do the same for
   `initial_state`.

3. **`ode/config.py:84-131`, field comments.** Remove `(unitless)` where a unit
   makes no sense: the seed, method, family names and backend. Keep units only
   on real quantities: times, rates, tolerances and state bounds. Leave the
   meaning of each comment unchanged.

4. **Lumped validation.** Split these so that each failing condition raises its
   own specific message:
   - `Normalizer.__init__` (`ode/dataset.py:307`)
   - `masked_mse` (`ode/model.py`)
   - `train.py:139-142` ("Counts/rate must be positive...")

   The same inputs must still be rejected, with the same exception types. Only
   the message text may change. For train.py, a small `positive_int` argparse
   type is acceptable.

   Also reduce the 9-parameter `_fit` (`ode/train.py:243`). For example, pass
   the two loaders as a tuple, or have `_fit` read `args.run` and
   `args.epochs` itself. Do not add a class for this.

5. **Tests.**
   - In `ode/tests/test_data.py`, replace
     `fingerprint([equation.coefficients, 0., 0., 0.])` with
     `equation.parameter_id`. If `fingerprint` becomes unused, drop the import.
   - Revert over-long comprehension variable names such as `job_index` back to
     `i` where the scope is a single line.
   - Wrap every line in `ode/` and `ode/tests/` that is over 110 characters.
     Find them with
     `grep -rnE '.{111,}' ode --include=*.py`.
   - Remove the double blank line before
     `test_failed_refined_solve_preserves_call_counts` in
     `ode/tests/test_reference.py`.

## Verify
```bash
S=/tmp/claude-1000/-home-trent-Coding-Projects-ML-ODE/4b88a66a-e777-4690-8b2d-de62d0d95419/scratchpad
PY=/home/trent/Coding/Projects/ML/ODE/.venv/bin/python
# BEFORE editing:
cd $S && PYTHONPATH=/home/trent/Coding/Projects/ML/ODE $PY snapshot.py $S/out_before $S/fixture.pt
# AFTER editing:
cd $S && PYTHONPATH=/home/trent/Coding/Projects/ML/ODE $PY snapshot.py $S/out_after $S/fixture.pt
$PY $S/compare.py $S/out_before $S/out_after   # every line must say SAME
cd /home/trent/Coding/Projects/ML/ODE && $PY -m unittest discover -s ode/tests -v   # 25 tests must pass
```

The snapshot script checks a lot of real behavior: sampled jobs, dataset
loading, normalization, encoding, reference validation, prediction, both CLI
dry runs, a tiny real generation, and a 2-epoch training run. It must run from
`$S`, and its output directories must not exist yet. If `$S` is gone, write an
equivalent before/after check in `/tmp` first.

In your report, give each fix and the file and lines it changed, and paste the
`compare.py` result.
