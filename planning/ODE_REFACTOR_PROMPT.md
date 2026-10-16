# Task: make `ode/` readable and properly documented, with no behavior change

You are refactoring the Python package in `ode/` of the EigenFlow repo (repo root:
`/home/trent/Coding/Projects/ML/ODE`). The code works: all 24 tests pass in about
1 second, and a real 384-trajectory dataset exists at `ode/data/linear-v1/` that
loads correctly. The problem is readability. The code is dense and full of clever
one-liners, the functions are too long, there are magic numbers, logic is
duplicated, and the docstrings are full of defensive caveats instead of saying
what a function takes and returns. One of the two maintainers knows Python well
but is still learning ODEs, so the code has to explain the math where it happens.

Read `AGENTS.md` first (section "Documentation for our two-person team"), then
`ode/MATH_GUIDE.md`. Then read every file in `ode/*.py` and `ode/tests/*.py` in
full before you edit anything.

## Hard constraints

- **Behavior must not change.** That covers sampled data and RNG seeding, file
  formats (`manifest.json`, `records.jsonl`, `.npz` keys, checkpoint dict keys,
  the `"linear-padded-v1"` encoding), CLI flags, CLI output JSON keys, error
  conditions, and numerical results. `ode/data/linear-v1/` must still load, and
  every `parameter_id` must stay byte-identical.
- Scope is `ode/*.py` and `ode/tests/*.py` only. Do not touch `fluid/`,
  `.experiments/`, `.venv/`, `planning/`, `ode/data/`, `ode/runs/`, or the
  untracked file at the repo root. You may update the file-responsibility table
  in `ode/README.md` if a responsibility moves. Do not edit any other markdown.
- No new dependencies and no new modules. Put helpers in the existing file
  where they belong.
- Do not generate datasets, train models, or commit.
- Tests patch these names, so they must remain importable at these paths, or
  you must update the tests in the same change: `ode.generate_data.solve_job`,
  `ode.generate_data.validate_reference`, `ode.generate_data.iter_jobs`,
  `ode.reference.solve_ivp`, `ode.reference.linear_reference`,
  `ode.train.TrajectoryDataset`, `ode.train.run_epoch`, `ode.dataset.np.load`,
  `ode.train.torch.cuda.is_available`. `solve_ivp` must still receive `rtol=`
  as a keyword argument, because a test inspects it.

## Step 0: capture a golden baseline before any edit

Write a throwaway script outside the repo, for example under `/tmp`. Use it to
pickle the following values, then rerun it after the refactor and assert
exact equality. Use `np.testing.assert_allclose(rtol=1e-12)` only for the
normalizer statistics, and only if you reorganize that math.

1. `[(j.trajectory_id, j.parameter_id, j.split, asdict(j.equation)) for j in iter_jobs(GeneratorConfig.load("ode/configs/generator.json"))]`
2. For `TrajectoryDataset("ode/data/linear-v1", split, 6)` with split set to
   `train` and to `validation`, record `len()`, plus `dataset[i]` for i in
   `0, 1, 100, 5000, len-1`, before and after assigning
   `Normalizer.fit(train)`. Also record `Normalizer.fit(train).to_dict()`.
3. `encode(LinearODE((6., 11., 6.), (0.3, -0.1, 0.2), 0.4, 1.3, 0.4), np.linspace(0, 5, 7), 6)`
4. `validate_reference(GeneratorConfig.load("ode/configs/generator.json"))`:
   for each case, record the name, `passed`, `error_ratio`, and
   `matrix_exponential_error_ratio`.
5. `predict()` output from an untrained checkpoint built after
   `torch.manual_seed(0)`. Copy the fixture in `tests/test_inference.py`.
6. The stdout of both dry runs: `python -m ode.generate_data --config ode/configs/generator.json`
   and `python -m ode.train --config ode/configs/generator.json`.

Work one file at a time in dependency order:

equations → analytic → config → reference → validate_reference → generate_data → dataset → model → predict → train → tests

After each file, run
`.venv/bin/python -m unittest discover -s ode/tests -v`.

## Documentation standard (apply to every module)

- **Module docstring.** In 3 to 6 lines, say what the module does and where it
  sits in the pipeline: `config → generate_data (uses equations, reference,
  analytic, validate_reference) → dataset → train (uses model) → predict`. Say
  it in plain words.
- **Public function and class docstrings** use Google style: a one-line summary,
  then `Args:`, `Returns:`, and `Raises:` where each applies. Every array
  argument and return value states its shape and dtype.
- Use one notation everywhere and define it once, in the `ode/__init__.py`
  docstring: `T` = number of time points, `M` = `max_order` (the padding
  width), `B` = batch size, `n` = the equation's order, and `state = [y, y',
  ..., y^(n-1)]`. All values are nondimensional. Link to MATH_GUIDE.md rather
  than repeating it.
- **Delete defensive meta-commentary.** Remove sentences like "Importing this
  module never integrates an equation", "This function is intentionally never
  called during a dry-run", "No test calls the real solve_ivp", "not a claim
  about accuracy", and "A 96 GB GPU does not remove...". Keep a caveat only when
  it is a real pitfall for the caller, and put it on a single `Note:` line.
- **Comments say why, not what.** Wherever there is math, add a small worked
  example with real numbers. Do not write essays: a public docstring should
  rarely run past 15 lines.

## Specific fixes, in priority order

### 1. One source of truth for the feature layout (biggest readability win)
`3 * max_order + 5` and the slice offsets are hard-coded in
`dataset.py:36-45`, `:128`, `:209`, `:221-229`, `:252-253`, `model.py:29`,
`tests/test_data.py:119` (`17`), and `tests/test_inference.py:37,41` (`29`, `17`).

In `dataset.py`, add `feature_width(max_order)` and one helper that returns
named slices: coefficients, initial_state, mask, forcing, time. Use them
everywhere listed above.

In the `encode` docstring, add a layout table for M=6:
`[0:6) a0..a5 | [6:12) y(0)..y^(5)(0) | [12:18) mask | 18 amplitude | 19 frequency | 20 sin(phase) | 21 cos(phase) | 22 elapsed time`.

### 2. Remove the duplicated equation fingerprint
`generate_data.py:28-30,54` and `dataset.py:91-93` compute the same SHA-256
twice, by different code. Move `fingerprint` into `equations.py` and add a
`LinearODE.parameter_id` property (coefficients, amplitude, frequency, phase).
Use it in both places, and import or re-export it so the tests still work.

In `LinearODE.__post_init__`, also convert `amplitude`, `frequency` and
`phase` to `float`. Otherwise `amplitude=1` hashes as `"1"` but
`amplitude=1.0` hashes as `"1.0"`.

Verify the change: every `parameter_id` in `ode/data/linear-v1/records.jsonl`
must equal `LinearODE(**record["equation"]).parameter_id`.

### 3. Flatten `reference.solve_job` (`reference.py:50-114`)
This function nests two closures (`integrate`, then `rhs` with `nonlocal
calls`) that mutate `result.nfev` from the inside, and it wraps everything in
one large `try`.

Split it into module-level pieces:
- `_integrate(equation, config, times, tolerance_divisor) -> (states, nfev)`.
  This one can keep a small inner `rhs` closure for the call budget.
- `_refinement_ratio(coarse, refined, config) -> float`
- `_independent_ratio(refined, equation, times, config) -> float`

`solve_job` should then read as roughly six top-level steps.

The two checks use different metrics: a per-component L2 norm over the whole
trajectory, and a pointwise error at selected times. Explain each in its own
docstring with its formula. Rename the near-identical `allowed` and
`allowance`. Document why the broad `except Exception` exists: every failure
becomes a ledger record instead of crashing the whole run.

### 4. Make `GeneratorConfig` self-explanatory (`config.py:18-98`)
- Add a short comment above each of the 24 fields giving its meaning, unit,
  and effect. In particular, explain the difference between `rtol`/`atol`
  (solver tolerances) and `reference_rtol`/`reference_atol` (acceptance
  thresholds for the checks). Also explain `refinement_factor`,
  `independent_check_points`, `max_abs_state`, `max_rhs_calls`, and
  `groups_per_split` (counted per order and per family). Note that
  `max_order` serves double duty as the model's padding width.
- Break `__post_init__` into grouped, commented blocks: integer counts,
  ranges, time window, tolerances, choices. Replace the `minimum = 0 if name
  == "seed" else 1` trick with plain code. Split error messages that cover two
  conditions so each check reports exactly what failed. Name the magic
  numbers `1e-12` and `100 * eps` and explain them.
- Split `load()` into `from_dict(data)` and `load(path)`. This sets up fix 7.
- Rewrite `summary()["raw_state_bytes"]` so it reads as "total stored floats ×
  8 bytes". The value must not change.

### 5. Explain the math in `equations.py`, `analytic.py` and `validate_reference.py`
- `stable_coefficients`: add a worked example. The roots −1±2i give
  (s+1)²+4 = s²+2s+5. `np.poly` returns `[1, 2, 5]`, the slice `[:0:-1]`
  turns that into `(5, 2)`, and the result is y'' + 2y' + 5y = 0. Explain
  why `object.__setattr__` is needed in a frozen dataclass.
- `analytic.linear_reference`: name the companion matrix and show it for
  n=3. Replace the fancy-index superdiagonal on line 30 with
  `np.eye(order, k=1)`, or something equally obvious. Draw the augmented
  (n+2)×(n+2) layout for forcing in a comment.
- `validate_reference.exponential_state`, lines 40-41: the hand-rolled
  `factorial(k)/factorial(k-j)` list is just the j-th polynomial derivative.
  Replace it with `np.polynomial.polynomial.polyder(polynomial, j)` and name
  the general Leibniz product rule in the docstring.
- `benchmark_cases`: above each case, comment the equation in math form, its
  roots, and its closed-form solution. The nested tuple literals such as
  `((complex(-0.5, 1), (0.7 - 0.2j,)),)` are unreadable. Build them from named
  locals, or use a `Mode = tuple[complex, tuple]` alias plus comments. Rewrite
  the particular-solution comment on lines 95-96 clearly: B = A·e^{iφ}/P(iω),
  y_p = Im(B·e^{iωt}), and because Im(w) = Re(−i·w) while `exponential_state`
  returns real parts, the code stores −i·B. Do the same for the resonance case
  (y = −½·t·cos t solves y'' + y = sin t).

### 6. `dataset.py` simplifications
- Lines 110 and 182-183: `bisect_right` over `_ends` gives exactly the same
  answer as `divmod(index, self.time_points)`, because every trajectory has
  the same length. Use `divmod`, then delete `_ends` and the `bisect` import.
- Replace `hasattr(self, "_preloaded")` with `self._preloaded = None` in
  `__init__`, and update `tests/test_data.py:189` to match.
- Split `TrajectoryDataset.__init__` (lines 58-110) into `_load_manifest()`,
  `_load_ledger()` and `_select_records()`, each with a docstring listing what
  it verifies.
- Give the magic numbers names: cache size `8` and scale floor `1e-8`.
- `Normalizer.fit`: the loop over `j` that merges per-trajectory moments uses
  the Chan et al. parallel variance algorithm. Keep the algorithm, because it
  is robust for constant columns. Extract it into one small `_merge_moments(...)`
  helper with clear names, and say in the docstring what it computes and why
  it streams rather than holding all data in memory.
- Add `Normalizer.transform_features(features, mask)` and
  `Normalizer.inverse_target(values)`. `transform()` should use the first.
  `predict.py:30,33` should use both, so the dummy `np.zeros_like(mask)`
  target and the inline inverse go away.

### 7. Shorten `train.main` (`train.py:77-194`, about 120 lines)
Extract these pieces:
- `parse_args(argv)`, including the count validation now on lines 98-101.
  A small `positive_int` argparse type is fine.
- `seed_everything(seed)`
- `build_report(...)`
- `save_checkpoint(...)`, which keeps the atomic `.pending.pt` →
  `.replace()` pattern and gets a comment explaining it.

Replace the JSON round-trip and filtered-dict comparison on lines 128-132
with dataclass equality:
`replace(GeneratorConfig.from_dict(training.manifest["config"]), backend=config.backend, workers=config.workers) == config`.
Name the gradient-clip constant `1.0`.

### 8. `generate_data.py`
- `iter_jobs`: name the RNG stream IDs (`0` for parameters, `1` for initial
  states) as constants and explain the SeedSequence key layout. Keep the
  values identical; the golden check (item 1 of step 0) must pass.
- `iter_results`: name `max_in_flight = 2 * config.workers` and explain the
  backpressure: it bounds memory while yielding in request order. Priming the
  window with `itertools.islice` is fine.
- `generate` (lines 96-174): extract `_build_manifest(config, validation)`
  and `_record_result(result, output, manifest) -> dict`. That function saves
  the archive on success and updates the counts.
- The leakage check is duplicated in `generate_data.py:141-145` and
  `dataset.py:98-102`, with different key types. Share one tiny helper,
  placed in `config.py` beside `SPLITS`.
- The provenance code (python version, dependency versions, `source_sha256`)
  is duplicated in `generate_data.py:118-121`, `train.py:156-159` and
  `validate_reference.py:120-123`. Make it one helper and keep the output
  keys the same.

### 9. Naming pass
Rename vague local names: `x, y, valid`, `m2s`, `j`, `i`, `a`, `z`, `item`,
`definitions`, `kwargs`, `factor`, `seen, key`.

Make the field and key names agree: `SolveResult.refinement_error` and
`independent_error` are ratios and appear in the ledger as `*_error_ratio`.
Rename the dataclass fields to `refinement_ratio` and `independent_ratio`,
but do **not** change the ledger keys.

### 10. Tests
- In `test_data.py`, the ledger and manifest fixtures at lines 139-149 and
  158-172 are duplicated. Extract one `_write_fixture_dataset(root, records,
  config=None)` helper.
- Replace the literals `17` and `29` with `feature_width(...)`.
- Rewrite the class and module docstrings to say what is being verified.
  Remove the "we never solve anything" disclaimers.
- Do not lose coverage. Every existing assertion must survive, either where
  it is or moved.

## Out of scope (do not do)
- Do not change the model architecture, loss, sampler distribution,
  tolerances or file formats.
- Do not implement `evaluate.py` or `plot_results.py`.
- Do not add docs, markdown files, type-checking config, linters, or
  dependencies.

## Done means
1. All tests pass: `.venv/bin/python -m unittest discover -s ode/tests -v`.
2. The golden script from step 0 passes with no differences.
3. `python -m ode.validate_reference` passes all 13 cases. Both dry runs print
   the same JSON as before.
4. Every public function has an Args/Returns docstring with shapes.
5. No function is longer than about 40 lines, except the CLI `main()`
   argument definitions.
6. Your final report gives the changes for each file in one or two lines
   each, anything you deliberately left alone and why, and the before and
   after line count of `ode/*.py`. Do not commit.
