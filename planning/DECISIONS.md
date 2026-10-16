# Decision register

## Established constraints

- October ODE delivery is the critical path; fluid implementation follows it.
- Use supervised PyTorch trajectory prediction for the initial ODE baseline.
- Keep new work in root-level track folders and preserve the experimental archive.
- The initial scaffold has progressed to a documented ODE framework. October 16 authorizes bounded real generation, timing and training runs.

## October 15 implementation decisions

- Represent scalar higher-order ODEs as states ordered `[y, y', ..., y^(n-1)]`. The solver path accepts arbitrary positive orders; scientific validation is still required for each supported regime.
- Begin with normalized constant-coefficient linear equations, with optional sinusoidal forcing. Sample decaying real/complex-conjugate modes to obtain stable unforced coefficient sets. This does not imply coverage of arbitrary nonlinear equations.
- Use a shared configurable masked-order MLP as the first framework baseline, superseding the earlier proposal for separate per-family models. `max_order` fixes encoding/checkpoint size; supported input orders do not establish learned order generalization.
- Use a deterministic seeded sampler and bounded threaded execution, with serial and spawned-process alternatives. Keep file writing in the coordinator. No throughput benchmark has run.
- Configure the example generator for 12 spawned process workers on the verified Ryzen 5 5600 host (6 cores, 12 logical CPUs), allowing Python-heavy callbacks to use multiple cores. Thread and serial modes remain available; optimal worker count and throughput are unmeasured.
- Default to DOP853 with a second tighter-tolerance solve. Generation now requires closed-form preflight for both configured families and per-trajectory independent matrix-exponential checks; refinement agreement alone is insufficient.
- Use joint parameter/initial-condition holdout partitions now. Separate parameter-only and initial-condition-only diagnostic sets remain planned.
- Reuse the existing isolated `.venv`; add SciPy while retaining the installed NumPy and PyTorch stack. Default training to CPU and record the tested environment.
- Add team documentation rules to `AGENTS.md` and explain all scaffold modules, math symbols, shapes and limitations for both collaborators.
- Verify both homogeneous and sine-forced references on fixed closed-form cases, including real/repeated roots and resonance; this does not extend the random sampler's root distribution. Save the bounded validation evidence under `ode/runs/20261015-reference-validation-01/`.
- Add explicit CPU/CUDA/auto device selection and host-RAM data preloading. Preserve CPU defaults and the existing local environment. NVIDIA execution requires a compatible CUDA build on the GPU host and remains untested here.
- Treat the stated 20,000+ cores/600+ nodes as shared cluster capacity; estimate one 32-core node until the actual allocation is known. Target GPU is RTX PRO 6000 Blackwell. Keep runtime projections separate from measured local reference timings.
- The user explicitly chose code updates and validation only for this task: do not generate the 384-trajectory dataset or launch training.
- Provide a separate 100,032-trajectory full-run preset using groups `[4168, 1042, 1042]`, preserving the 2/3, 1/6, 1/6 split and all existing scientific ranges. Keep the 384-trajectory example available. Document local/HPC generation and matching-config training; launching the full run is left to the user.

## October 15 proposals (resolved October 16 below)

| Decision | Proposed starting point | Must resolve before reference generation/release |
|---|---|---|
| Second-order family | Damped linear oscillator | Forcing policy, coefficient bounds, initial-condition bounds |
| Third-order family | Stable linear constant-coefficient system | Parameterization, stability constraints, coefficient bounds |
| Model arrangement | Shared masked-order MLP framework | Real-data adequacy versus family-specific baselines |
| Prediction mode | Initial conditions and elapsed time to full state | Time horizon, sampling density, extrapolation policy |
| Numerical reference | Configurable adaptive integration plus refinement | Analytic checks and scientific validation of draft tolerances |
| Dataset | Complete trajectories grouped by configuration | Counts, seeds, storage format and partition proportions |
| Accuracy | Per-trajectory relative L2 with aggregate reporting | Primary output, scaling, denominator floor and near-zero reporting |
| Compute budget | Small reproducible baseline first | Available hardware, maximum runtime and initial dataset size |

This table records historical draft choices; the resolved v1 specification below supersedes it. Record accepted choices with date, rationale, alternatives considered and the evidence that would justify revisiting them. Implementation decisions above can be revised when measured evidence warrants it.

## October 16 resolved scientific contract (v1)

The current request explicitly authorizes resolving the freeze and running small
real-data experiments; it supersedes the October 15 generation/training deferral.
The following scientific choices are accepted before this session's generation.
Historical proposals above are retained as history, not current instructions.

- **Families:** second-order damped oscillator and third-order stable linear
  system. Both have homogeneous and sinusoidal variants, equally represented.
  Write `y^(n) + sum(a_k y^(k), k=0..n-1) = A sin(omega*t + phi)`.
  State and coefficient order is ascending derivative order. All quantities are
  nondimensional; a single time unit and state unit define the derivative units.
- **Bounds:** independently sample damping `d` and (for order 3) `r` uniformly
  in `[0.1,1]`, mode frequency `w` in `[0.5,2]`. Order 2 has roots `-d +/- iw`;
  order 3 adds `-r`. Thus order 2 coefficients are `(d*d+w*w, 2*d)`, bounded
  by `a0 in [0.26,5]`, `a1 in [0.2,2]`. Order 3 coefficients are
  `(r*(d*d+w*w), d*d+w*w+2*d*r, 2*d+r)`, bounded by `[0.026,5]`, `[0.28,7]`,
  `[0.3,3]` respectively. These are correlated induced bounds, not independent
  coefficient sampling. Each initial derivative is uniform in `[-1,1]`.
  Homogeneous forcing is exactly zero. Forced cases sample `A in [0.1,1]`,
  `omega in [0.5,2]`, `phi in [-pi,pi]`, independently. Stability follows from
  negative real roots; forced resonance and transient amplification still occur.
- **Time:** 101 uniformly spaced points on `[0,5]`, including endpoints;
  `dt=0.05`. Inputs use elapsed time. Training-bound and horizon extrapolation
  are outside October acceptance. No out-of-range dataset is required for v1;
  such a future study needs a separate version and must not join release metrics.
- **Reference:** retain DOP853 `rtol=1e-9`, `atol=1e-11`, a second solve with
  ten times tighter tolerances, and five matrix-exponential checks. Acceptance
  tolerances remain `1e-6` relative and `1e-8` absolute. Closed-form preflight
  is mandatory. Prior 13-case evidence supports these choices; every new solve
  must still pass both checks. Never remove validation to accelerate generation.
- **Primary metric:** per-trajectory full-state `RMS(pred-ref)/max(RMS(ref),1e-6)`.
  Use only real derivatives, unit component scales, no centering, uniform time
  weights including endpoints. This equals relative L2 except near zero; its
  L2 denominator floor is `1e-6*sqrt(T*n)`. Aggregate with equal trajectory
  weights. Report mean, median, p95, worst, errors >=5%, nonfinite failures,
  and identical per-component and order/forcing-family breakdowns. Also report
  absolute RMSE and maximum error, especially for reference RMS <=1e-6.
  Standardized MSE remains the optimization loss; checkpoint selection uses
  validation mean relative L2. Test target is strictly below 5%; nonfinite
  outputs preclude passing. Report per-family outcomes even if pooled mean passes.
- **Splits:** primary train/validation/test remain disjoint in parameters and
  initial conditions, in proportions 2/3, 1/6, 1/6. No time-point splitting.
  Optional `validation_parameter` crosses validation parameters with training
  ICs; `validation_initial` crosses training parameters with validation ICs.
  Those diagnostics intentionally share validation identities and are never
  described as independent test sets. No test identity participates. Loader
  checks enforce these rules; only core training contributes to normalization.
  Keep the 384-case smoke recipe unchanged; enable diagnostics for the pilot.
- **Architecture decision and escalation:** start with the existing shared
  3x128 SiLU MLP, raw time, four order/forcing strata. A smoke run checks wiring,
  not achievable accuracy. On a larger pilot, allow 100 baseline epochs and
  choose the best validation-relative-L2 checkpoint. If mean remains >=5%,
  compare 3x256 under the same recipe; then compare a separately versioned
  Fourier time encoding, retaining raw time and matching the compute budget.
  Use only validation/diagnostics to select changes. If the October 23 baseline
  deadline is threatened, first remove sine forcing while retaining orders 2
  and 3, with an explicit new dataset recipe and decision. Never silently narrow
  a claimed domain or use the test split for model selection.
- **Compute/data size:** 384 trajectories are authorized for smoke and timing.
  The 100,032 preset is a capacity option, not a selected experiment. Choose
  the first pilot size after measured generation/training timing below. Actual
  cluster allocation remains unknown; a local 12-thread CPU is not an allocation.
- **Fluid:** leave implementation placeholders unchanged until November 2. Their
  existence is not evidence of a working fluid solver or model.


### October 16 measured size/worker decision

Three repeated full 384-trajectory generations (including preflight and I/O)
gave median wall times 4.039s serial, 1.174s with six processes, 1.448s with
12 processes. Native BLAS/OpenMP threads were one. Keep all reference checks;
refinement dominates validation cost, while five matrix exponentials were
about 1% of serial numerical-stage time. Select **six local workers**.
The 100,032 preset stays unused; its worker field is a local fallback and must
be overridden with measured allocation-appropriate settings on the cluster.

Select the **6,656-trajectory initial pilot**: 4,096 train, 1,024 validation,
1,024 reserved test, 256 parameter-only and 256 IC-only validation diagnostics.
This is 16 times the smoke core population and roughly 13.4 MB of raw state
arrays; its training preload is about 58 MB for 4,096*101*35 float32 values.
A simple six-worker smoke-rate projection is about 20s generation, explicitly
not an HPC estimate. Verify actual pilot timing and keep its test set unused
while assessing the 100-epoch baseline. Expansion toward 100k requires a
validation learning-curve benefit and a measured target compute budget.


### October 16 pilot milestone evaluation and portability

The user is obtaining allocation details in a meeting and requested flexible
resource settings. Add independent JSON execution profiles with explicit CLI
overrides; CPU auto detection uses affinity and SLURM task allocation, never an
assumed 32-core node. A preload array-memory cap is configurable. Scientific
sampling settings remain unchanged when moving between machines. GPU settings
are an editable example until exercised on actual hardware.

The planned 100-epoch 3x128 pilot completed in 374.6s.
Select epoch 96 solely by validation relative L2
(3.578%). Before reading test results, freeze
that checkpoint for one pilot milestone evaluation of the reserved 1,024-case
joint test set, plus the validation diagnostics. This answers the requested
baseline adequacy check. No further model tuning will use those test results;
subsequent model development requires a newly reserved test cohort or a clearly
documented final reuse policy. This is pilot evidence, not an October release.


### October 16 pilot outcome

One frozen-checkpoint test evaluation returned mean 3.689%, median 2.704%, p95
8.976%, worst 36.908%, 216/1,024 cases >=5%, and zero numerical prediction
failures. Third-order sine forcing averaged 5.573%, so pooled acceptance is not
per-family acceptance. Validation diagnostics were 3.993% for parameters and
2.938% for ICs. Keep the unchanged 3x128 baseline and pilot size; do not launch
100k or change architecture just to satisfy an untested capacity assumption.
Follow-up reliability studies use validation and an explicit future test plan.
Checkpoint/runtime/plot evidence is in `ode/runs/20261016-pilot-baseline-01/`.
Actual allocation and NVIDIA execution remain unresolved pending the user's
meeting; no further request for those details is needed during this session.

## October 16 error-reduction proposals — preparation only

The user requested a higher-level review, a plan and preparation, explicitly
excluding execution. This supersedes earlier run authorization for this new
task. Existing results remain historical evidence; no new generation, training
or model evaluation was performed to prepare the plan.

See [the staged plan](../ode/error_reduction/README.md), its saved-validation
review and draft experiment manifest. Proposed order: diagnose physical/time
errors; compare longer constant-rate and scheduled optimization; compare losses
with matched trajectory batches; add forcing-phase features; compare capacity;
then consider Fourier features, exact initial conditions and a separate linear
superposition architecture. These are hypotheses and comparisons, not adopted
model changes or measured improvements. Each stage requires its own control.

The provisional next engineering goals are <1% mean overall and within each
family, p95 <3%, <1% of trajectories at or above 5%, and no numerical prediction
failures. These goals neither alter the frozen metric nor retroactively change
the historical release target. They are not application-specific guarantees.

Preserve the current domain and validated references while testing improvements.
Use validation for selection; the previously inspected pilot test must not become
a repeated tuning target. Plan a fresh disjoint final cohort and explicit
external-evaluation identity checks. Resource profiles remain editable, with
matched batch policies across devices for scientific comparisons.

Prepared files do not implement schedulers, new objectives/features, independent
optimization seeds or external test evaluation. No architecture or dependency
has been changed by this planning task. Actual experimentation remains deferred
until the user requests it.

## October 16 combined options and cost–benefit review — provisional

Reviewed the new parked accuracy options in root AGENTS.md and incorporated all
six into [the combined review](../ode/error_reduction/COST_BENEFIT.md). Added
natural-mode/decay feature settings, width/depth controls through 512x4, a cheaper
pointwise relative-squared loss and conditional group-learning-curve/full-data
comparisons to the draft manifest. All comparisons remain unexecuted.

The proposed sequence now favors diagnostics, matched longer/cosine optimization,
forcing and natural-mode features, correctly weighted physical-state loss, then
width 256. Higher width/depth, 100k data, exact-IC/physics alternatives and a
linearity-based model remain conditional. This updates a proposal order only;
no architecture, objective, scientific range or release gate was adopted.

Analytical costs: the full preset contains 16.28125x the pilot training population.
Simple local scaling projects about 1.69 hours for 100 epochs or 5.08 hours for
300. A 3x256 model has about 3.80x the baseline dense forward arithmetic; 4x512
about 21.96x. These are not measured candidate wall times or GPU estimates.
Parameter counts, formulas and artifact hashes are retained in cost_benefit.json.
No expected error percentage or dollar saving is asserted.

Clarify AGENTS.md's weighting shorthand: inverse RMS weights an RMSE-like loss;
mean squared relative error requires inverse RMS squared on physical squared
errors. Weighting standardized MSE without undoing component scaling still
optimizes a different objective. Full-trajectory relative RMS remains a separate
comparison with a matched sampler control.

Preserve validation cohorts across data-size comparisons, independent final-test
identity checks, and the user instruction to prepare without executing. The
analysis did not modify AGENTS.md, runtime code, existing datasets/checkpoints,
the virtual environment or the fluid track.
