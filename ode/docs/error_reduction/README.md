# ODE error-reduction plan

Prepared October 16, 2026. Initial planning was completed without experiments.
The user subsequently authorized the first-priority stage: baseline validation
[diagnostics](../../runs/archive/20261016-validation-diagnostics-01/README.md) are complete,
and the matched GPU constant/cosine optimization pair is complete. Mean
validation errors are 2.276%/2.210%, but cosine worsens p95, worst case and both
forced-family means. See the [measured comparison](../../runs/archive/20261016-optimization-gpu-cosine-01/README.md).
Retain constant as the reliability control. Remaining feature,
loss and architecture comparisons are conditional. The draft manifest is not an
automatic runner or a trainer configuration. Scientific data/metric definitions
and the baseline architecture remain fixed for the optimization comparison.

The [combined cost–benefit review](COST_BENEFIT.md) incorporates the additional
options in AGENTS.md, including natural modes/decay, width 512/depth 4 and the
full preset. Its revised priority order supersedes the earlier ordering below
where stated; all settings remain proposals.

The aim is to improve accuracy across the existing order-2/order-3,
homogeneous/sine-forced domain while measuring the cost. We need to reduce the
error tail and derivative errors as well as the pooled mean. The previous 5%
mean target established a functioning baseline; it is not a scientific standard
for a high-accuracy solver.

## What the existing evidence says

Only saved training logs and validation/diagnostic records were analyzed for
this plan. No checkpoint was loaded or used for new inference. The previously
reported test result is historical evidence; it is not used to choose experiments.
The compact, source-hashed review is in [baseline_review.json](baseline_review.json).

| Observation from validation | Interpretation and uncertainty |
|---|---|
| Mean error improved from 5.63% at epoch 50 to 4.17% at 75 and 3.58% at 96; only 100 epochs ran | More optimization is a plausible cheap first experiment. Convergence is not established. |
| Original AdamW used constant 1e-3; no scheduler existed during preparation | The subsequently measured controls above test longer optimization and rate decay. |
| Unforced family means are 2.23% and 2.20%; forced means are 4.79% and 5.09% | Forcing representation and interactions deserve attention. This does not prove a particular architecture is responsible. |
| Third-order forced component means: y 4.54%, y' 5.73%, y'' 6.52% | Improving position alone could hide poor derivative prediction. |
| Validation p95 is 8.98%, worst 26.23%, and 208/1,024 cases exceed or equal 5% | A lower mean alone is insufficient for the next engineering goal. |
| Parameter-only diagnostic mean is 3.99%; IC-only mean is 2.94% | Parameter generalization appears harder in these diagnostic cohorts; this is not a controlled causal comparison. |
| At epoch 96, training standardized MSE is 0.000565 versus validation 0.002189 | There is a generalization gap. Training loss is collected during updates, so this is not an exact train/eval-mode comparison or proof of overfitting. |
| No validation trajectory reaches the 1e-6 near-zero floor | Adjusting that floor cannot fix the measured validation errors. |

Some large relative errors involve small reference amplitudes, but the largest
validation error also has absolute RMSE 0.1353 at reference RMS 0.5159. There are
substantial physical-state errors as well as small-denominator sensitivity.

## Proposed success criteria

Use the existing metric definition and domain throughout comparisons. As a
**provisional next engineering goal**, seek mean relative L2 below 1% overall
and within each order/forcing family, p95 below 3%, fewer than 1% of cases at or
above 5%, and zero nonfinite predictions. Retain absolute errors, component
breakdowns and every worst case; investigate every case above 10%.

These are proposed goals, not measured outcomes, established scientific
standards, or changes to the historical acceptance test. Smaller errors may
still be necessary for a particular application. Runtime, derivative consistency
and comparison with classical methods determine whether the surrogate is useful.

## Staged experiment order

Every comparison changes one factor against an explicit matched parent. Screen
one training seed, then confirm only the selected method against its control
using three optimization seeds on the same data. Do not launch the whole table
as a grid. Settings in [experiments.json](experiments.json) are drafts.

| Order | Comparison | Why / what it resolves | Preparation still needed |
|---|---|---|---|
| 0 | Diagnose time-local and physical-consistency errors | Locate errors at initial time, early/late times, high frequency, low damping, and forcing near a natural frequency | Implemented and measured on baseline train/validation |
| 1 | 300-epoch 3x128 constant-LR control versus the same model with cosine LR decay | Determine how much error is optimization-limited before changing architecture | Implemented and measured on local AMD GPU; no broad scheduler win |
| 2 | Forcing phase, then natural-frequency/decay features | Expose the known forced and homogeneous time dependence | Versioned encoding, mode recovery and checkpoint/inference support |
| 3 | Physical relative-squared loss, followed conditionally by full-trajectory relative loss | Align the objective while first preserving the existing batch layout | Training-only RMS weights; full-trajectory sampler only for the later variant |
| 4 | 3x256 versus 3x128, followed conditionally by width512/depth4 | Separate width and depth effects and measure the cost | Existing flags; matched one-factor controls |
| 5 | Add a modest Fourier time encoding | Test whether an oscillatory representation improves remaining time-dependent errors | Versioned features; conditional on earlier results |
| 6 | Enforce the full initial state exactly | Eliminate initial-state violations by construction | New output transform; verify early-time gradients and derivative consistency |
| 7 | Exploit linear superposition in the architecture | Use the known structure of these linear equations to simplify generalization | Separate research branch and mathematical/data design; higher effort |

The order is a proposal informed by the review. The older decision register's
width-then-Fourier fallback remains historical; no new architecture is adopted by
this planning exercise. If an earlier change achieves the provisional goals,
stop adding complexity and move to reproducibility and final evaluation.

### 0. Diagnose before attributing causes

After execution is authorized, evaluate only training/validation material to
measure per-component error over time, initial-state error, and residuals
`d(s_hat)/dt - rhs(t, s_hat)`. Here `s_hat` is the predicted full state and `rhs`
is the first-order form of the ODE. In particular, compare the time derivative
of the predicted y with the separately predicted y'. Independent output heads
currently have no constraint making those quantities agree.

Slice validation results by natural/forcing frequency, damping, reference
amplitude, and the relative location of forcing and natural frequencies. Keep
those slices descriptive until enough cases support a conclusion. Define any
future extra challenge cases before seeing their results. These diagnostics are
implemented and measured in `ode/runs/archive/20261016-validation-diagnostics-01/`.
The original preparation reviewed saved aggregate records only.

### 1. Give optimization a fair comparison

Run both constant-LR and scheduled-LR models from scratch for 300 epochs, with
identical seed, batches, data and architecture. Propose cosine decay from 1e-3
to 1e-5 over the run. Select checkpoints using validation mean relative L2.
The constant control separates the benefit of more updates from the scheduler.
Record p95, family/component means and failure rates at selected checkpoints.

The best-inference checkpoint stores weights and preprocessing but no AdamW moments,
shuffle/RNG state or scheduler state. Loading it with a fresh optimizer is a
warm start, not an exact continuation. For the initial comparison, fresh runs
avoid that ambiguity. The authorized implementation now saves a separate
`last.pt` with those states; a resume CLI is still future work. The existing
best-inference checkpoint contract is preserved.

The authorized implementation also exposes/records weight decay and an
independent model/shuffle seed. Both GPU controls retain baseline decay 0.01
and training seed 20261015 while keeping scientific data seed unchanged.

### Loss choices: preserve a cheap comparison before changing batches

The current loss averages squared errors after each component is divided by
its training standard deviation. The reported metric instead divides a whole
trajectory's physical-state RMS error by that trajectory's reference RMS.
Checkpoint selection uses the correct metric, but it does not change gradients.

The combined review adds a less invasive first comparison: physical squared
error weighted by `1/max(r_i,epsilon)^2`, using the existing shuffled time-point
batches. It estimates mean **squared** relative trajectory error, not mean
relative L2. A weight of `1/r_i` on MSE does not implement either metric. See
[COST_BENEFIT.md](COST_BENEFIT.md#correct-the-loss-weighting-shorthand) for the
unit/scaling details. The full-trajectory approach below remains optional.

For trajectory i, let `e_i` be physical-state RMSE over all stored times and
active components and `r_i` the reference RMS. Compare the baseline with a
trajectory-aware objective based on `e_i/max(r_i, 1e-6)`. A smoothed square root
near exact zero is needed for stable gradients and must be documented. A squared
relative loss emphasizes outliers more strongly and is a separate variant.

A loss-only comparison needs a matched batch control: use ten complete
trajectories per step (1,010 time points) for both standardized MSE and the new
relative objective. This separates loss weighting from the change in batch
correlation/update count. The old 1,024-point shuffled baseline alone is not a
matched control for a trajectory-batched model.

Use training trajectories only to calculate any sampling weights or weight caps;
never infer them from validation/test statistics. Full trajectories, or an
explicitly documented unbiased sampling scheme, must retain equal trajectory
weights. If stabilized/capped weights are needed near zero, document that the
training loss is a surrogate while evaluation stays frozen. Retain absolute
error monitoring so a relative objective does not neglect large-amplitude cases.

### Feature choices: forcing, natural modes, then generic Fourier features

Keep raw elapsed time and all original descriptors. First add two deterministic
features, `A*sin(omega*t+phi)` and `A*cos(omega*t+phi)`, using the equation inputs
only; set both to zero for homogeneous equations. No reference state is an input.
Compare this alone, then separately add the natural damped modes
`exp(-d*t)*cos(w*t)`, `exp(-d*t)*sin(w*t)` and (order3) `exp(-r*t)`, derived from
physical coefficients. The combined review specifies the root/mask/encoding
requirements. Compare these before generic Fourier features.

If useful headroom remains, append `sin(b*t), cos(b*t)` with a small fixed set of
angular frequencies, initially b in [0.5, 1, 2, 4] radians per nondimensional time
unit. Those are draft settings based on the frozen input range, not a proven
optimal bandwidth. Retain raw time to distinguish nonperiodic decay. Increasing
bandwidth can add complexity without improving this problem.

Fourier mappings have helped MLPs fit oscillatory detail in low-dimensional
regression; transfer to this equation-conditioned surrogate is a hypothesis.
[Tancik et al., NeurIPS 2020](https://papers.neurips.cc/paper_files/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html).

### 6. Satisfy initial conditions by construction

One candidate predicts the full state as
`s_hat(t) = s0 + (t/T)*N_theta(inputs,t)`, with T=5. The physical initial state
`s0=[y(0),y'(0),...]` is then exact at time zero. Apply this in physical state
units after undoing normalization; masks still remove inactive derivatives.

This construction does not automatically enforce `d(y_hat)/dt = y_prime_hat`
or the ODE. Its gradient shrinks near time zero, so check early-time accuracy
and compare with a matched unconstrained model before keeping it. Constructed
trial solutions that preserve initial/boundary conditions have precedent, but
this particular shared-model parameterization remains untested here.
[Lagaris, Likas and Fotiadis](https://arxiv.org/abs/physics/9705023).

### 7. Use the strongest known structure if simpler changes fall short

For fixed equation coefficients, the solution is linear in the initial state
and the two forcing amplitudes `A*cos(phi)` and `A*sin(phi)`. A possible model is

```text
s_hat(t) = Phi_theta(a,t)*s0
         + R_sin_theta(a,omega,t)*A*cos(phi)
         + R_cos_theta(a,omega,t)*A*sin(phi)
```

`Phi` is an n-by-n matrix mapping initial state to future state; each R is an
n-component response to a unit sine/cosine input. Exact constraints are
`Phi(a,0)=I`, `R_sin(...,0)=R_cos(...,0)=0`. The learned functions would handle
coefficient/time dependence while linearity in initial conditions and forcing
is enforced. This is an architecture tailored to the current linear families,
not evidence of general nonlinear ODE learning.

Each existing parameter group has only one forcing amplitude/phase; separately
identifying both forcing-response columns may require unit-basis reference
trajectories or another explicit training design. Do not assume the present data
fully identifies this decomposition. Keep it a separate versioned study.

A small physics-residual penalty is another later option if diagnostics find
inconsistent derivatives. Correctly differentiate physical time and undo target
normalization; balance residual and data terms carefully. Physics penalties can
make optimization harder in some systems, so they are not an automatic accuracy
upgrade. [Krishnapriyan et al., NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/df438e5206f31600e6ae4af72f2725f1-Abstract.html).

## Data, measurement and final-test policy

Reuse the pilot's numerical data for the first comparisons. More data and more
capacity address different limitations. If a training-group learning curve shows
continued validation improvement, compare 25%, 50% and 100% of training parameter
groups, keeping every selected group's IC trajectories together and fitting
normalization on that subset only. Increase training groups in a bounded step
only after that evidence; no automatic move to 100k is proposed.

All family bounds, forcing policy, horizon, reference checks, metric scaling and
near-zero denominator remain fixed. Sampling difficult regimes is a later
versioned training change; retain the original validation distribution and
report a separate challenge set. Reducing scope is an explicit deadline fallback
with a revised claim, not an improvement on the same task.

The old pilot test has already been examined. Future candidate selection uses
validation/diagnostics, followed by a fresh locked final cohort sampled after
selection with independent seed/identity checks. The current evaluator requires
the dataset manifest to match the checkpoint, so external-holdout evaluation
needs an explicit implementation: preserve the training manifest, validate
compatible scientific recipe, verify disjoint parameter/IC identities and record
the evaluation manifest separately. Do not weaken identity checks globally.

For repeated optimization seeds, pair candidate/control seeds while leaving
reference-data seed and split membership unchanged. Parameter groups share IC
trajectories, so uncertainty estimates should resample parameter groups within
family rather than treating all trajectories as independent. Three seeds are
an initial robustness check, not proof across all possible datasets.

Compare runtime and error with DOP853 and the independent matrix-exponential
reference on the same equations, requested outputs, precision and batch sizes.
Record startup/load costs separately from repeated loaded inference. A GPU can
make an experiment faster; accuracy changes still require controlled validation.

## Implementation preparation and resource budget

The initial gate is two fresh 300-epoch optimization runs. At the measured
3.746 seconds per baseline epoch, each projects to about 18.7 CPU minutes;
together about 37.5 minutes plus setup. This is a local planning estimate, not a
GPU reservation or a promise. Wider/trajectory-batched/derivative models require
their own measurements. Decide each later experiment after the preceding result.

Use existing editable resource profiles. Pointwise controls use batch size
1024; trajectory-loss controls use ten trajectories (1,010 points), and later
comparisons inherit their selected parent's batch policy. The CUDA example
defaults to 4096 points, so override it for matched scientific comparisons. Record device/build, threads, precision and all
resolved options. Allocation-specific throughput tuning is a separate study.

Before an experiment phase, implement only its missing capability and document
it. Required behavioral checks include: scheduler endpoints and resume state;
independent optimization seed; hand-computable trajectory loss; deterministic
features with no target leakage; new-encoding checkpoint reload; exact initial
state in physical units; physical-time derivative scaling; and external cohort
identity/normalizer isolation. Scheduler/state, independent seed and derivative
scaling checks now pass; checks for unimplemented later stages remain pending.

`experiments.json` is a proposal manifest, not accepted by today's trainer. The
constant/cosine controls and independent seeds are now implemented and measured;
new losses/features and external-holdout evaluation still need implementation.
The original preparation changed no production runtime code; subsequent
authorized implementation is tracked separately in the decision register.
