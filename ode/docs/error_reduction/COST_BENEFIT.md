# Combined accuracy options: cost–benefit review

Prepared October 16, 2026 from the current AGENTS.md, existing source/configs and
saved pilot/validation records. **No generation, training, inference or new model
evaluation was run.** Numerical estimates below are arithmetic projections,
not measured candidate performance. Expected accuracy gains remain unknown.

Recommendation: diagnose, improve optimization, add forcing/natural-mode features,
then test a correctly weighted objective and modest capacity. Reserve full-data
scaling and a new response-operator architecture for evidence-based follow-up.

## Coverage of AGENTS.md and the earlier plan

| AGENTS.md option | Combined treatment |
|---|---|
| Full 100,032-case preset | Included as a conditional scaling study; compare fixed updates and fixed epochs separately |
| Forcing/natural sin/cos and exponential decay features | Extend the earlier forcing-phase proposal with an isolated natural-mode feature comparison |
| Inverse-reference weighting, cosine decay, longer training | Separate optimization and loss comparisons; correct the weighting formula for squared error |
| Width 256–512 and depth 4 | Width-only and depth-only controls, with exact parameter/arithmetic costs below |
| Linearity through basis responses | Retain as a higher-effort research branch, including forcing phase and identifiability checks |
| Diagnose worst cases first | Use validation and training evidence; previously generated test plots are historical, not tuning material |

The earlier plan also contributes exact initial-state enforcement, generic
Fourier features, physical derivative diagnostics, repeat-seed confirmation and
fresh final-test safeguards. None is assumed to improve accuracy until measured.

## What counts as a cost or a benefit

Implementation cost includes changes to data encoding, normalization, checkpoint
compatibility, inference and meaningful verification. Low effort means existing
settings or a localized change; medium means several pipeline contracts; high
means a new model/data design. These are relative planning judgments, not labor
quotes. Calendar cost matters for a two-person team approaching the October
release and the November fluid work.

Compute cost includes training and validation, generation, preprocessing and
inference. Parameter/arithmetic counts are exact for the stated architecture;
wall-time and accuracy effects are unmeasured. Hardware access, queue time,
allocation prices and billing rules are unknown, so no dollar costs are invented.
Once those are known, apply site rates to allocated node/GPU/core hours as the
site defines them. Extra hardware accelerates experiments; it does not by itself
establish lower error.

Benefit means lower frozen relative-L2 mean **and** improved tails/family/component
behavior, plus acceptable runtime and reproducibility. Diagnostic work has high
information value but does not directly lower prediction error.

## Relative return and recommended priority

| Option | Plausible benefit / evidence | Implementation cost | Compute/inference cost | Main uncertainty | Priority |
|---|---|---|---|---|---|
| Validation failure and derivative diagnostics | Identifies time/regime/consistency problems; forced families are about twice as inaccurate | Low for saved records; medium for new derivatives/time-local reports | Small diagnostic workload; not measured | Attribution is still a hypothesis | First |
| Longer training + cosine schedule | Best checkpoint was epoch 96/100; improvement had not clearly saturated | Low–medium; scheduler and seed/state logging | Two 300-epoch controls project to ~37.5 local CPU minutes; inference unchanged | Extra optimization may plateau or overfit | High |
| Forcing-phase features | Directly exposes a known periodic input to the network | Medium: encoding, normalizer, model/checkpoint and inference contracts | Two extra features add ~0.7% dense forward MACs; trig/preprocessing overhead unmeasured | Does not automatically describe accumulated system response | High |
| Natural-frequency/decay features | Exposes the actual homogeneous time dependence of these families | Medium: recover stable modes and version features | Three damped-mode features add ~1.1% dense forward MACs; exp/trig/root costs unmeasured | ODE-specific prior, root handling and coefficient interpolation still matter | High |
| Pointwise relative-squared weighting | Better matches equal relative importance of trajectories, especially small-amplitude responses | Medium; less invasive than full-trajectory batching | One scalar per trajectory; loss/preprocessing overhead unmeasured; inference unchanged | May emphasize noise/outliers; squared relative loss is not mean relative L2 | High after representation |
| Full-trajectory relative loss | More directly matches the reported trajectory objective | Medium–high: sampler, physical units, smoothing and matched batch control | Different batch correlations/update count; measure overhead | Can optimize tails poorly or become unstable near small norms | Conditional |
| Width 256, depth 3 | Tests whether modest extra capacity helps difficult parameter interactions | Low with existing CLI | ~3.80x dense forward arithmetic; wall time need not scale proportionally | Existing train/validation gap means larger capacity alone may not generalize better | Medium |
| Depth 4 / width 512 | More expressiveness if smaller model remains capacity-limited | Low to configure; higher tuning/compute cost | 256x4: ~5.59x arithmetic; 512x4: ~21.96x | Diminishing returns and additional tuning | Later |
| Exact initial state | Guarantees an important input constraint at t=0 | Medium: physical output transform and compatibility checks | Small transform overhead; early-time gradient effects unmeasured | Does not guarantee correct derivatives or lower full-trajectory error | Conditional on diagnostic evidence |
| Linear superposition model | Enforces exact dependence on initial state and forcing amplitudes | High: response heads, constraints and training identifiability | New output workload and possible basis data; benchmark required | Highest structural potential here, narrower transfer to nonlinear fluid dynamics | Later research branch |
| Generic Fourier time features | Alternative oscillatory representation | Medium; shares feature plumbing | Eight extra features for four frequencies add ~2.8% dense MACs | Redundant or less useful than equation-conditioned features | Fallback |
| 100k dataset | Denser parameter coverage; parameter holdout currently harder than IC holdout | Config exists; fair-cohort/subset comparison still needs work | 16.28x training trajectories; ~1.69 h/100 epochs or ~5.08 h/300 under simple local scaling | Benefit unmeasured; extra data and extra updates are confounded at equal epochs | Conditional on learning curve |
| Physics-residual penalty | Could improve inconsistent independently predicted derivatives | High: differentiable physical-time residual and loss balancing | Derivative/backprop work unmeasured | Optimization can become harder; use only if diagnostics support it | Later |

All benefit judgments are hypotheses. This table ranks the value of a controlled
experiment, not a predicted percentage-point gain.

## Network arithmetic: why "larger" has very different costs

For input width d=23, hidden width w, h hidden layers and output width m=6:

```text
forward dense MACs per point = d*w + (h-1)*w*w + w*m
parameters = d*w + w + (h-1)*(w*w+w) + w*m + m
```

A MAC is one multiply-and-accumulate operation. Counts exclude activations,
normalization, loss, gradient/optimizer operations and data handling. A dense
backward pass adds more work, so these ratios describe the forward network,
not exact training or inference wall time.

| Hidden layers x width | Parameters | Forward MACs/point | Arithmetic relative to current model |
|---|---:|---:|---:|
| 3x128 | 36,870 | 36,480 | 1.00x |
| 3x256 | 139,270 | 138,496 | 3.80x |
| 4x256 | 205,062 | 204,032 | 5.59x |
| 3x512 | 540,678 | 539,136 | 14.78x |
| 4x512 | 803,334 | 801,280 | 21.96x |

Test 3x256 against 3x128 before widening to 512. Compare 4x256 against 3x256
before attributing gains to depth. Each uses the same selected loss, schedule,
training cohort, update policy, precision and seed. A faster GPU may change the
wall-time trade-off substantially, so measure rather than multiply these counts
by the CPU latency.

## Feature design: add what the equation already tells us

Forcing features should preserve phase: use `A*sin(omega*t+phi)` and
`A*cos(omega*t+phi)`, with zero forcing features when A=0. Keeping original
parameters and elapsed time avoids discarding information about the response.

For roots `-d +/- i*w`, propose the homogeneous feature pair
`exp(-d*t)*cos(w*t)` and `exp(-d*t)*sin(w*t)`. For third order, add the real-mode
term `exp(-r*t)`; the third slot is zero/masked for second order. The existing
family has one complex pair, plus that real root for order 3. These terms expose
the damped modal shape directly, rather than requiring the MLP to multiply
separate decay and oscillation features. The mathematics follows the
[constant-coefficient solution structure](https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/pages/unit-ii-second-order-constant-coefficient-linear-equations/).
Their benefit for this learned parameterized map is our inference and remains
untested.

Recover d, w and r from the physical equation coefficients, not from reference
states or hidden validation metadata. For order 2, d=a1/2 and
w=sqrt(a0-d*d); for order 3, classify the polynomial's real and conjugate roots.
Define numerical root tolerances and unsupported cases. Cache roots per equation
when appropriate, and include cold/warm preprocessing in later runtime reports.

Compare forcing features alone, then add damped-mode features as a separate
step. Each deterministic feature leaves references valid, but changes saved
input/preprocessing dimensions and requires an explicitly versioned checkpoint.

## Correct the loss-weighting shorthand

AGENTS.md suggests weighting loss by `1/reference RMS`. That depends on what
"loss" means. If e is physical-state trajectory RMSE and r its reference RMS:

```text
relative L2                 = e / max(r, epsilon)
squared relative L2         = e^2 / max(r, epsilon)^2
```

Multiplying an MSE by `1/r` is neither of those. Also, weighting standardized MSE
by `1/r^2` still retains unwanted per-component standard-deviation weights.

A cheaper first loss experiment keeps today's shuffled time-point batches:
undo target scaling on errors, then weight each sample's mean active-component
squared physical error by `1/max(r_i, epsilon)^2`. Cache r_i from **training
references only**. Because all trajectories have the same T, uniform time-point
sampling estimates the mean squared relative trajectory error. This preserves
batch layout and provides a cleaner low-cost comparison with the original MSE.
Use a documented training-only stabilization rule if required; leave evaluation
unchanged. One float64 weight per training trajectory uses ~32 KiB in the pilot.

Full-trajectory mean relative RMS remains a second experiment if needed. It
requires smoothing near zero and a matched full-trajectory MSE control; otherwise
loss choice and batch layout change together. Keep absolute-error reporting in
both variants. Changing the evaluation denominator to lower reported errors is
outside this plan.

## The 100k option: affordable generation, uncertain learning value

The preset has 66,688 training trajectories versus 4,096 in the pilot: **16.28125x**.
Its 100,032 core trajectories exclude the pilot's separate diagnostics; it is not
a drop-in replacement for the full diagnostic protocol.

Simple local projections, with the same 3x128 model and batch 1024:

| Work | Projection | Basis / limitation |
|---|---:|---|
| Generate 100,032 core trajectories | ~3.3–5.1 minutes | Linear scaling of measured pilot and smoke rates; no large-run I/O measurement |
| 100 training epochs | ~1.69 hours | 374.581s pilot x 16.28125; assumes costs scale with counts |
| 300 training epochs | ~5.08 hours | Three times the above; unchanged optimization may need a different schedule |
| Three 300-epoch training seeds | ~15.25 hours | Excludes controls, setup, queue and verification |
| Train+validation preload arrays | ~1.10 GiB | Exact float32 array-size arithmetic; excludes process/temporary memory |

At equal epochs, 100k gives roughly 16.28x the training updates, so an apparent
gain cannot be attributed solely to data diversity. At a fixed update budget,
100 baseline epochs correspond to only about 6.14 full-dataset epochs. Report
both data exposure and update counts.

Hold the validation cohort fixed when studying training size. Today's trainer
and evaluator do not yet support arbitrary training-group subsets/external
manifests, so this comparison has an implementation cost even though the full
preset exists. Preserve all group ICs when subsetting. Begin with a training-
group learning curve and, if warranted, a modest expansion before 100k.

Combining 4x512 with 100k would multiply dense-network work by about **358x**
relative to the current model/data at equal epochs. This is arithmetic work,
not a forecast of wall time. Such a combined change would also hide which factor
caused an improvement.

## What transfers to the fluid milestone

Optimization discipline, reliable splits, normalized loss design, failure
analysis, checkpoint portability and runtime measurement are reusable. Natural
ODE modes and enforced linearity are specific to this benchmark; nonlinear
Navier–Stokes evolution does not generally obey the same superposition rule.
Prefer a limited, measured ODE study that leaves time for the November solver.
No evidence here establishes accuracy or speed for the unimplemented fluid track.

AGENTS.md's numerical-comparison caveat is useful, but the distinction matters:
the matrix-exponential solution formula is exact mathematically, while SciPy's
implementation uses a numerical approximation with finite-precision error.
[Official SciPy expm documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html).
Whether this surrogate wins on a particular batched workload is a benchmark
question, not something this analysis assumes in either direction.

## Updated decision sequence and stopping rule

1. Review training/validation failure regimes and physical consistency.
2. Run the matched longer-optimization pair only when execution is requested.
3. Compare forcing features, then natural decay/mode features, individually.
4. Test the cheaper physical relative-squared objective; use full-trajectory
   loss only if a more direct objective justifies the sampler/optimization work.
5. Compare width 256. Proceed to greater width/depth only with demonstrated value.
6. Choose between a group learning curve/data expansion and a linear-response
   research branch based on the remaining error pattern. Keep exact-IC, generic
   Fourier and residual-penalty alternatives conditional.

Keep a candidate if its mean/tail/family improvement is repeatable across paired
seeds and worth its measured training/inference/engineering cost. Report relative
mean reduction, p95/failure changes, extra training hours and latency ratio.
Prefer candidates that improve accuracy without worsening cost; where there is a
trade-off, retain both on the accuracy-versus-cost frontier for review. Stop
adding complexity when the provisional goals are met and move to packaging and
fresh final evaluation. No model-specific gain or dollar saving is claimed yet.

The calculations and source hashes are in [cost_benefit.json](cost_benefit.json).
The combined draft comparisons are in [experiments.json](experiments.json).
