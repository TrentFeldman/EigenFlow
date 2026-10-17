# Project status

Updated: October 16, 2026.

GPU access correction: the sandbox cannot see GPU device files, but the host
detects the AMD Radeon RX 7900 XTX. A two-epoch ROCm training/save/reload check
passed outside the sandbox; no environment change was needed. The user is
handling the CPU-job stop. NVIDIA target verification remains separate.


First accuracy stage complete: baseline train/validation diagnostics and matched
300-epoch constant/cosine controls ran on the local 7900 XTX. Validation means
are 2.276%/2.210%, but cosine worsens p95, worst case and forced-family means.
Retain constant as the reliability control and cosine as the pooled-mean
candidate. Selection uses validation only; the old test was not re-evaluated.
See the [comparison](../ode/runs/archive/20261016-optimization-gpu-cosine-01/README.md).

Verified analytical review: [combined cost–benefit analysis](../ode/docs/error_reduction/COST_BENEFIT.md)
incorporates all six parked AGENTS.md options and updates the draft experiment
order/settings. That review was analytical; execution of the initial stage was
subsequently authorized. Later feature/loss/architecture stages remain proposals.

Verified preparation: [error-reduction plan](../ode/docs/error_reduction/README.md),
source-hashed saved-validation review, draft comparison settings and a prepared
control command. The initial preparation preserved model/trainer/generator code
and existing data/checkpoints; subsequent authorized implementation and runs are
tracked separately below.

Verified documentation handoff: [professor-facing brief](../ode/PROFESSOR_BRIEF.md)
explains the equations, reference checks, model, evaluation and pilot limitations.
Checked against the implementation and recorded results; no experiments rerun.

Current phase: scientific ODE v1 specification resolved; CPU pilot trained and
measured. **The unchanged 3x128 baseline achieved 3.689% mean relative L2 on a
1,024-trajectory joint test set**, evaluated once after validation-only checkpoint
selection. This passes the pooled pilot target, with substantial tail errors and
a 5.573% mean for the third-order sine-forced family. No October release declared.

## October 18 freeze and requested work

| Task | State | Evidence / next action |
|---|---|---|
| Families, bounds, forcing and time domain | verified | Resolved v1 contract in DECISIONS.md and ode/docs/BLUEPRINT.md; 101 points on [0,5] |
| Reference protocol | verified | All 6,656 pilot references pass preflight, refinement and independent checks |
| Primary metric / near-zero / component policy | verified | Frozen RMS-normalized relative L2, 1e-6 floor; metric tests and full reports |
| Joint, parameter-only and IC-only splits | verified | Pilot diagnostic partitions and loader ownership checks; no test leakage |
| Extrapolation policy | verified | Outside October acceptance; no out-of-range dataset required for v1 |
| Dataset size / workers | verified | Selected 6,656-case pilot; measured six local workers beat twelve; 100k unused |
| Evaluation, plotting and independent CLI | verified | Smoke/pilot reports, SVG medians/worst cases, JSON prediction and reload checks |
| Runtime benchmark | verified locally | Hardware/batch/thread details and warmed latency in pilot runtime.json |
| Real small end-to-end run | verified | 384 references → five epochs → independent reload/evaluation/plots |
| Baseline adequacy / fallback plan | verified for pilot pooled target | 100-epoch baseline, explicit family/tail limitations; controlled fallback order recorded |
| Test discovery / environment / stale docs | verified | 42 discovered tests pass, pip check passes, direct pins and recorded environment |
| Portable CPU/GPU resources | verified locally | Editable profiles, CLI precedence, scheduler-aware auto CPUs, preload cap; bitwise matching smoke rerun |
| Error-reduction first stage | verified | Diagnostics and matched GPU controls complete; no broad scheduler win; later comparisons conditional |
| Actual HPC allocation | blocked | User obtaining details in a meeting; need host/access, CPU/RAM/time/account/partition/GPU allocation |
| NVIDIA target training and throughput | blocked | Local AMD GPU works through ROCm; NVIDIA target_smoke.sh requires allocated execution and CUDA torch |
| Fluid implementation | planned | Placeholders unchanged; primary work starts November 2 |

## Measured evidence

- [Generation benchmark](../ode/runs/archive/20261016-generation-benchmark-01/README.md):
  three repetitions of 384 cases for each setting; median serial 4.039s,
  six workers 1.174s, twelve workers 1.448s. All 3,456 references passed.
- [Small end-to-end smoke](../ode/runs/archive/20261016-smoke-01/README.md): 384 valid
  references, five CPU epochs; best validation error 54.247%. Pipeline passed;
  smoke accuracy failed the target. No smoke test-set scoring.
- [Pilot result](../ode/runs/archive/20261016-pilot-baseline-01/README.md): 6,656 valid
  references in 13.056s; 100 epochs in 374.581s; best epoch 96. Validation mean
  3.578%, parameter-only 3.993%, IC-only 2.938%. One reserved-test use.
- Pilot test: mean **3.689%**, median **2.704%**, p95 **8.976%**, worst **36.908%**;
  **216/1,024 cases >=5%**, zero nonfinite predictions. Test family means:
  order-2 homogeneous 2.253%, order-2 sine 4.842%, order-3 homogeneous 2.090%,
  order-3 sine **5.573%**. Per-component and absolute/near-zero records retained.
- Loaded CPU prediction median latency: 0.086 ms / one point, 0.187 ms / 101,
  2.434 ms / 4,096; 30 repetitions, five warmups, two CPU threads. Not GPU timing.
- Independent checkpoint reload and CLI agreement passed. Resource-profile smoke
  [rerun](../ode/runs/archive/20261016-resource-smoke-01/README.md) produced bitwise identical
  selected weights with matching training options.
- [Matched GPU optimization](../ode/runs/archive/20261016-optimization-gpu-cosine-01/README.md):
  constant/cosine mean 2.276%/2.210%, p95 5.918%/6.247%, worst 20.064%/27.076%,
  83/92 of 1,024 cases >=5%; zero nonfinite predictions. Both selected epoch
  295. Training took 1,158.780/1,156.422 seconds, with no large observed device
  speedup in this small pipeline. CPU reload evaluation confirms GPU selections.
  Parameter-only means 2.733%/2.617%; IC-only 1.770%/1.561%.
- [Environment evidence](../ode/runs/archive/20261016-environment-01/README.md): Python
  3.14.8, NumPy 2.5.2, SciPy 1.18.1, torch 2.14.1+rocm7.14. No package changes.
  NVIDIA-specific CUDA readiness fails on this AMD host; local ROCm training and
  reload now pass. NVIDIA target testing is not claimed.
- `.venv/bin/python -m unittest discover -v`: **42 tests passed**. The previous
  24-test note was stale; 25 original tests plus 17 metric/resource/schedule/diagnostic checks now pass.
  The earlier 13-case reference benchmark remains valid historical evidence.

## Next actions and risks

1. After the user's allocation meeting, edit the CUDA resource profile and run
   `ode/slurm/target_smoke.sh` inside the real allocation; measure generation workers
   there. Do not reuse the old assumed 32-core node or local ROCm wheel.
2. By October 23, retain the functioning pilot baseline and both GPU controls.
   Longer training reduced mean error; cosine's small pooled gain worsened the
   joint tail and forced-family means. The next proposed comparison is forcing
   features against the constant reliability control; it has not been launched.
   The pilot test has been consumed once; do not repeatedly tune against it.
3. By October 27, consolidate failure analysis and select the release candidate.
   Keep test protocol changes explicit and preserve independently reserved data.
4. By October 30–31, package a reproducible checkpoint, environment and artifacts,
   verify clean installation/inference, and document acceptance limits. A local
   pilot passing the pooled metric is not yet a packaged release.

No 100k dataset was generated. Single-seed pilot evidence does not establish
robustness across random seeds, out-of-domain parameters, new orders or longer
times. The full-state metric can hide individual component/family weakness, so
reports retain those breakdowns. Large small-file datasets may behave differently
on cluster storage. Local ROCm execution/reload passes; seed robustness,
throughput tuning and the allocated NVIDIA target require separate evidence.

## Later milestones

| Due | Deliverable | State |
|---|---|---|
| Oct 23 | Trained ODE baseline | verified locally; target portability pending |
| Oct 27 | Higher-order reliability evaluation | active |
| Oct 30–31 | Reproducible ODE v1.0 | planned |
| Nov 6 | Validated fluid reference solver | planned |
| Nov 13 | Reproducible fluid dataset | planned |
| Nov 20 | Trained single-step fluid predictor | planned |
| Nov 27 | Evaluated autoregressive rollout and stabilization | planned |
| Dec 4 | Working comparison demo | planned |
| Dec 11 | Evaluated semester release | planned |

Review: October 16; froze scientific contract, established full CPU pipeline and
pooled accuracy feasibility; retained failure/timing evidence; selected pilot
instead of 100k; target allocation/CUDA unresolved. Next deliverable is allocated
hardware verification and baseline handoff, target October 23.
