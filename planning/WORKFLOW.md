# Working system

## Session cycle

1. Read root `AGENTS.md`, `planning/STATUS.md` and the active track blueprint.
2. Select one bounded task with a deliverable and a verification condition. Update its status to active.
3. Perform the work in the track folder. Keep changes small; leave the archive and virtual environment alone.
4. Verify the relevant scientific or software behavior. Record the evidence and failures; empty files do not count as implemented work.
5. Update task status, decisions and the next action before ending the session. Commit or push only when requested.

Task states: planned, active, blocked, verified. A blocked task records the reason and the smallest action needed to unblock it. Only evidence-backed work becomes verified.

## Experiment records

Use a unique run ID such as `YYYYMMDD-track-purpose-01`. Each future run gets a folder under its track's `runs/`, containing a run note, resolved configuration, metrics, representative plots and local checkpoint. Keep large generated datasets in the track's ignored `data/` folder. Do not commit checkpoints or data.

Record these fields in every run note:

| Field | Required content |
|---|---|
| Question | Hypothesis and the single intended comparison |
| Provenance | Source revision and uncommitted changes, dataset version/checksum, split identifiers |
| Configuration | Equation family, bounds, time grid, numerical tolerances, model settings, seed |
| Environment | Python and dependency versions, hardware, runtime |
| Outcome | Validation metrics, failure cases, artifact paths and completion status |
| Decision | Keep, reject or investigate; next controlled experiment |

Fit normalization on training data only. Use validation data for model selection. Reserve the test set for milestone evaluation; record every test use and do not repeatedly tune against it. Failed or interrupted runs remain labeled as such. Never overwrite a completed run to reuse its identifier.

## Weekly review

Update `STATUS.md` with the review date, task states, measured progress, risks, next deliverable and target date. Link supporting run records. Distinguish proposed targets from observed results. Record scope and deadline changes in `DECISIONS.md` and keep the mission checklist consistent with verified work.

## Release gate

- Versioned dataset recipe, split manifest, configuration and recorded environment are available.
- Reference-solver validation and model evaluation are documented, including worst cases and limitations.
- Checkpoint has an accessible local or distribution location and checksum; checkpoint files remain outside Git.
- Standalone inference loads saved preprocessing and weights without invoking training.
- Setup instructions and exact reproduction commands have been exercised from a clean documented setup.
- Examples, metrics and plots correspond to the released checkpoint.
- A release report states whether accuracy targets passed, failed or were not evaluated.

Runnable ODE checks and dependency specifications are documented in `ode/README.md` and `ode/ENVIRONMENT.md`. The generator and trainer default to dry-run; bulk dataset generation and model training are explicitly deferred for the current task. Small numerical validation cases are authorized and recorded separately from dataset/model runs. No automated workflow or experiment service is required at this stage.

## Documentation handoff

Both collaborators must be able to trace an input through the pipeline. Explain equations and derivative order before relying on specialist terms; document tensor shapes, units, assumptions and numerical choices. Keep the math guide, module docstrings and configuration explanations synchronized with the code. Mark unfinished placeholders clearly.
