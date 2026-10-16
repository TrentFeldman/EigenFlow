# EigenFlow blueprint

Status: ODE implementation is available with dry-run defaults and bounded closed-form reference checks; fluid implementation remains pending. No training dataset or trained model exists. Model accuracy and target-hardware throughput are unmeasured. The mission and deadlines in [AGENTS.md](../AGENTS.md) govern this plan.

## Project map

| Location | Responsibility |
|---|---|
| `planning/WORKFLOW.md` | Work cycle, experiment records, review and release gates |
| `planning/STATUS.md` | Current milestone, task state, risks and next action |
| `planning/DECISIONS.md` | Decisions and unresolved choices |
| `ode/BLUEPRINT.md` | October model design and evaluation contract |
| `ode/` | ODE implementation placeholders and future local artifacts |
| `fluid/BLUEPRINT.md` | November–December sequence and validation gates |
| `fluid/` | Fluid implementation placeholders and future local artifacts |

Keep each track self-contained. Do not build shared infrastructure until both tracks have a demonstrated need. Documented placeholders reserve future responsibilities. See `ode/README.md` for implemented interfaces and safe checks. Data directories and model weights remain ignored under existing repository rules.

## Delivery order

1. By October 18: freeze ODE families, parameter and initial-condition bounds, time domain, data schema, split policy and metrics.
2. By October 23: validate reference trajectories, generate a versioned dataset and train a functioning baseline.
3. By October 27: evaluate held-out generalization, inspect failures and improve reliability using validation data.
4. October 30–31: freeze ODE v1.0, verify independent inference and document measured results and reproduction steps.
5. November–December: advance the fluid track through its reference, dataset, model, rollout, demo and release gates.

No training result is complete without its configuration, dataset identity, checkpoint and evaluation record. Deadline changes and reduced scope must be recorded explicitly.
