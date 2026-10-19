# EigenFlow

Neural networks that predict solutions to differential equations, measured
against trusted numerical solutions.

| Track | What it is | State |
|---|---|---|
| [ODE](ode/README.md) | Predicts solutions of linear constant-coefficient ODEs of order 1 to 4, with or without a sine-wave forcing term | Two trained models in [ode/models/](ode/models/) |
| [Fluid](fluid/README.md) | ML-assisted 2D Navier-Stokes fluid simulation | Planned, target December 11, 2026 |

## Setup

Run everything from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r ode/requirements.txt
.venv/bin/python -m ode check
```

GPU setup is in [ode/README.md](ode/README.md#setup).
