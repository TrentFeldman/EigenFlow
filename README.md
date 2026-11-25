# EigenFlow

Neural networks that predict solutions to differential equations, measured
against trusted numerical solutions.

| Track | Contents | State |
|---|---|---|
| [ODE](ode/README.md) | Linear constant-coefficient ODEs of order 1 to 4, with or without sine forcing | Two released [models](ode/models/) |
| [Fluid](fluid/README.md) | ML-assisted 2D Navier-Stokes simulation | Prototype, target December 11, 2026 |
| [Sim](sim/README.md) | Sussman's AMReX multiphase solver, modified for speed | Working in 2D and 3D |

| Reference | Contents |
|---|---|
| [ODE math guide](ode/MATH_GUIDE.md) | Derivations behind the ODE track |
| [fluid_pod](fluid/pod/README.md) | AMReX plotfiles to HDF5, POD and ParaView |
| [Solver on a workstation](docs/README_LOCAL.md) | Solver inputs, outputs and run settings |
| [Solver on FSU RCC](docs/README_HPC.md) | Slurm jobs and cluster setup |
