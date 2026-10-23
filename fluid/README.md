# Fluid prototype

A C++17 foundation for a 2D incompressible fluid simulator and ML supersampler.
Grid storage, visual initializers, geometry, output, configuration and update
buffer management work. **There is no Navier–Stokes solver yet.** The executable
exports the initial state at time zero; it does not advance the physics.

## Build and test

Requires CMake 3.20+, a C++17 compiler, and the HDF5 C++ development library.
For example, the HDF5 package is `libhdf5-dev` on Debian/Ubuntu or `hdf5` on Arch.
From this folder:

```bash
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug -j
ctest --test-dir build/debug --output-on-failure

cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j
ctest --test-dir build/release --output-on-failure

./build/debug/fluid_demo 17 17 vortex
./build/release/fluid_demo --config configs/default.cfg
```

Debug keeps debugging information; Release enables optimization. Both builds
run the same tests, including checks that remain active with `NDEBUG` defined.
Build products and test output stay in ignored `build/` folders. The executable
is now under the chosen build folder; an old root-level binary is not updated.

## Run and replay

```bash
./build/debug/fluid_demo                    # 64x64 gradient
./build/debug/fluid_demo 8 4 still
./build/debug/fluid_demo 17 17 center
./build/debug/fluid_demo 17 17 inward
./build/debug/fluid_demo 17 17 vortex
./build/debug/fluid_demo --help
```

Each invocation creates a fresh folder in **`saves/` under the working directory**:

```text
saves/YYYYMMDD-initialization/
    grid.h5
    grid.svg
    resolved.cfg
    environment.txt
    status.txt
```

Dates use UTC. Further invocations use `-001`, `-002`, etc.; existing folders,
including interrupted ones, are never reused. Old `saves/grid.h5` and
`saves/grid.svg` files are left alone. The program prints the new folder path.

Edit [configs/default.cfg](configs/default.cfg) to set grid size, domain,
material, pattern, future time settings, folder name and output root. It uses
plain `key=value` lines, with `#` comments. Unknown/duplicate keys, invalid
numbers and unsupported boundary/layout values are errors. Missing settings
use defaults, and the saved `resolved.cfg` always lists every setting.
Relative paths resolve against the **working directory**, including during replay.
No RNG is involved, so no seed is needed.

To repeat an initialization, substitute the folder printed by your run:

```bash
./build/debug/fluid_demo --config saves/YYYYMMDD-initialization/resolved.cfg
```

This creates another folder with the same numeric initial state. Timestamps,
paths and HDF5 container bytes need not match. `environment.txt` records compiler,
build type, HDF5 runtime version and a SHA-256 fingerprint of the implementation
and build recipe. Keep the corresponding source as well: the fingerprint
identifies it but does not archive it.

`status.txt` starts as `running` and becomes `initialized` after both exports
finish. Caught failures become `failed` when the filesystem permits; a crash or
forced interruption may leave `running` and partial files. `initialized` means
export succeeded, not that physical validation or simulation completed.

For a **recorded scientific experiment**, set `output_root=runs` and a descriptive
`name`, giving `runs/YYYYMMDD-purpose[-NNN]/`. Follow the repository run convention:
add a run README with question, dataset identity, results, failure cases and
keep/reject/investigate decision. Ordinary initialization previews default to
`saves/`, as requested. Do not commit generated HDF5/SVG datasets.

## Code responsibilities

| Module | Responsibility |
|---|---|
| `FluidGrid.h/.cpp` | `Cell`, domain geometry, `[height][width]` storage, checked and periodic access |
| `FlowPresets.h/.cpp` | Named deterministic initialization; resets every cell |
| `Simulation.h/.cpp` | Owns current/next state and clock; interface for a future numerical stepper |
| `GridOutput.h/.cpp` | Console, HDF5 and SVG exporters; no state updates |
| `RunConfig.h/.cpp` | Validated configuration and round-trip recipe files |
| `RunOutput.h/.cpp` | Unique run folders, environment/status records and initial-state export |
| `Validation.h/.cpp` | Periodic diagnostics, exact Taylor–Green reference and rollout acceptance harness |
| `main.cpp` | CLI arguments and error reporting |
| `tests/test_fluid.cpp`, `tests/test_cli.cmake`, `tests/test_validation.cpp` | Software regression, CLI replay and analytical validation-tool checks |
| `tools/benchmark_compression.py`, `tools/compression_fixture.cpp` | Isolated compression measurement; production writer remains uncompressed |
| `cmake/BuildInfo.h.in` | Build metadata template populated by CMake |

The core library has no HDF5 dependency. `fluid_output` links the HDF5 C++ library.
Output functions now live outside `FluidGrid`; the old member calls `grid.save()`,
`grid.save_svg()` and `grid.print()` are replaced by free functions:

```cpp
#include "FluidGrid.h"
#include "FlowPresets.h"
#include "GridOutput.h"

FluidGrid grid(8, 4, Domain{2.0, 1.0, -1.0, -0.5});
initialize_flow(grid, FlowPattern::Vortex);
print_grid(grid);
// Parent folders must already exist. These explicit paths are overwritten.
save_hdf5(grid, "saves/manual.h5");
save_svg(grid, "saves/manual.svg");
```

## Geometry and stored fields

All seven fields are **cell-centered**, as selected for this stage. Both axes
are periodic. Width and height are runtime cell counts; the default physical
rectangle is 2 m by 2 m with lower-left origin (-1 m, -1 m).

`dx = length_x / width` and `dy = length_y / height` are cell sizes in metres.
The center of column `x` is `origin_x + (x + 0.5) * dx`, likewise for row `y`.
There are no duplicated endpoint cells or stored ghost cells. For example, a
four-column domain from -1 m to +1 m has centers -0.75, -0.25, 0.25, 0.75 m.
Increasing y points upward.

`grid.at(x, y)` returns a reference and throws `out_of_range` for invalid indices.
`grid.periodic_at(x, y)` takes signed indices and wraps them: column -1 is the
last column, and column `width` is column zero. Multiple wraps also work. Wrapped
access defines neighbor lookup; it does not smooth discontinuities or enforce
incompressibility.

| Field | Units / meaning |
|---|---|
| `velocity_x`, `velocity_y` | m/s |
| `pressure` | Gauge pressure in Pa; negative is allowed |
| `density` | kg/m³; uniform, default 1 |
| `kinematic_viscosity` | m²/s, momentum diffusivity; uniform, default 0.01 |
| `force_x`, `force_y` | Force per volume in N/m³, **not acceleration** |

Density must be positive; viscosity may be zero but not negative. The example
material values are not calibrated. Temperature/energy are outside the current
incompressible, isothermal scope. Geometry and config reject invalid values;
manual edits through `at()` remain the caller's responsibility.

## Initial patterns and views

All presets set pressure and forces to zero and apply uniform configured material
values. Let `X = 2*(x+0.5)/width - 1` and `Y = 2*(y+0.5)/height - 1` be dimensionless
cell-center coordinates. Each axis is normalized separately. Velocity scale is
1 m/s:

| Pattern | Velocity `(u, v)` |
|---|---|
| `still` | `(0, 0)` |
| `gradient` (default) | `((Y+1)/2, 0)` |
| `center` | `(X, Y)` |
| `inward` | `(-X, -Y)` |
| `vortex` | `(-Y, X)` |

Compared with the old endpoint sampling, the edge cells now lie inside the
boundary: a two-row gradient has speeds 0.25 and 0.75 m/s, not 0 and 1.
A one-cell axis remains at its midpoint. On nonsquare physical domains, the
normalized `vortex` is an elliptical visual pattern rather than rigid rotation.

These are **visual initializers, not validated periodic incompressible flows**.
Center/inward have nonzero divergence; the nonconstant presets have jumps at
some periodic seams. Appropriate periodic initial conditions and pressure
correction belong to the pending solver work.

The console uses one dominant-direction character per cell (`>`, `<`, `^`, `v`),
`.` for zero velocity and `?` for nonfinite velocity. Ties favor horizontal.
The SVG shows exact velocity angles and relative lengths, with zero-speed dots;
the fastest arrow spans 75% of a cell. It is an **index-space** chart, not a
physical-aspect-ratio plot when `dx != dy`. Scaling is per export; use its maximum
speed label when comparing charts. Open `grid.svg` in a browser/vector editor.

## Update boundary

`Simulation` owns independent current and next grids. A future `FluidStepper`
receives a const current grid, a writable next grid, time and positive `dt`
(seconds). Each attempt starts next as a copy of current. On success the buffers
swap and time/step count advance. An exception or changed geometry preserves
the current state and clock; a retry resets next before calling the stepper.

The copy preserves unchanged fields and makes failed retries predictable; it is
not a performance optimization. Steppers must not retain buffer references.
`advance()` checks time and geometry bookkeeping, **not physical correctness or
stability**. No concrete fluid stepper is included. `Validation.h` now supplies
independent diagnostics and a Taylor–Green rollout evaluator for checking that
stepper once it exists. See [the validation protocol](docs/numerical-validation.md).

The config records positive `dt` and `duration` with `dt <= duration` for future
use. The CLI currently ignores them for evolution and always saves `time=0`,
`step=0`. A CFL/adaptive time-step policy, pressure discretization and handling
of cell-centered pressure/velocity coupling still need numerical design.

## HDF5 format 2

`save_hdf5()` writes `/cells/velocity_x`, `velocity_y`, `pressure`, `density`,
`kinematic_viscosity`, `force_x`, `force_y`. Each dataset has shape
`[height, width]`, row zero first, portable little-endian float64 values and a
`units` attribute. Datasets remain contiguous and uncompressed; no loader exists.

Root attributes are `format_version=2`, `width`, `height`, `axis_order=yx`,
`field_location=cell_center`, `boundary_x=periodic`, `boundary_y=periodic`,
`length_x`, `length_y`, `origin_x`, `origin_y`, `dx`, `dy`, `length_units=m`,
`time`, `time_units=s`, and `step`. The version changed because geometry and
state-time metadata are now part of the contract. The resolved config alongside
the snapshot records the preset and future time settings.

Inspect with `h5dump -H <run-folder>/grid.h5` or
`h5dump -d /cells/velocity_x <run-folder>/grid.h5`.
Explicit-path exports overwrite; nonfinite fields/times are rejected before
opening the file. SVG also rejects overflowing speeds. I/O errors throw and
may leave partial output. The unique folders protect previous CLI runs.

## Human review pending

- [Compression](docs/compression-measurements.md): 1024² Taylor–Green shrank
  56→18.8 MiB, but cached copy/flush rose 42→433 ms; production compression is off.
- [Validation](docs/numerical-validation.md): reference/diagnostic checks pass;
  actual CFD solver validation is pending. Review formulas and proposed tolerances.
- Supersampling pairs: choose filtered/downsampled reference vs separate coarse
  simulation before implementing the paired-data pipeline.

## Pinned work

5. **Numerical solver validation:** diagnostics, the analytical Taylor–Green
   reference and rollout acceptance harness now exist and their checks pass.
   Actual CFD accuracy/stability and spatial/time convergence remain unmeasured:
   no numerical stepper exists. Gates and evidence are in the
   [validation protocol](docs/numerical-validation.md).
6. **Supersampling pairs:** decide whether low resolution comes from filtering
   and downsampling a high-resolution reference or a separate coarse simulation.
   Specify filtering, alignment, fields and normalization; split by full
   trajectory/initial condition so neighboring frames cannot leak across splits.

Compression has now been [measured in isolation](docs/compression-measurements.md);
it is still disabled in the production writer. The synthetic 1024² results save
space but increase cached repack/flush time. Actual trajectory/streaming-writer
measurements are still needed before adopting a filter.

Also still pinned: production compression/chunking, fewer saved fields, export cadence,
background I/O and float32 export. Keep double reference arithmetic initially;
float32 snapshots would halve the numeric payload, but measure rounding error
before changing the format. Float32 ML training does not cancel reference error.
Changing solver precision is a separate decision.

Budget for at least 16× potential data growth: 4× per axis means 16× cells;
128² → 1024² means 64× cells. HDF5 avoids JSON overhead but does not compress
these snapshots. All seven fields and SVG output remain enabled.

Verification evidence and limitations: [housekeeping checks](docs/housekeeping-checks.md).
