# Housekeeping verification — October 20, 2026

Scope: software foundation items 1–4. No CFD equations, analytical validation,
training data generation or performance benchmark were implemented in this task.
Numerical validation and supersampling-pair design remain pinned in the track
README, together with precision and I/O optimization decisions.

Environment: CMake 4.4.4, GCC 16.2.1, C++17, HDF5 2.2.0, Linux.
Both Debug and Release builds passed all seven CTest suites. Assertions use
explicit checks rather than C `assert`, so Release tests execute their checks.

Reproduce from `fluid/`:

```bash
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug -j
ctest --test-dir build/debug --output-on-failure
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j
ctest --test-dir build/release --output-on-failure
```

| Suite | Evidence |
|---|---|
| `geometry` | Rectangular spacing/origin/cell centers; const/mutable access; positive, negative and multiple-period wrapping; 1×1 with extreme signed indices; rejection of invalid sizes, bounds and unrepresentable domains |
| `presets` | All five presets on odd/even rectangles, rows, columns and 1×1; center/inward and vortex relationships; cell-center sampling; uniform material; invalid parameters |
| `simulation` | Synthetic periodic shift exposes accidental in-place writes; independent input ownership; correct time/count after success; unchanged state/time after exceptions and geometry changes; clean retry; invalid time steps |
| `config` | Full resolved configuration round trip; exact neighboring-double precision; reproducible initial velocities; comments/spacing; unknown/duplicate keys, bad values, unsupported layout/boundaries and I/O failures |
| `output` | All seven HDF5 fields round-trip bit-for-bit on 3×2, including signed zero and extreme finite doubles; shape/type/units/geometry/time attributes; contiguous uncompressed layout; nonfinite rejection preserves existing files; replacement with 1×1; SVG direction/dot and console checks; output failure paths |
| `runs` | Distinct folders for repeated configurations; previous outputs preserved; recipe/environment/status records; running/failed folder preservation; invalid output root |
| `cli` | Actual positional invocation creates output under its working directory; saved-config replay makes another folder; invalid dimensions create no run |

Generated fixtures are synthetic, deterministic and use no external dataset or
RNG seed. Outputs and CTest logs remain under each ignored build folder. A test
failure preserves its fixtures for inspection. Production compiler warnings
(`-Wall -Wextra -Wpedantic`) are also enabled for the test executable.

No functional test failures were observed in the Debug or Release checks.
A concurrent initial build emitted a GNU Make temporary jobserver FIFO collision
warning; both builds completed successfully. Subsequent verification uses
sequential builds to avoid that shared temporary-file collision.

Limits: these checks do not establish incompressibility, time-step stability,
conservation, reference accuracy, GPU performance, sustained I/O throughput,
crash-safe file writes, or cross-platform behavior. The test steppers are
artificial buffer tests, not physical solvers. HDF5 files can be partial after
an I/O failure; interrupted status may remain `running`. No load/resume API exists.

Decision: keep the tested housekeeping foundation; implement and validate the
numerical solver separately before using snapshots as reference training data.
