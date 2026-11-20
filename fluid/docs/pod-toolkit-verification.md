# POD toolkit verification: November 19, 2026

Scope: `fluid_pod` (AMReX plotfile reader, HDF5 conversion, POD, reconstruction,
export) as described in [pod/README.md](../pod/README.md). This is not a
validation of the multimaterial solver or of the physical results.

Environment: Linux, GCC 16.2.1, C++17, CMake 4.4.5, HDF5 2.2.0, reference
LAPACK/BLAS, ParaView 6.2.1 / VTK 9.7.2 (sidecar check only).

Reproduce from `fluid/`:

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j
ctest --test-dir build/release --output-on-failure
```

## Results

- Debug and Release each pass 14 of 14 CTest suites: the 8 existing suites and
  the 6 new `pod_*` suites.
- `-DFLUID_BUILD_POD=OFF` builds and passes the original 8 suites.
- All targets compile without warnings under `-Wall -Wextra -Wpedantic`.
- A Debug build with `-fsanitize=address,undefined` passes the six `pod_*`
  suites and a full convert → pod → reconstruct run on real data. There were
  no sanitizer reports.

| Suite | Evidence |
|---|---|
| `pod_reader` | Synthetic plotfiles with 3 levels (ratios 2 then 4), two grids per level split across two `Cell_D` files at non-zero offsets. Ghost cells are filled with 1e30 so that any use of them is caught. Every composited value equals the analytic field at the centre of the supplying level's cell, compared **bit for bit**. This holds for cell-centred, nodal, float32 (2 ghosts) and big-endian data. A 3D header is rejected with a "2D only" message, and a truncated FAB file is reported rather than read. |
| `pod_convert` | Run directory containing two series, a checkpoint and an `.old.` backup. The finest level appears only after step 0 and its snapshots composite from level 0. Steps come out sorted, the backup and checkpoint are skipped, and output shapes and values are exact. The dry run writes nothing, existing output is not overwritten without `-f`, and no `.partial` file remains. `-s`, `-v`, `--steps`, `--every`, `--f32`, `--no-mask` and `--gzip 0` behave as documented, and an unmatched `-v` pattern is an error. |
| `pod_pod` | Analytic rank-2 fluctuation field, Cartesian and RZ. Numerical rank is 2 by both `svd` and `gram`, and two modes capture energy 1 − 1e-12. `svd` and `gram` singular values agree to 1e-9. The stored modes are orthonormal in the weighted inner product, including the 2πr weights, to 1e-10. `-e`, `-r`, `-t`, `--normalize` and `--csv` work, and a missing `-v` or bad `-m` is rejected. |
| `pod_reconstruct` | Rank-2 reconstruction of rank-2 data matches the original to 1e-10 relative. The output can be decomposed again by `pod`. Requesting more modes than are stored is an error. |
| `pod_export` | `.npy` magic, 64-byte header alignment, dtype and shape string, and byte size are correct; `attributes.json` is written; existing directories are protected. |
| `pod_cli` | Help, version, unknown commands and options, missing inputs, globbing and slice parsing. |

## Real data: `~/dat/run_000`

The run has 21 steps (0 to 588, t = 0 to 6) in two series: `nddataPLT` with
46 variables and `MOF_PLT` with 14. Both use 2 AMR levels on a 128×64 finest
grid, the level-1 grids change between snapshots, and there are 21 checkpoint
directories.

- **Coordinates.** `nddataPLT` stores `X` and `Y` as variables. On all
  118,208 snapshot-cells supplied by level 1 they equal the grid's cell
  centres exactly. On level-0 cells they differ by exactly half a fine cell,
  which is what injection should give.
- **Agreement with the prototype.** The velocity POD energy fractions (72.5232,
  19.0237, 5.5718 %, …) match the first prototype exactly. Singular values
  differ only by the expected factor √(dx·dy) = 1/64, from the area weights.
- **`--check`.** ΦᵀWΦ − I is 4.7e-15 for `svd` and 2.6e-13 for `gram`. The
  actual rank-k error equals the error predicted from the singular values to
  all printed digits. Numerical rank is 20 for 21 snapshots, as expected once
  the mean is removed.
- **numpy.** Modes exported with `export` and checked independently are
  orthonormal to 2.7e-15.
- **ParaView.** `pvpython` opened the snapshot sidecars and the POD sidecar
  through `VTKHDFReader`. They gave 21 time steps (POD: 6 modes), 8192 cells,
  bounds [−1, 1] × [0, 1], and the same array values (first entries and sums
  bit for bit) as the HDF5 data read through `export` and numpy.

## Known gaps

- It has not been built on Windows/MSVC or macOS. Fortran hidden-length
  arguments are passed for gfortran-built LAPACK; MKL and OpenBLAS ignore them.
- Nodal, face-centred, float32 and big-endian inputs are covered by synthetic
  data only. The real run is cell-centred float64.
- XDMF output was dropped. ParaView 6.2's XDMF2 reader failed on files from
  HDF5 2.2, even for a plain contiguous dataset, and the XDMF3 reader crashed.
  VTKHDF replaced it.
- POD convergence with respect to snapshot count was not studied: `run_000`
  has only 21 snapshots.
