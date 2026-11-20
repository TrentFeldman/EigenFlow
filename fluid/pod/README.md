# fluid_pod: AMReX plotfiles → HDF5 → POD

`fluid_pod` turns the 2D AMReX output of the multimaterial solver (or any 2D
AMReX code) into one HDF5 file per run. It then reduces that file with proper
orthogonal decomposition (POD). It is a single C++17 executable built with the rest of
`fluid/`; LAPACK does the linear algebra.

```text
run_000/                       fluid_pod convert         fluid_pod pod            fluid_pod reconstruct
  nddataPLT00000000/   ─┐
  nddataPLT00000030/    ├──►  run_000.h5           ──►  run_000.nddataPLT  ──►  ...rank5.h5
  MOF_PLT00000000/      │     /nddataPLT  /MOF_PLT       .pod.h5                 (+ error report)
  chk00030/  (skipped)  ┘     + .vtkhdf for ParaView    + .vtkhdf (modes)
```

**Status: prototype.** It is checked on synthetic plotfiles and on one real run
(`~/dat/run_000`); see [verification](../docs/pod-toolkit-verification.md).
It handles 2D only.

## Build

`fluid_pod` is part of the normal `fluid/` build and needs LAPACK in addition
to HDF5. On Arch, LAPACK is the `lapack` package; on Debian/Ubuntu it is
`liblapack-dev`. MKL and OpenBLAS also work. From `fluid/`:

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j
ctest --test-dir build/release -R pod --output-on-failure   # the six pod_* suites
./build/release/fluid_pod --help
```

If CMake cannot find LAPACK, it prints a notice and skips `fluid_pod`; the rest
of `fluid/` still builds. Add `-DFLUID_BUILD_POD=OFF` to skip it on purpose.
Use a Release build for real data, because Debug builds are unoptimized.

## Quick start

Shell examples from here on are [fish](https://fishshell.com/) (tested with fish 4.9).

```fish
set P ./build/release/fluid_pod

$P info ~/dat/run_000                        # what series, steps, times, grids, variables?
$P info ~/dat/run_000 --vars                 # ... including every variable name
$P convert ~/dat/run_000 -o run_000.h5       # all series, all variables
$P pod run_000.h5 -s nddataPLT -v x_velocity,y_velocity -e 0.999
$P reconstruct run_000.nddataPLT.pod.h5 -k 5 --compare
```

Output of the `pod` step on `run_000`, with 21 snapshots of a 128×64 grid and
two velocity components:

```text
 mode          sigma    energy%       cum%      rel.err
    1   7.219737e-01    72.5232    72.5232   5.2418e-01
    2   3.697689e-01    19.0237    91.5469   2.9074e-01
    3   2.001149e-01     5.5718    97.1187   1.6974e-01
    4   1.132240e-01     1.7837    98.9023   1.0477e-01
    5   6.782299e-02     0.6400    99.5424   6.7650e-02
    ...
```

`rel.err` is the relative error of the **fluctuation** after the mean is
removed, using the first k modes. `reconstruct --compare` reports the error of
the **full field**, which is smaller because the mean is reconstructed exactly.

Every command has `--help`, and `fluid_pod help <command>` shows the same text.

## Concepts

**Series.** A run directory holds several plotfile series, named after the
directory name with the step digits removed. `nddataPLT00000030` belongs to
series `nddataPLT`. Each series becomes one HDF5 group. Directories without
an AMReX plotfile `Header` are skipped and reported by `info`. This covers
checkpoints (`chk*`) and AMReX's `*.old.*` backups.

**Uniform grid.** POD needs every snapshot on the same grid, but AMR grids move
when the solver regrids. `convert` composites every snapshot onto one uniform
grid at the **finest resolution that appears anywhere in the series**. It
copies coarse values into each fine cell they cover (piecewise-constant
injection), and finer levels overwrite coarser ones. A series whose finest
level only appears after a few steps still works: the early snapshots are
copied up from the coarse levels. `level_mask` records which AMR level
supplied each value.

**Weights.** POD uses the physical inner product ∫ u·v dA. Each cell is
weighted by its area dx·dy, or by 2πr·dx·dy in RZ coordinates
(`coord_sys = 1`). Singular values therefore have physical meaning:
σ² = Σₜ ∫ |u′|² dA. `--no-weights` switches to the plain Euclidean product.

## Commands

### `info PATH...`

`PATH` can be a run directory, one plotfile, or a `fluid_pod` HDF5 file.
- For a run directory, it prints a table of series showing file count, step
  and time ranges, grid size and AMR level count.
- For a single plotfile, it prints each level's grids, dx, coverage and ghost
  cells.
- For an HDF5 file, it prints the series in it, or the energy table if it is
  a POD file.

### `convert [options] PATH...`

| Option | Meaning |
|---|---|
| `-o FILE` | Output file. Default: `<run-dir-name>.h5` in the current directory. |
| `-s GLOBS` | Only these series: `-s nddataPLT`, `-s 'MOF*,nd*'`. |
| `-v GLOBS` | Only these variables: `-v 'x_velocity,y_velocity,PRES*'`. A pattern that matches nothing in any series is an error. |
| `--steps A:B` | Only plotfiles whose step number is in [A, B]. Either side may be empty: `--steps 300:`. |
| `--times A:B` | Only plotfiles whose physical time is in [A, B]. |
| `--every N` | Keep every Nth plotfile, applied after `--steps` and `--times`. |
| `--f32` | Store float32 instead of float64. Half the size; `pod` still computes in float64. |
| `--gzip L` | gzip level 0–9. The default is 4; 0 is fastest (see below). |
| `--no-mask` | Do not store `level_mask`. |
| `--no-vtk` | Do not write the ParaView `.vtkhdf` sidecars. |
| `-j N` | Reader threads. The default is all cores. |
| `-n` | Dry run: print the plan and estimated size, write nothing. |
| `-f` | Overwrite an existing output file. |
| `-q`, `--verbose` | Print less, or print per-batch progress and the skipped directories. |

The file is written as `OUT.partial` and renamed only when complete. A failed
or interrupted conversion therefore never leaves a file that looks valid.
Memory is bounded at about 1 GiB of snapshots in flight, whatever the run
length.

Conversion speed for `run_000`, Release build on this laptop: 42 plotfiles,
60 variables, 79 MiB raw.

| Settings | Time | File size |
|---|---|---|
| `--gzip 0` | 0.11 s | 79 MiB |
| default (`--gzip 4`) | 0.84 s | 19.3 MiB |
| `--f32` | 0.42 s | 9.0 MiB |

HDF5 compresses serially, so on big runs use `--gzip 0` or `--gzip 1` when
speed matters more than disk.

### `pod [options] SNAPSHOTS.h5`

| Option | Meaning |
|---|---|
| `-s NAME` | Series to decompose. Optional when the file has only one. |
| `-v GLOBS` | **Required.** Variables stacked into the state vector, in this order. |
| `-k K` | Number of modes to store. Default 10, capped at the rank. |
| `-e FRAC` | Store the fewest modes that reach this energy fraction, e.g. `-e 0.99`. With `-k`, the smaller count wins. |
| `-t SLICE` | Snapshot indices as a Python slice: `5:` skips 5 startup snapshots, `::2` takes every other one, `-10:` the last ten. |
| `-r X0:X1,Y0:Y1` | Physical sub-rectangle; either side may be empty: `-r ',0.2:'`, `-r '-0.5:0.5,'`. |
| `--normalize` | Divide each variable by its fluctuation RMS. **Use this when mixing units** (velocity with pressure); otherwise the largest-magnitude variable dominates. |
| `--no-mean` | Decompose raw snapshots instead of fluctuations. |
| `--no-weights` | Plain Euclidean inner product. |
| `-m svd\|gram` | `svd` (default): thin SVD of the snapshot matrix with LAPACK `dgesdd`, the most accurate. `gram`: method of snapshots, an eigendecomposition of XᵀX; faster and smaller when grid points ≫ snapshots, accurate to about √ε relative. |
| `--check` | Verify ΦᵀWΦ = I and compare the actual rank-k error with the error predicted from the singular values. |
| `--csv PREFIX` | Also write `PREFIX_energy.csv` and `PREFIX_coeffs.csv`. |
| `-o FILE` | Default: `<input-stem>.<series>.pod.h5`, plus `.vtkhdf`. |

### `reconstruct [options] POD.h5`

This rebuilds the decomposed snapshots from the first K stored modes
(`-k K`, or `-e FRAC`). `--compare` prints the area-weighted relative L2 error
of each variable against the original file. `--compare-csv FILE` writes the
error for every snapshot. `--no-output` compares without writing fields. The
output has the same layout as `convert` output, so `pod`, `export` and
ParaView all read it.

### `export [options] FILE.h5`

This writes every dataset to `.npy`, so `numpy.load` works without h5py.
Dataset `/a/b/c` becomes `a.b.c.npy`, and the attributes of every group go to
`attributes.json`. `-p /nddataPLT` limits the export to one group,
`--only '*/x_velocity,*/time'` filters by path, and `--list` lists datasets
without exporting.

## Recipes

```fish
set P ./build/release/fluid_pod

# Velocity + pressure POD, skipping the first 5 snapshots, modes to 99.9 % energy
$P pod run.h5 -s nddataPLT -v 'x_velocity,y_velocity,PRES_MG' --normalize -t 5: -e 0.999 -o uvp.pod.h5

# POD of the region near an interface, method of snapshots, CSV for plotting
$P pod run.h5 -s nddataPLT -v 'x_vel*,y_vel*' -r '-0.5:0.5,0:0.5' -m gram -k 20 --csv near_interface -o near_interface.pod.h5

# Error versus rank (store enough modes first)
$P pod run.h5 -s nddataPLT -v x_velocity,y_velocity -k 16 -o uv.pod.h5
for k in 1 2 4 8 16
    echo -n "k=$k "
    $P reconstruct uv.pod.h5 -k $k --compare --no-output | tail -1
end

# Batch-convert many runs into float32 files, keeping only the series that matters.
# The trailing / makes run_*/ match directories only (skips run_000.tar.gz).
for r in ~/dat/run_*/
    $P convert $r -s nddataPLT -v 'x_vel*,y_vel*,PRES*' --f32 -o (basename $r).h5 -q
end
```

Quote glob patterns given to `-v`, `-s` and `--only` (`'x_vel*'`) so that fish
passes them through instead of expanding them, or failing with "No matches for wildcard".

## Output files

All arrays are row-major with x varying fastest: shape `[t, y, x]` for
snapshot fields, `[mode, y, x]` for modes. Every file carries `format` and
`format_version` attributes.

**Snapshot file** (`format = eigenflow-snapshots`): one group per series.

| Path | Shape / type | Content |
|---|---|---|
| `/<series>/fields/<var>` | `[nt, ny, nx]` f64 or f32 | composited field |
| `/<series>/level_mask` | `[nt, ny, nx]` u8 | AMR level that supplied each value |
| `/<series>/time`, `step` | `[nt]` | physical time and solver step, sorted by step |
| `/<series>/x`, `y` | `[nx]`, `[ny]` | cell-centre (or node) coordinates |
| attributes | | `variables` (in stored order), `prob_lo`, `prob_hi`, `dx`, `origin`, `coord_sys`, `centering`, `index_type`, `ref_ratio`, `target_level`, `plotfiles` |

**POD file** (`format = eigenflow-pod`):

| Path | Shape | Content |
|---|---|---|
| `/modes/<var>` | `[k, ny, nx]` | spatial modes φᵢ, the part of each mode belonging to that variable |
| `/mean/<var>` | `[ny, nx]` | temporal mean, in original units |
| `/coeffs` | `[nt, k]` | temporal coefficients aᵢ(t) = σᵢ vᵢ(t) |
| `/singular_values`, `energy_fraction`, `cumulative_energy` | `[rank]` | every singular value, not only the k stored |
| `/weights` | `[ny, nx]` | inner-product weights W |
| `/time`, `snapshot_index`, `x`, `y` | | the snapshots and region used |
| attributes | | `source`, `series`, `variables`, `var_scale`, `mean_subtracted`, `normalized`, `weighting`, `method`, `region_index`, `numerical_rank`, geometry |

Reconstruction: `field_v(t) = mean_v + var_scale_v · Σᵢ coeffs[t,i] · modes_v[i]`.
The modes satisfy `Σ_v Σ_cells W φᵢ φⱼ = δᵢⱼ`.

**ParaView sidecar** (`*.vtkhdf`): open it with *File → Open*. It is a few KiB
of VTK's native HDF5 image format, whose arrays are HDF5 external links into
the `.h5`, so **keep the two files in the same directory**. For snapshot files,
ParaView time is physical time. For POD files, "time" is the mode number,
`mode_<var>` holds the modes and `mean_<var>` holds the mean.

XDMF (`.xmf`) was tried first and dropped: ParaView 6.2's XDMF2 reader cannot
read HDF5 2.x files, and its XDMF3 reader crashed on them.

## Using the results elsewhere

```python
import numpy as np                      # after: fluid_pod export run_000.nddataPLT.pod.h5
d = "run_000.nddataPLT.pod_npy/"
modes = np.load(d + "modes.x_velocity.npy")      # (k, ny, nx)
a     = np.load(d + "coeffs.npy")                # (nt, k)
mean  = np.load(d + "mean.x_velocity.npy")       # (ny, nx)
u_rank3 = mean + np.einsum("tk,kyx->tyx", a[:, :3], modes[:3])   # var_scale is 1 unless --normalize
```

With h5py the paths are the same (`f["modes/x_velocity"][...]`). From C++ the
repository already links the HDF5 C++ API: `H5::H5File("run.h5", H5F_ACC_RDONLY)`.

## Limitations

- **2D only.** 3D headers are rejected with a message. Checkpoint directories and
  particle data are not read.
- **Injection, not interpolation.** Regions never refined show the coarse
  level's staircase on the fine grid, and the modes inherit it. `level_mask`
  shows where. In `run_000`, x < −0.75 is never refined.
- **Snapshots count equally.** POD does not weight by Δt; use `--every` or
  `-t` to even out irregular output cadence.
- Face-centred data is displayed on cells in ParaView, offset by half a cell.
  The stored arrays and attributes keep the exact staggering. Nodal data gets
  uniform weights, not trapezoid weights.
- `pod` holds the snapshot matrix in memory: 8·m·n bytes, plus a second copy
  with `--check -m svd`. It uses 32-bit LAPACK indices, so m ≤ 2³¹ rows. Use
  `-r`, `-v` or `-m gram` for very large runs.
- Built and tested on Linux (GCC 16, HDF5 2.2, reference LAPACK). The code
  avoids POSIX-only APIs, but it has not been built on Windows or macOS.
