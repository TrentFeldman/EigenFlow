# Compression measurement

Question: how do lossless HDF5 filters change size and cached repack/flush cost?

Configuration: 1024×1024, seven float64 fields, 3 repetitions; seed 20261020.
Inputs: deterministic gradient, exact Taylor–Green t=0.3, synthetic 5-mode field, and
seeded random velocity/pressure. Material and zero forces remain constant in all profiles.
Multiscale/noise pressures are synthetic, not validated Navier–Stokes states.

All trials passed bitwise field readback and h5diff (values, shapes and attributes).

| Input | Layout/filter | MiB | Size ratio | Write+fsync ms | vs plain | Warm read ms |
|---|---|---:|---:|---:|---:|---:|
| gradient | contiguous | 56.006 | 1.00× | 30.4 | 1.00× | 24.7 |
| gradient | chunk64 | 56.110 | 1.00× | 75.5 | 2.48× | 30.4 |
| gradient | gzip1_chunk64 | 0.486 | 115.16× | 71.0 | 2.34× | 56.7 |
| gradient | shuffle_gzip1_chunk64 | 0.431 | 129.92× | 163.2 | 5.37× | 67.1 |
| gradient | shuffle_gzip4_chunk64 | 0.254 | 220.58× | 306.0 | 10.07× | 140.0 |
| gradient | shuffle_gzip1_chunk256 | 0.276 | 202.57× | 198.1 | 6.51× | 61.2 |
| taylor_green | contiguous | 56.006 | 1.00× | 42.3 | 1.00× | 21.5 |
| taylor_green | chunk64 | 56.110 | 1.00× | 92.4 | 2.18× | 27.6 |
| taylor_green | gzip1_chunk64 | 20.720 | 2.70× | 453.3 | 10.71× | 121.1 |
| taylor_green | shuffle_gzip1_chunk64 | 18.819 | 2.98× | 433.3 | 10.24× | 97.8 |
| taylor_green | shuffle_gzip4_chunk64 | 18.542 | 3.02× | 536.2 | 12.67× | 141.7 |
| taylor_green | shuffle_gzip1_chunk256 | 18.338 | 3.05× | 518.1 | 12.24× | 85.5 |
| multiscale | contiguous | 56.006 | 1.00× | 41.8 | 1.00× | 19.2 |
| multiscale | chunk64 | 56.110 | 1.00× | 86.6 | 2.07× | 28.9 |
| multiscale | gzip1_chunk64 | 16.014 | 3.50× | 376.2 | 8.99× | 106.0 |
| multiscale | shuffle_gzip1_chunk64 | 14.035 | 3.99× | 363.1 | 8.68× | 87.2 |
| multiscale | shuffle_gzip4_chunk64 | 13.459 | 4.16× | 477.9 | 11.43× | 125.4 |
| multiscale | shuffle_gzip1_chunk256 | 13.261 | 4.22× | 419.6 | 10.03× | 72.3 |
| noise | contiguous | 56.006 | 1.00× | 42.5 | 1.00× | 19.4 |
| noise | chunk64 | 56.110 | 1.00× | 92.3 | 2.17× | 28.5 |
| noise | gzip1_chunk64 | 23.259 | 2.41× | 526.6 | 12.38× | 121.4 |
| noise | shuffle_gzip1_chunk64 | 22.173 | 2.53× | 529.7 | 12.45× | 129.8 |
| noise | shuffle_gzip4_chunk64 | 21.848 | 2.56× | 663.3 | 15.59× | 166.8 |
| noise | shuffle_gzip1_chunk256 | 20.923 | 2.68× | 617.3 | 14.51× | 88.5 |

Times are medians; raw trials/min/max are retained in CSV. Trial order is seeded/shuffled.
Write includes repack input reading, process startup, HDF5 close, file fsync and directory fsync.
Read includes process startup, decompression, materialization and checksum of all fields.
Inputs/outputs are page-cache warm; no cache dropping, GPU transfer or concurrent CFD was measured.
A single snapshot is not a sustained-write benchmark; Btrfs/filesystem caching may affect results.
Read-time checksums prevent unused reads. Per-field stored bytes are in field_storage.json.

Four of seven fields are uniform: density, viscosity and two forces. These favor compression.
Use velocity/pressure field sizes as a separate indicator; future turbulent frames may compress differently.
No guarantee of 16× compression or reduction in required generation bandwidth follows from this run.

Failures: none in completed trials. If interrupted, running/failed status and partial CSV remain.
Decision: investigate on actual CFD trajectories. Keep production output contiguous/uncompressed
until an end-to-end writer benchmark establishes a throughput benefit for the chosen workload.

Reproduce in a NEW folder with:
```bash
python3 tools/benchmark_compression.py --fixture build/release/fluid_compression_fixture --size 1024 --repeats 3 --output runs/20261020-compression-baseline-02
```

Reference: https://portal.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_p__u_g.html
