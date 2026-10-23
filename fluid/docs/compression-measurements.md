# Compression measurement before adoption

The production `save_hdf5()` remains contiguous, uncompressed float64. The
standalone benchmark uses the installed HDF5 `h5repack` tool to compare copies
of current-format snapshots. It does not add compression settings to the writer.

The completed [baseline run](../runs/20261020-compression-baseline-01/README.md)
contains raw trials, medians/min/max, per-field sizes, deterministic recipes,
binary fingerprint, commands, environment and decision. It tested four synthetic
1024² snapshots, six layouts/filter combinations and three shuffled repetitions:
72 trials. All field values matched bit-for-bit after repacking, and `h5diff`
verified values, shapes and attributes. No failed trial was observed.

Measured on Ryzen 5 5600, GCC 16.2.1 Release, HDF5 2.2.0, Linux/Btrfs on
`/dev/nvme0n1p2`. Each snapshot has seven float64 planes: 56 MiB payload.
Below is one consistent candidate, 64×64 chunks with byte shuffle and gzip
level 1, alongside the contiguous baseline. Times are medians of three trials.

| Input | Plain MiB | Compressed MiB | Size reduction | Plain write+flush | Compressed write+flush |
|---|---:|---:|---:|---:|---:|
| Gradient | 56.01 | 0.43 | 130× | 30 ms | 163 ms |
| Exact Taylor–Green | 56.01 | 18.82 | 2.98× | 42 ms | 433 ms |
| Synthetic five-mode field | 56.01 | 14.04 | 3.99× | 42 ms | 363 ms |
| Seeded random velocity/pressure | 56.01 | 22.17 | 2.53× | 43 ms | 530 ms |

The gradient is an unusually favorable toy, not a realistic compression target.
Four of seven fields (density, viscosity, two forces) are constant in all cases.
Uncompressed those occupy 32 of 56 MiB. The per-field measurements separate
this benefit from compression of velocity/pressure. Noise and multimode fields
are synthetic stress cases, not evolved or physically validated CFD trajectories.

Considering only the 24 MiB velocity/pressure payload, the same filter stores
18.54 MiB for Taylor–Green (1.29× reduction), 13.76 MiB for multimode (1.74×),
and 21.90 MiB for noise (1.10×). Much of the overall saving comes from uniform
fields, which matters when estimating benefits for a future reduced-field dataset.

With shuffle+gzip1/64, median full-field warm reads were about 98 ms for
Taylor–Green, 87 ms for multimode, and 130 ms for noise; corresponding plain reads
were about 22, 19 and 19 ms. No training crop/partial-chunk access was measured.
Larger 256×256 chunks saved a little more space in these cases but also increased
write time. Gzip level 4 brought small extra savings and higher compression cost;
chunking without filters slightly increased file size and copy time.

These are **cached repack-and-flush measurements**, including input reads,
process startup, close and file/directory `fsync`. Read timing includes process
startup, full materialization, decompression and a checksum. They establish a
local CPU/storage tradeoff, not sustained CFD export speed, hardware write
bandwidth, cold-cache training throughput, or GPU performance. Trial order is
seeded/shuffled; input/output pages remain cache warm. Filesystem/device caching
and concurrent activity can change absolute timings. Three repeats are a pilot,
not a statistical characterization of long runs.

Decision: retain uncompressed production output and investigate compression
again on actual trajectories with a direct streaming-writer benchmark. These
measurements establish substantial space savings but no end-to-end write-speed
benefit on this local workload. They do not guarantee 16× savings or remove the
planned 16×/64× resolution-growth issue. Float32 export, fewer fields, cadence
and background I/O remain separate, deferred decisions.

## Reproduction

No Python packages are needed; use system Python plus HDF5 tools `h5repack` and
`h5diff`. Build the C++ fixture tool and use a new run directory every time:

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j
python3 tools/benchmark_compression.py --size 1024 --repeats 3 \
    --output runs/20261020-compression-baseline-02
```

The script refuses an existing output directory. It preserves fixtures, trials
and running/failed status on interruption. Generated `.h5` files are gitignored;
small recorded metrics and run notes are retained. Benchmark tools are separate
from the production save API and do not constitute a load/resume feature.

Filter/chunk syntax and external repacking behavior follow the
[official HDF5 tool documentation](https://portal.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_p__u_g.html).
