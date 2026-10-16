# Measured compute and remaining allocation questions

Updated October 16, 2026. Local timings below are measured. The old assumption
of one 32-core HPC node is withdrawn: no scheduler allocation, cluster login or
NVIDIA GPU is available in this session. Cluster-wide core counts are not a job
allocation. This generator operates on one node.

## Local generation benchmark

Host: AMD Ryzen 5 5600, six physical cores / 12 logical CPUs, 31 GiB RAM.
Python 3.14.8, NumPy 2.5.2, SciPy 1.18.1. Native OpenMP/BLAS threads were one.
Each measurement generated the same 384-case recipe in a fresh temporary
location, including preflight, process startup, two solves, matrix exponentials,
compressed archives and ledger writes. Three repetitions per worker setting.

| Execution | Median seconds | Trajectories/second |
|---|---:|---:|
| Serial | 4.039 | 95.1 |
| Six processes | 1.174 | 327.0 |
| Twelve processes | 1.448 | 265.2 |

Evidence: [benchmark report](runs/20261016-generation-benchmark-01/report.json).
No reference failures occurred. Stage sums are worker-seconds and cannot be
added to infer elapsed time in parallel runs. In the first serial run, coarse
integration took 1.595s, refinement 2.064s and matrix exponentials 0.039s.
Refinement is expensive but mandatory; independent checks cost about 1% of
numerical-stage time in this configuration. Keep both safeguards.

**Use six workers locally.** Twelve workers were about 23% slower on this small
workload; this does not establish the best setting on another CPU or filesystem.
Repeat there with actual allocated CPU counts. Avoid concurrent heavy work when
benchmarking. Filesystem cache state was not controlled, and /tmp is not cluster
scratch; these are local planning measurements, not universal throughput claims.

## Dataset size and training budget

Select the 6,656-case pilot: 4,096 train; 1,024 validation; 1,024 reserved test;
256 parameter-only and 256 IC-only validation diagnostics. Config:
[generator_pilot.json](configs/generator_pilot.json). The smoke-rate projection
was about 20s generation; the run note records measured pilot time separately.
The 100,032-case preset remains unused until learning curves justify expansion.

At M=6, training preload uses 35 float32 values per time point (23 inputs, six
targets, six masks): 140 bytes/point, about 58 MB for pilot training and 14.5 MB
for validation, excluding temporary arrays/model/loader overhead. Raw pilot
reference states occupy about 13.4 MB; archives also contain times/metadata.
Use `--preload --loader-workers 0` to avoid shuffled archive decompression.

The pilot budget is 100 CPU epochs, batch 1024, two PyTorch threads, 3x128 SiLU
MLP, AdamW at 1e-3. Training and validation-relative-L2 timings are recorded in
`runs/20261016-pilot-baseline-01/run.json`. Different batch sizes change the
optimization path. No extrapolated GPU time is used to justify this experiment.
Inference timing uses `benchmark.py inference`, excludes checkpoint loading,
warms up, and synchronizes GPU work. Reports include device/build, hardware,
batch sizes, thread count, median/p95 latency and throughput.

## Actual target allocation: unresolved

Needed: cluster login/access, scheduler job/account/partition, allocated CPUs,
RAM and wall limit, GPU model/edition and count, driver and supported CUDA
PyTorch environment. There are no SLURM/PBS allocation variables or configured
SSH host in the local session. No target job has been submitted.

Inside an actual allocated GPU job, activate its CUDA environment and run:

```bash
# Copy/edit the execution profile for the actual allocation.
bash ode/target_smoke.sh 20261016-target-smoke-01 ode/configs/resources_cuda.example.json
```

The script checks the NVIDIA build and forward/backward execution, captures
scheduler/GPU/environment evidence, generates 384 cases, trains five epochs,
reloads the checkpoint through the CLI, evaluates validation and measures GPU
inference. It fails on this local ROCm environment by design. It does not request
or guess scheduler resources, install packages, or launch the 100k job.

Choose the CUDA installation for the target driver/Python using the official
[PyTorch installer](https://pytorch.org/get-started/locally/). The CUDA and ROCm
builds share a `torch.cuda` API, so availability alone does not identify NVIDIA;
the readiness gate also checks `torch.version.cuda` and `torch.version.hip`.
Record the actual tested build before treating it as a reproducible target lock.
