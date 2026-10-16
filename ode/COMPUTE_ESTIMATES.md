# Generation and training estimates

These are planning estimates, not measured HPC or GPU results. The only measured
runtime is a small local numerical-validation suite on the Ryzen 5 5600. No
training dataset or trained model was created for these estimates.

## Assumptions

- One allocated HPC node with 32 CPU cores, 32 process workers, and local scratch
  storage. The cluster's 20,000+ total cores across 600+ nodes are shared capacity,
  not the resources assigned to this job. This generator is single-node.
- One NVIDIA RTX PRO 6000 Blackwell. The workstation model has 96 GB GDDR7;
  exact workstation/server/Max-Q edition and host CPU affect performance.
  [NVIDIA specifications](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/).
- Current order-2/order-3 linear homogeneous and sinusoidally forced families,
  time interval [0, 5], 101 saved times, two adaptive solves and five independent
  matrix-exponential checkpoints per trajectory.
- Training uses the current 36,870-parameter MLP, 20 epochs, float32, the current
  approximately 2/3 training, 1/6 validation and 1/6 test proportions, and
  `--device cuda --preload --loader-workers 0`. Test data is not scored in training.
- Exclude scheduler queue time, installing dependencies and copying data between
  machines. Do not interpret runtime estimates as accuracy or convergence claims.

## Rough wall times

| Total trajectories | CPU generation on one 32-core node | Training on one RTX PRO 6000 Blackwell, 20 epochs |
|---|---|---|
| 384 (current example) | 5-30 seconds | 5-30 seconds |
| About 10,000 | 20 seconds-2 minutes | 1-5 minutes |
| About 100,000 | 2-15 minutes | 10-45 minutes |

These deliberately broad ranges include process/GPU startup and expected data
handling overhead. They assume good local-scratch performance. A shared filesystem
with expensive small-file operations can exceed these ranges. Generation writes
one archive per trajectory; storage/metadata overhead can dominate large runs.

The validation cases locally took approximately 6-20 ms per job, including two
solves and five independent checks, but excluding archive writing and process
startup. They are chosen benchmarks, not random-sample throughput measurements.
For larger runs, estimate:

```text
generation seconds ~= startup + N * seconds_per_job / (workers * efficiency) + I/O
```

Use measured target-node timings when available. Efficiency is less than 1;
32 workers do not imply 32-fold speedup. Higher orders, longer horizons and
stiff equations can make each trajectory much more expensive.

The model is small enough that CPU collation, transfers and launch overhead may
dominate GPU arithmetic. `--preload` avoids repeated decompression during shuffled
training. At max_order=6, preloaded examples use about 140 bytes per time point,
or 14,140 bytes per trajectory before other overhead; train plus validation need
about 4.5 MB for the current example and about 1.18 GB for 100,000 total trajectories.
This is host RAM, not GPU VRAM. Transient loading/statistics and worker copies can
increase peak memory. Use zero loader workers with preloading.

The default batch size is 256. Trying 1024-4096 can reduce GPU launch overhead,
but changes optimization and must be validated rather than assumed equivalent.
Twenty epochs on the default configuration mean 2,020 training updates plus
validation. Reaching a target error may require many more epochs, more data or
a different architecture. A 200-epoch run is roughly ten times the epoch work;
startup and one-time loading do not multiply by ten.

## Future HPC execution examples

These commands are examples only and were not executed. Request one CPU node
through the site's scheduler first; match workers to the allocated CPUs, not
the cluster's advertised total. Paths should be placed on allocated scratch.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m ode.generate_data --backend process --workers 32 \
  --generate --output ode/data/linear-v1

python -m ode.train --device cuda --preload --loader-workers 0 \
  --data ode/data/linear-v1 --run ode/runs/blackwell-baseline \
  --epochs 20 --batch-size 256 --train
```

Limiting native-library threads avoids spawning a full BLAS thread pool inside
every solver process. Run inside a suitable environment on the allocated host.
The local ROCm environment cannot provide NVIDIA CUDA support on another machine.
Use a Blackwell-compatible CUDA PyTorch build on the GPU host; choose the current
supported build with the cluster administrator. [PyTorch's CUDA support matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)
and [Blackwell support announcement](https://pytorch.org/blog/pytorch-2-7/).

No multi-node partition/merge layer, scheduler submission automation or GPU
benchmark is implemented. The next useful measurement is a small benchmark on
the actual allocated hardware, before reserving a long job.
