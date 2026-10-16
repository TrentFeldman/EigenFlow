# Python environment

Use the existing root `.venv`. The current request authorizes checking it and adding the missing SciPy dependency; the installed PyTorch/ROCm stack was retained. No system Python packages were changed.

Verified local runtime:

| Component | Version / state |
|---|---|
| Python | 3.14.8, isolated virtual environment |
| NumPy | 2.5.2 |
| SciPy | 1.18.1 |
| PyTorch | 2.14.1+rocm7.14 |
| GPU visible to PyTorch | No locally; CPU path verified, CUDA option added but not hardware-tested |

From the repository root:

```bash
source .venv/bin/activate
python -m ode.check_environment
python -m pip check
```

Without activation, use `.venv/bin/python` explicitly. This avoids accidentally using system Python or a different environment. `check_environment` checks isolation/import readiness and prints versions; `pip check` checks installed dependency compatibility.

For a **new clone** without `.venv`, create one with a supported interpreter (Python 3.11 or newer; only 3.14.8 has been checked here), then install dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r ode/requirements.txt
.venv/bin/python -m ode.check_environment
.venv/bin/python -m pip check
```

`requirements.txt` gives compatible ranges, not a complete reproducible lock. The table records the tested local versions. Choose a PyTorch build appropriate to the machine; this host's ROCm build is not a portable dependency pin. A fresh installation on another machine has not yet been verified. Release packaging must record and exercise the full environment used for the actual model.

For the RTX PRO 6000 Blackwell, use a compatible NVIDIA CUDA PyTorch installation on that host and request `--device cuda`. The local ROCm installation is retained. Check [PyTorch's supported CUDA builds](https://github.com/pytorch/pytorch/blob/main/RELEASE.md) with the cluster's driver/module setup; installing the generic requirements does not by itself guarantee Blackwell compatibility. The trainer sets a default cuBLAS workspace configuration before CUDA initialization and requests deterministic operations, but equality across CPU/GPU or different library releases is not guaranteed.

Generation worker count and training CPU threads are separate controls. The example generator configuration uses 12 spawned process workers for this host's Ryzen 5 5600 (6 physical cores, 12 logical CPUs). Processes allow Python-heavy solver callbacks to execute across cores; full utilization or optimal throughput is not guaranteed without benchmarking. The Python configuration class retains conservative thread-based fallback defaults. PyTorch DataLoader workers are also processes, despite the name; the trainer still starts with zero loader workers and four compute threads. A future throughput benchmark should compare worker counts and native-library thread counts to avoid CPU oversubscription. No performance benchmark has been run. See [PyTorch reproducibility guidance](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/randomness.md) for limits across platforms and releases.
