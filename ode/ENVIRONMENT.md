# Python environment

Use the existing root `.venv`. The current request authorizes checking it and adding the missing SciPy dependency; the installed PyTorch/ROCm stack was retained. No system Python packages were changed.

Verified local runtime:

| Component | Version / state |
|---|---|
| Python | 3.14.8, isolated virtual environment |
| NumPy | 2.5.2 |
| SciPy | 1.18.1 |
| PyTorch | 2.14.1+rocm7.14 |
| GPU visible to PyTorch | No; the initial trainer uses CPU |

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

Generation worker count and training CPU threads are separate controls. PyTorch DataLoader workers are processes, despite the name. Start with zero loader workers and four generation threads. A future throughput benchmark should compare threads, processes and native-library thread counts to avoid CPU oversubscription. No performance benchmark has been run. See [PyTorch reproducibility guidance](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/randomness.md) for limits across platforms and releases.
