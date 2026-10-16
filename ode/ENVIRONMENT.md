# Python environment

Use the existing root `.venv`; this session inspected it without installing or
replacing packages. The tested CPU runtime is Python **3.14.8**, NumPy **2.5.2**,
SciPy **1.18.1**, PyTorch **2.14.1+rocm7.14**. No GPU is visible locally.
`.python-version` records the interpreter; `requirements.txt` pins numerical
packages and the public PyTorch version, while `requirements-local.txt` also pins
the exact local ROCm suffix. These are direct-dependency pins, not a portable
wheel archive or a lock of every transitive dependency.

```bash
.venv/bin/python -m ode.check_environment
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -v
```

For a new machine, create a separate environment using Python 3.14.8 if available.
Install the appropriate CPU/CUDA/ROCm PyTorch build from its official source,
then install `ode/requirements.txt` and run the checks. Use
`ode/requirements-local.txt` only with a source that supplies that exact local
ROCm wheel; do not assume it exists on the default package index. A new clone
installation has not been tested in this session. Do not replace this local
ROCm environment to prepare the NVIDIA machine.

For NVIDIA choose a build matching the target driver, GPU and interpreter using
[PyTorch's official installer](https://pytorch.org/get-started/locally/). The
public version pin does not prove that a compatible CUDA wheel is available.
If the cluster cannot supply that version/Python combination, record and test a
separate target environment instead of silently claiming the local pins apply.
The CUDA API is also used by ROCm; check build provenance as well as availability.

```bash
# Inside an allocated NVIDIA job, using its activated environment:
python -m ode.check_environment --require-cuda --output environment.json
bash ode/target_smoke.sh unique-target-run-id ode/configs/resources_cuda.example.json
```

Edit the resource profile for the allocation; workers may be explicit or auto. The script
records `pip freeze`, driver details and scheduler evidence, exercises model
training and reload, and benchmarks inference. Actual target execution remains
blocked on access/allocation. Local failure evidence is in
[runs/20261016-environment-01](runs/20261016-environment-01/cuda-readiness.json).

Generation defaults to six process workers after a measured 1/6/12 comparison.
Set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` for generation
so each process does not start its own large native thread pool. Training CPU
threads, DataLoader workers and solver workers are separate controls. CPU pilot
training uses two threads, zero loader workers and host-RAM preloading.
Plotting exports SVG directly and adds no package dependency.
