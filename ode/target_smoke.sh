#!/usr/bin/env bash
# Run from repo root INSIDE an allocated NVIDIA GPU job. No scheduler request
# is guessed here. Arguments: new run ID, editable resource-profile JSON.
# Activate the cluster's tested CUDA environment first; never use local ROCm.
set -euo pipefail
run_id=${1:?Provide a unique run ID}
resource_profile=${2:-ode/configs/resources_cuda.example.json}
if [[ ! "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo 'Run ID must be a simple directory name.' >&2
    exit 2
fi
run_dir="ode/runs/$run_id"
data_dir="ode/data/$run_id"
mkdir "$run_dir"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m ode.check_environment --require-cuda --output "$run_dir/environment.json"
python -m pip freeze > "$run_dir/requirements-resolved.txt"
if command -v nvidia-smi >/dev/null; then
    nvidia-smi > "$run_dir/nvidia-smi.txt"
fi
if command -v scontrol >/dev/null && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol show job "$SLURM_JOB_ID" > "$run_dir/allocation.txt"
fi
python -m ode.generate_data --generate --resources "$resource_profile" --output "$data_dir" > "$run_dir/generation.json"
python -m ode.train --train --data "$data_dir" --run "$run_dir/train" --epochs 5 --resources "$resource_profile" --device cuda > "$run_dir/training.log"
python -m ode.predict --checkpoint "$run_dir/train/best.pt" --input ode/configs/predict_example.json --output "$run_dir/prediction.json" --device cuda
python -m ode.evaluate --checkpoint "$run_dir/train/best.pt" --data "$data_dir" --output "$run_dir/evaluation.json" --device cuda
python -m ode.benchmark inference --checkpoint "$run_dir/train/best.pt" --device cuda --repeats 30 --output "$run_dir/runtime.json"
