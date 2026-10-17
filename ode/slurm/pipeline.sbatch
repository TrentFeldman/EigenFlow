#!/usr/bin/env bash
# Run the full ODE pipeline (generate -> train -> evaluate -> plot) as one
# SLURM job on FSU's HPC. Submit from the repository root:
#
#   sbatch ode/slurm/pipeline.sbatch --name 20261101-full-01 --preset full --epochs 300
#
# Everything after the script name goes to `python -m ode pipeline`. If the
# job hits its time limit or is preempted, submit the SAME command again: the
# pipeline skips finished stages and training resumes from last.pt.
#
# Before the first submission, check the values marked CHANGE ME against
# `sinfo -s` and FSU RCC's documentation (partition names and GPU types are
# site-specific, so none are guessed here). Worker and thread counts need no
# editing: they follow --cpus-per-task automatically.
#
#SBATCH --job-name=eigenflow-ode
#SBATCH --partition=CHANGE_ME          # a GPU partition you are allowed to use
#SBATCH --gres=gpu:1                   # one GPU; the model is far too small for more
#SBATCH --cpus-per-task=16             # generation uses all of these in parallel
#SBATCH --mem=32G                      # full preset preloads ~1.1 GiB; headroom for generation
#SBATCH --time=04:00:00
#SBATCH --output=ode/runs/slurm-%j.log # progress lines and errors land here

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

# CHANGE ME: load whatever gives you Python 3.14 + the CUDA PyTorch build,
# e.g. `module load cuda` and a venv created per ode/docs/ENVIRONMENT.md.
source .venv/bin/activate

echo "job ${SLURM_JOB_ID:-local} on $(hostname): ${SLURM_CPUS_PER_TASK:-?} CPUs, GPU ${CUDA_VISIBLE_DEVICES:-none}"
python -m ode check || echo "environment check reported a problem; see above"
python -m ode pipeline "$@"
