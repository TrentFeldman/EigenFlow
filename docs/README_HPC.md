# README_HPC: running simulations on FSU's supercomputer (RCC)

These are personal notes; git ignores this file and it isn't part of Sussman's repo. For how the simulation code itself works, see `README_LOCAL.md`.

> **The golden rule: we never edit Sussman's code.** We only build it, feed it an inputs file, and change settings on the command line. If something seems to need a code change, ask Sussman.

**Status (Nov 2026):**
- **Account works.** User `USERNAME`, sponsored by Sussman.
- **The code builds on RCC.** Sussman commits from RCC's own login nodes.
- **These job scripts haven't been run on RCC yet.** The restart logic and command-line settings were checked on the laptop.

---

## Words you'll see

| Word | Meaning |
|---|---|
| **RCC** | FSU's Research Computing Center: the supercomputer ("the cluster"). |
| **Login node** | The machine you land on when you `ssh` in. Use it only for editing files and submitting jobs, **never for running simulations**. |
| **Compute node** | The machines that do the work. You don't log into them; Slurm runs your job there. |
| **Slurm** | The queue manager. You give it a *job script*, it waits until cores are free, then runs it. |
| **Job / job ID** | One submitted script. `sbatch` prints its number (job ID); you use it to watch or cancel the job. |
| **Queue / account (`-A`)** | Which line you wait in. Ours: `genacc_q` (long runs), `backfill` (≤4 h), `quicktest` (≤10 min). |
| **Cores / ranks / `-n`** | How many CPU cores the job gets. The code splits the domain into pieces and each MPI *rank* (one per core) owns some. |
| **AMReX** | The library the code is built on. It handles the grid and **AMR** (adaptive mesh refinement: fine cells only where interesting things happen, like the water-air interface). |
| **Inputs file** | A text file of `key = value` settings that defines the problem. Any setting can also be overridden on the command line as `key=value`. |
| **Checkpoint (`chkNNNNN/`)** | A save point. If a job dies or hits its time limit, the next job continues from here. |
| **Plotfile (`nddataPLT…/`)** | A snapshot of the flow for visualization. |

---

## Step 0: one-time setup

1. **Be on the FSU network.** Off campus, connect to the FSU VPN first; SSH won't work otherwise.
2. **Log in** (password login; there's no SSH key set up):
   ```bash
   ssh USERNAME@hpc-login.rcc.fsu.edu
   ```
3. **Storage.**
   - **Home** (`/gpfs/home/USERNAME`) has a 150 GB quota. Check usage with `gpfs_quota scratch`. The code plus a few runs fit; a big training set doesn't.
   - **Scratch** (`/gpfs/research/scratch/USERNAME`) is unlimited but gets wiped periodically and is never backed up. The account doesn't have it yet; request it with an RCC support ticket. It's only needed for big 3D sweeps.
   - The account is only in the `genacc` group: **general queues only, no group storage.**

## Step 1: download and build the code (once, and again only when you deliberately update)

```bash
cd ~ && git clone https://github.com/msussman42/amrex_implicit_interfaces

# Building is real work, so do it on a compute node, not the login node.
# This asks Slurm for an interactive shell with 8 cores for 1 hour:
srun --pty -t 1:00:00 -n 1 -c 8 -A genacc_q /bin/bash

module load gnu/13 openmpi/4.1.6          # compiler + MPI (confirmed available)
cd ~/amrex_implicit_interfaces/NS_fluids_lib
make -j8 DIM=2 USE_MPI=TRUE USE_OMP=FALSE  # makes amr2d.gnu.MPI.ex  (~10-20 min)
make -j8 DIM=3 USE_MPI=TRUE USE_OMP=FALSE  # makes amr3d.gnu.MPI.ex  (only if doing 3D)
exit                                       # give the cores back
```

**What you should see:** `ls ~/amrex_implicit_interfaces/NS_fluids_lib/*.ex` lists the executable(s).

- **All the build settings are passed on the `make` line.** No files get edited.
- **Job scripts must load the same two modules** you built with.
- **Don't `git pull` and rebuild in the middle of a set of runs.** Sussman pushes almost daily. Every run directory records the code version in `code_version.txt`.
- **If the build fails with `stdlib.h: No such file or directory`:** that happened on the Arch laptop, not on RCC. Apply the same one-line `Make.unknown` patch the laptop uses (see `README_LOCAL.md`, Quick start). It's the only allowed edit.

---

## Step 2: the big run, high-res Rayleigh–Taylor (2D)

**What it is:** a layer of water sitting on top of air in a tall 8 cm × 32 cm box, under gravity. The flat surface has a tiny ripple (0.08 mm). Heavy-over-light is unstable, so the ripple grows into a falling water "spike" and a rising air "bubble" that crashes into the floor. This is the classic Rayleigh–Taylor instability.

It uses Sussman's stock `run2d/inputs.rayleigh_taylor` (probtype 602) unchanged. Only these settings are overridden from the job script:

| Setting | Stock | Our run | Why |
|---|---|---|---|
| `amr.n_cell` | 32 128 | **64 256** | 2× finer base grid |
| `amr.max_level` | 2 | **3** | One more refinement level → finest cell 0.16 mm (stock 0.63 mm), like a 512×2048 grid but only near the interface |
| `amr.max_grid_size` | 1024 | **32** | Cuts the grid into enough pieces to keep 32 cores busy |
| `DistributionMapping.strategy` | SFC | **KNAPSACK** | The code aborts if any core owns zero grid pieces on any level (`cannot have a dormant/idle processor on any level`). The default SFC assignment can leave a core empty; knapsack never does. Our first run died this way at step 5217 |
| `stop_time` | 10 s | **0.7 s** | The interesting part is over by ~0.5 s |
| plot output | every 100 steps | **every 5 ms of sim time** (`amr.plot_per=0.005`) | Evenly spaced frames (~140) for a movie |
| `amr.check_int` | 100 | **500** | Fewer, still regular, save points |
| `ns.visual_nddata_format` | 0 (ASCII) | **1** (binary AMReX) | Much faster to write, smaller, opens in yt/ParaView/VisIt |

**Expected cost (a rough estimate, the pilot replaces it):**
- **Time step:** ~5e-5 s, limited by surface tension and speed at the finest level, so ~14,000 steps.
- **Cells:** ~30k at the start, growing to ~100k+ as the interface tangles.
- **Wall time:** ~400 core-hours, roughly **15–30 h** on 32 cores.
- **Disk:** ~10–15 GB.

The job is set to 32 cores and a 3-day limit.

### 2a. Create the job folder ON RCC

Do this logged into RCC. Running `scp` on RCC fails because the laptop copy isn't there. The folder `~/rt2d_hires/` gets:
- `inputs`: a copy of `inputs.rayleigh_taylor`. Don't edit it; overrides go in the script.
- `run_rt.sh`: the Slurm job script. It's the same as the commented one at `~/hpc_jobs/rt2d_hires/run_rt.sh` on the laptop.
- `logs/`: where Slurm writes job logs. It **must exist** before you submit, or the job dies instantly.

```bash
mkdir -p ~/rt2d_hires/logs && cd ~/rt2d_hires
cp ~/amrex_implicit_interfaces/NS_fluids_lib/run2d/inputs.rayleigh_taylor inputs
cat > run_rt.sh <<'EOF'
#!/bin/bash
#SBATCH -J rt2d_hires
#SBATCH -A genacc_q
#SBATCH -n 32
#SBATCH -t 2-00:00:00
#SBATCH -o logs/%j.out
#SBATCH --mail-type=END,FAIL
# One job on genacc_q, ~45 h estimated (pilot: 6.5 s/step on 32 cores).
# If it hits the time limit, submit it again: it continues from the last good checkpoint.

module load gnu/13 openmpi/4.1.6
EXE=$HOME/amrex_implicit_interfaces/NS_fluids_lib/amr2d.gnu.MPI.ex
HIRES='amr.n_cell=64 256 amr.max_level=3 amr.max_grid_size=32
       DistributionMapping.strategy=KNAPSACK
       stop_time=0.7
       amr.plot_int=-1 amr.plot_per=0.005 amr.check_int=200
       ns.visual_nddata_format=1'

d=$SLURM_SUBMIT_DIR/run
mkdir -p "$d" && cd "$d"
if [ -f DONE ]; then echo "run already finished"; scancel -u "$USER" -n "$SLURM_JOB_NAME" -t PENDING; exit 0; fi
[ -f inputs ] || cp "$SLURM_SUBMIT_DIR/inputs" .
git -C "$HOME/amrex_implicit_interfaces" rev-parse HEAD > code_version.txt

# Pick the newest checkpoint that was fully written: the run got to the next
# step after it, or the run ended cleanly right after it. A checkpoint being
# written when the 4 h limit hit is half-finished and gets skipped.
all=$(ls -d chk[0-9][0-9][0-9][0-9][0-9] 2>/dev/null | sort)
newest=$(echo "$all" | tail -n 1)
last=
for c in $(echo "$all" | sort -r); do
  n=$((10#${c#chk}))
  if grep -q "^STEP = $((n+1)) " run.out 2>/dev/null ||
     { [ "$c" = "$newest" ] && tail -n 1 run.out 2>/dev/null | grep -q finalized; }; then
    last=$c; break
  fi
done
if [ -n "$last" ]; then
  # delete half-written ones newer than $last, and old ones except the one just before $last
  for c in $all; do [[ "$c" > "$last" ]] && rm -rf "$c"; done
  echo "$all" | awk -v L="$last" '$0<L' | head -n -1 | xargs -r rm -rf
fi
echo "job $SLURM_JOB_ID restarting from: ${last:-the beginning}" | tee -a restarts.log

start=$(date +%s)
srun "$EXE" inputs $HIRES $EXTRA ${last:+amr.restart=$last} >> run.out 2>> run.err
rc=$?
if [ $rc -eq 0 ] && [ -z "$EXTRA" ]; then
  touch DONE; echo "finished"
  scancel -u "$USER" -n "$SLURM_JOB_NAME" -t PENDING
elif [ $rc -ne 0 ] && [ $(( $(date +%s) - start )) -lt 600 ]; then
  echo "crashed within 10 min (exit $rc): cancelling the queued chunks, see run/run.err"
  scancel -u "$USER" -n "$SLURM_JOB_NAME" -t PENDING
fi
EOF
chmod +x run_rt.sh
ls ~/amrex_implicit_interfaces/NS_fluids_lib/amr2d.gnu.MPI.ex   # must exist; if not, do Step 1
```

### 2b. Smoke test: does it start at all? (≤10 min)

```bash
# on RCC
cd ~/rt2d_hires
EXTRA=max_step=2 sbatch -A quicktest -t 00:10:00 -n 8 run_rt.sh
squeue --me                    # ST column: PD = waiting, R = running; gone = finished
```
When it's gone from `squeue`:
```bash
grep "^STEP = " run/run.out    # want: STEP = 1 and STEP = 2
tail -n 3 run/run.out          # want: a line ending in "finalized"
tail run/run.err               # errors show up here
rm -rf run                     # IMPORTANT: otherwise the next job "restarts" from this test
```

### 2c. Pilot: how fast is it? (~1 h)

```bash
EXTRA=max_step=200 sbatch -t 02:00:00 run_rt.sh
# when done:
grep "^STEP = " run/run.out | tail -n 2     # how far it got and the time step size
seff <jobid>                                # how busy the cores were; max memory
```
**Time per step** = job run time (from `seff`) ÷ 200. Multiply by ~14,000 for the full run, and set `-t` in `run_rt.sh` to about 1.5× that. Then `rm -rf run` again.

### 2d. The real run: one job, ~2 days

**Pilot result** (2026-11-19): 200 steps on 32 cores took 21.5 min, so ~6.5 s/step. The time step was 2.3e-5 s and still ramping up.
**Estimate:** ~18,000 steps at ~9 s each (steps get slower as the interface stretches), so **~45 h**.

```bash
cd ~/rt2d_hires
sbatch -A genacc_q -t 2-00:00:00 -n 32 run_rt.sh
```

- **Why not ask for 5 days?** Long requests wait longer in the queue. A tight limit is safe because:
  - running out isn't a disaster: `sbatch` the same line again and it continues from the newest fully written checkpoint (one every 200 steps, ~30 min);
  - a half-written checkpoint from a killed job is detected and skipped.
- **Progress:** `grep "^STEP = " run/run.out | tail -n 1`. TIME goes to 0.7.
- **Finished** when `run/DONE` exists. You also get an email.
- **Alternative if the queue is slow:** `backfill` in 4-h pieces (`sbatch -A backfill -t 4:00:00 run_rt.sh`, resubmit each time it ends).

---

## Step 3: watching, cancelling, fixing

| I want to… | Command |
|---|---|
| See my jobs | `squeue --me` |
| Guess when it starts | `squeue --start --me` |
| See progress | `grep "^STEP = " ~/rt2d_hires/run/run.out \| tail -n 1` |
| Cancel a job | `scancel <jobid>` (all mine: `scancel -u $USER`) |
| See a finished job's result | `sacct -j <jobid> --format=JobID%20,State,Elapsed,ExitCode,MaxRSS` |
| See how efficient it was | `seff <jobid>` |
| Find why it crashed | `tail run/run.err` (look for `amrex::Error::…`), and `tail run/run.out` (Fortran prints `… invalid` then `stop`) |
| Get a shell inside a running job | `srun --pty --overlap --jobid <jobid> /bin/bash` |

| Symptom | Likely cause / fix |
|---|---|
| Job vanishes instantly, no `run/` | `logs/` folder missing → `mkdir logs` |
| `module: command not found` or `.ex: No such file` | Build didn't finish (Step 1), or the modules differ from the build |
| Run restarts from an old state | A leftover `run/` from a test → delete it before the real run |
| Stuck in `PD` for a long time | Busy queue. Fewer cores or a shorter `-t` start sooner, or use `backfill` |
| `Disk quota exceeded` | Home is full (150 GB). Copy results off (Step 4) and delete them on RCC |

## The queues we can use

| `-A` | Max time | Max cores per job | Use for |
|---|---|---|---|
| `genacc_q` (default) | 14 days | 400 | Real runs |
| `backfill`, `backfill2` | 4 h | 512 | Short runs, or long runs in 4-h pieces (they restart from checkpoints) |
| `quicktest` | 10 min | 8 | "Does my script even work?" |
| `condor` | 90 days | 8 | Not usable: no MPI |

- **Memory:** ~1.9 GB per core by default, which is plenty.
- **Queue wait:** RCC says "a few minutes" on average, but it can be days.
- **Command-line `sbatch` options beat the `#SBATCH` lines** inside the script.

---

## Step 4: getting results back

Use the transfer host `export.rcc.fsu.edu`, not the login node. Run this from the laptop:

```bash
rsync -rtv --info=progress2 USERNAME@export.rcc.fsu.edu:rt2d_hires/ /mnt/cold/rt2d_hires/
```

- **`-rt`, not `-a`.** exFAT/NTFS drives can't store Unix permissions, and `-a` errors on them. On ext4 or btrfs, `-a` is fine.
- **Delete from RCC after copying.** Home is small, and scratch gets wiped.
- **Very large transfers:** RCC recommends Globus.

To look at the results, see "Outputs" in `README_LOCAL.md`. The quickest way is `yt.load("run/nddataPLT00000100")` in Python.

---

## Appendix: many runs at once (parameter sweeps)

To make a training set (e.g. 50 cone runs with different angles), use a **job array**: one script, Slurm runs it N times, and each copy reads its own line of `params.txt`.

```
~/sweeps/cone2d/
  inputs          # copy of the stock inputs file
  params.txt      # line 1 = settings for run 0, line 2 = run 1, ...
  run_sweep.sh    # script below
  logs/           # must exist
  run_000/ run_001/ ...   # created by the jobs
```

Example `params.txt` (cone knobs are explained in `README_LOCAL.md`):
```
ns.axis_dir=0 ns.radblob=0.174533 ns.adv_vel=0.2143
ns.axis_dir=0 ns.radblob=0.261799 ns.adv_vel=0.2143
ns.axis_dir=2 ns.radblob3=0.1 ns.xblob3=-0.5 ns.yblob3=0.5 ns.xblob10=0.2143
```

`run_sweep.sh`:
```bash
#!/bin/bash
#SBATCH -J sweep
#SBATCH -A genacc_q
#SBATCH -n 6
#SBATCH -t 1-00:00:00
#SBATCH -a 0-49%10          # runs 0..49, at most 10 at the same time
#SBATCH -o logs/%A_%a.out
#SBATCH --mail-type=END,FAIL

module load gnu/13 openmpi/4.1.6
EXE=$HOME/amrex_implicit_interfaces/NS_fluids_lib/amr2d.gnu.MPI.ex
# the cone inputs ship with slow "spectral" mode on; these turn it off (see README_LOCAL.md)
FAST='ns.enable_spectral=0 amr.time_blocking_factor=1 amr.space_blocking_factor=1 1 1 1 ns.visual_nddata_format=1'

P=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$SLURM_SUBMIT_DIR/params.txt")
d=$SLURM_SUBMIT_DIR/run_$(printf %03d "$SLURM_ARRAY_TASK_ID")
mkdir -p "$d" && cd "$d"
[ -f inputs ] || cp "$SLURM_SUBMIT_DIR/inputs" .
echo "$P" > params_used.txt
git -C "$HOME/amrex_implicit_interfaces" rev-parse HEAD > code_version.txt

last=$(ls -d chk[0-9][0-9][0-9][0-9][0-9] 2>/dev/null | sort | tail -n 1)
srun "$EXE" inputs $FAST $P $EXTRA ${last:+amr.restart=$last} >> run.out 2>> run.err
```

Use the same flow as Step 2:
1. **Smoke test:** `EXTRA=max_step=2 sbatch -A quicktest -t 00:10:00 -a 0 run_sweep.sh`, then `rm -rf run_000`.
2. **Pilot:** `sbatch -a 0 run_sweep.sh`.
3. **Everything:** `sbatch run_sweep.sh`.

Then:
- **Resubmit only the runs that died:** `sbatch -a 7,12 run_sweep.sh`.
- **List the runs that didn't finish:** `grep -L finalized run_*/run.out`.

**Cost guide.** These are laptop numbers; RCC's older Xeons are ~1.3–1.6× slower, and the pilot is the real answer.

| Run | Laptop (6 cores) | RCC (est.) |
|---|---|---|
| cone 2D, fast | ~10 min | ~15 min |
| cone 3D, fast | ~4 h | ~5–6 h |
| boiling 2D-RZ | ~10–11 h | ~14–17 h |

**Storage to budget:**
- 300 2D runs × 150 snapshots ≈ 200 GB.
- 20 3D runs ≈ 400 GB.
- Neither fits on RCC (no group storage), so keep the full set on the local cold-storage drive.
