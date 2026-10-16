# Prepared commands — not executed

These commands are written for a future authorized experiment. This task only
reviews saved records and prepares the plan. There is no automatic runner.
Choose new run names if these paths ever exist. Read README.md and experiments.json
before starting any phase.

The first longer constant-learning-rate control uses existing CLI options:

```bash
.venv/bin/python -m ode.train \
  --config ode/configs/generator_pilot.json \
  --resources ode/configs/resources_local.json \
  --data ode/data/20261016-pilot-6656 \
  --run ode/runs/20261016-error-reduction-O0-constant-300 \
  --width 128 --depth 3 --epochs 300 --learning-rate 0.001 \
  --batch-size 1024 --train
```

This starts from scratch with the original data seed and model seed, while
changing only the epoch budget relative to the recorded baseline. It does not
resume the old checkpoint. AdamW's installed weight-decay default is 0.01;
future trainer changes must make that explicit before claiming exact controls.

For a confirmed target CUDA environment, the execution-profile substitution is:

```text
--resources <edited-copy-of-resources_cuda.example.json> --device cuda --batch-size 1024
```

Keep the explicit batch-size override for matched accuracy comparisons. Edit
workers/threads/RAM limits for the allocation; generation workers have no role in
this training-only comparison. CPU/GPU results may differ and must be recorded.

No command is supplied for the scheduler, relative loss, new feature encoding,
independent optimization seeds, or external final test: today's CLI does not
implement them. Their draft settings and prerequisite changes are recorded in
experiments.json. Implement and check the capability for the selected phase,
then write its actual reproduction command. Do not pass experiments.json to
ode.train, ode.generate_data, or the resource-profile loader.

The initial two-run block is a proposed roughly 37.5-minute local CPU budget,
plus setup, based on prior timing. It has not been launched or hardware-reserved.
A later width comparison changes only --width against its selected matched parent;
it is not permission to combine several untested changes into one run.
