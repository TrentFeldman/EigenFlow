# Execution-profile reproducibility — October 16, 2026

Question: do the new portable resource profiles preserve the already measured
CPU training path when resolved options match? Base revision `339fe2a6315333d576d7ac47363cf9ed51d77776`
plus this session's uncommitted implementation. Source/config/environment hashes
and options are in run.json. This reuses the 384-case smoke dataset with seed
20261015; no new dataset or test evaluation was performed.

Executed `python -m ode.train --resources ode/configs/resources_local.json
--batch-size 256 --train --data ode/data/20261016-smoke-384
--run ode/runs/20261016-resource-smoke-01 --epochs 5` with the local .venv.
All selected checkpoint tensors are **bitwise identical** to the original
five-epoch smoke; see reproducibility.json. This confirms CLI precedence and
real preloading/training after profile integration. The same 54.25% validation
limitation applies; this is not an additional accuracy candidate.

Decision: keep profiles, explicit overrides and the preload array-memory cap.
Actual GPU profile values remain editable and unverified until allocated access.
