# 384-case CPU smoke — October 16, 2026

Question: does generate → train → independent checkpoint prediction → evaluate →
plot work on real reference data? **Yes**, with accuracy limitations preserved.

Base revision `339fe2a6315333d576d7ac47363cf9ed51d77776` plus this session's uncommitted
implementation. `run.json` and `generation-manifest.json` retain source hashes,
resolved settings and dataset identity. Dataset: `ode/data/20261016-smoke-384`;
manifest `60295a13a76d8cea5588d97053c72cb4111bfeaa9b8757285b4d5ad817ebc3cd`. All 384 references passed in
1.454s (original 12-worker setting; later local default is six).
The resolved original recipe is saved here so that execution change is explicit.

Seed 20261015, frozen order-2/order-3 homogeneous/sine ranges, 101 times on [0,5],
256 train/64 validation/64 reserved test. No diagnostics in this smoke recipe.
3x128 MLP, five epochs, batch 256, AdamW lr 1e-3, CPU two threads, RAM preload.
Python 3.14.8, NumPy 2.5.2, SciPy 1.18.1, PyTorch 2.14.1+rocm7.14; Ryzen 5 5600.
Training took 1.630s. Epoch 5 selected at
54.247% mean validation relative L2: **5% target
not met**. Test was not evaluated. This tests pipeline wiring, not convergence.

Artifacts: `run.json`, `evaluation.json`, `prediction.json`, `reload-check.json`,
`plots/`, and local gitignored `best.pt` (SHA-256 `a36e287ffa7d619428be156eea61dc9e7e3782d1deefde45fadafe768576ea13`).
Reloads are repeatable; CLI/API predictions agree. Plot includes family median
and worst validation cases, with separate state/error panels per derivative.

Decision: retain the working pipeline and run the separately documented larger
pilot. Reproduce with the saved recipe, `--train --epochs 5 --batch-size 256
--threads 2 --preload` and fresh dataset/run names. `ode/README.md` gives the
complete generation/evaluation/prediction/plot commands. No release claim.
