"""Run the whole ODE workflow with one command that is safe to rerun.

    python -m ode pipeline --name 20261017-pilot-02 --preset pilot --epochs 300

Stages run in order. A stage whose output already exists is skipped, so
rerunning the exact same command after a crash, Ctrl-C or a job time limit
continues where the last attempt stopped:

  1. generate  ode/data/<preset>-<count>/         solve the ODEs (shared by runs)
  2. train     ode/runs/<name>/best.pt, run.json   resumes from last.pt
  3. evaluate  ode/runs/<name>/evaluation.json     validation splits only
  4. plot      ode/runs/<name>/report.svg

The test split is never scored here; it stays sealed for the final release.
Options this command does not know are passed to `train` unchanged (see
`python -m ode train --help`), e.g. --epochs, --batch-size, --resources.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .config import CONFIG_DIR, DIAGNOSTIC_SPLITS, GeneratorConfig


ROOT = Path(__file__).parent
PRESETS = {"small": CONFIG_DIR / "generator.json", "pilot": CONFIG_DIR / "generator_pilot.json",
           "full": CONFIG_DIR / "generator_full.json", "wide": CONFIG_DIR / "generator_wide.json",
           "final": CONFIG_DIR / "generator_final.json"}


def _stage(number, name, skipped=None):
    """Print a stage banner; return True when the stage should run."""
    print(f"\n== {number}/4 {name}" + (f": skipped, {skipped}" if skipped else ""), flush=True)
    return skipped is None


def main(argv=None):
    """Run every unfinished stage; return 0 on success, 1 if generation had failures."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="Run folder under ode/runs/, e.g. 20261017-pilot-02")
    parser.add_argument("--preset", default="pilot",
                        help="small (384 equations), pilot (6,656), full (100,032), wide (480,000, wider "
                             "damping/frequency), final (824,960, orders 1-4, decay to 3, frequency 0.25-8, times 0-10), or a generator JSON path")
    parser.add_argument("--data", type=Path, help="Dataset folder to create or reuse; default ode/data/<preset>-<count>")
    args, train_args = parser.parse_known_args(argv)

    config_path = PRESETS.get(args.preset, Path(args.preset))
    config = GeneratorConfig.load(config_path)
    label = args.preset if args.preset in PRESETS else config_path.stem
    data = args.data or ROOT / "data" / f"{label}-{config.trajectory_count}"
    run = ROOT / "runs" / args.name
    manifest_path = data / "manifest.json"

    # Imported here so `--help` stays instant (torch is slow to import).
    from .generation.generate_data import main as generate
    from .training.train import main as train
    from .evaluation.evaluate import main as evaluate
    from .evaluation.plot_results import main as plot

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] == "incomplete":
            raise SystemExit(f"{data} is a partial dataset from an interrupted run; delete it and rerun")
        existing = GeneratorConfig.from_dict(manifest["config"])
        if replace(existing, workers=config.workers, backend=config.backend) != config:
            raise SystemExit(f"{data} was generated with different settings than preset {args.preset!r}")
        _stage(1, "generate", f"{data} already exists")
    elif _stage(1, "generate") and generate(["--generate", "--config", str(config_path), "--output", str(data)]):
        print("Some equations failed to solve; inspect records.jsonl before training (train --allow-failures).")
        return 1

    run_record = run / "run.json"
    status = json.loads(run_record.read_text())["status"] if run_record.exists() else None
    if status == "complete":
        _stage(2, "train", f"{run} already finished (raise --epochs and use train --resume to extend)")
    elif _stage(2, "train"):
        resume = (run / "last.pt").exists()
        if run.exists() and not resume:
            raise SystemExit(f"{run} exists but has no last.pt to resume from; pick a new --name")
        train(["--train", "--data", str(data), "--run", str(run), *train_args, *(["--resume"] if resume else [])])

    if (run / "evaluation.json").exists():
        _stage(3, "evaluate", "evaluation.json exists")
    elif _stage(3, "evaluate"):
        import torch
        splits = ["validation", *(DIAGNOSTIC_SPLITS if config.diagnostic_groups else ())]
        evaluate(["--checkpoint", str(run / "best.pt"), "--data", str(data), "--splits", *splits,
                  "--output", str(run / "evaluation.json"),
                  "--device", "cuda" if torch.cuda.is_available() else "cpu"])

    if (run / "report.svg").exists():
        _stage(4, "plot", "report.svg exists")
    elif _stage(4, "plot"):
        plot(["--run", str(run)])

    report = json.loads(run_record.read_text())
    print(f"\nDone. Best validation error {report['best_validation_relative_l2']:.2%} at epoch "
          f"{report['best_epoch']} of {len(report['epochs'])}.\n"
          f"  model:  {run / 'best.pt'}\n  scores: {run / 'evaluation.json'}\n  chart:  {run / 'report.svg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
