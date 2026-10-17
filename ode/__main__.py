"""Single entry point: `python -m ode <command> [options]` from the repo root.

Each command forwards its options to one module's own CLI, so
`python -m ode train --help` shows every training option. `pipeline` chains
generate -> train -> evaluate -> plot and is safe to rerun after a crash or a
SLURM time limit: finished stages are skipped and training resumes.
"""

from importlib import import_module
import sys


# command: (module inside ode/, one-line description shown by --help)
COMMANDS = {
    "pipeline": ("pipeline", "Generate data, train, evaluate and plot in one resumable run"),
    "generate": ("generation.generate_data", "Solve ODEs numerically and write a dataset"),
    "validate": ("generation.validate_reference", "Check the numerical solver against exact formulas"),
    "train": ("training.train", "Train the neural network on a dataset"),
    "evaluate": ("evaluation.evaluate", "Score a trained model on a dataset split"),
    "plot": ("evaluation.plot_results", "Draw charts for a trained, evaluated run"),
    "diagnose": ("evaluation.diagnose", "Break validation error down by regime and residual"),
    "predict": ("evaluation.predict", "Predict one equation from a JSON request"),
    "runs": ("tools.runs", "List trained runs and datasets with their scores and sizes"),
    "neurons": ("tools.neurons", "Animate the neurons changing during training (needs --snapshot-every)"),
    "check": ("tools.check_environment", "Report Python, packages and GPU visibility"),
    "benchmark": ("tools.benchmark", "Time generation or inference on this machine"),
}


def main(argv=None):
    """Dispatch to a command's main(); print the command list on misuse."""
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print("usage: python -m ode <command> [options]   (add --help after a command)\n")
        for name, (_, description) in COMMANDS.items():
            print(f"  {name:<10} {description}")
        return 0 if not argv or argv[0] in ("-h", "--help") else 2
    module, _ = COMMANDS[argv[0]]
    sys.argv[0] = f"python -m ode {argv[0]}"
    return import_module(f".{module}", __package__).main(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
