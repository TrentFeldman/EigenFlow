"""Train the mixed-order baseline only when --train is explicitly provided.

Default execution prints a plan. Real training requires a completed dataset,
a new run directory and the training flag. Test trajectories are never scored
here; model selection uses validation data only.
"""

import argparse
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import GeneratorConfig
from .dataset import Normalizer, TrajectoryDataset
from .model import OrderConditionedMLP, masked_mse


def seed_worker(worker_id):
    """Seed NumPy/Python from the distinct seed PyTorch assigns each worker."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def run_epoch(model, loader, device, optimizer=None):
    """Return sample-weighted masked MSE; no optimizer means validation.

    Targets are standardized. This loss is not the release's relative
    trajectory L2 metric, which requires complete trajectory evaluation.
    """
    model.train(optimizer is not None)
    total, count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            features, target, mask = (batch[name].to(device) for name in ("features", "target", "mask"))
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss = masked_mse(model(features), target, mask)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; inspect data/scaling and optimization")
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            total += float(loss.detach()) * len(features)
            count += len(features)
    if not count:
        raise ValueError("Cannot run an epoch with an empty dataset")
    return total / count


def main(argv=None) -> int:
    """Validate a training plan or train with explicit data/run destinations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/generator.json")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--allow-failures", action="store_true", help="Opt in after inspecting failed reference solves")
    args = parser.parse_args(argv)
    config = GeneratorConfig.load(args.config)
    if (min(args.epochs, args.batch_size, args.width, args.depth, args.threads) < 1
            or args.loader_workers < 0 or not np.isfinite(args.learning_rate)
            or args.learning_rate <= 0):
        parser.error("Counts/rate must be positive; loader-workers may be zero")
    if not args.train:
        print(json.dumps({
            "mode": "dry-run; no model trained or files written",
            "architecture": {"max_order": config.max_order, "width": args.width, "depth": args.depth},
            "epochs": args.epochs, "batch_size": args.batch_size,
            "cpu_threads": args.threads, "loader_workers": args.loader_workers,
            "selection": "validation masked MSE; test set remains reserved",
        }, indent=2))
        return 0
    if args.data is None or args.run is None:
        parser.error("--train requires --data and a new --run directory")
    random.seed(config.seed)
    np.random.seed(config.seed % (2**32))
    torch.manual_seed(config.seed)
    torch.set_num_threads(args.threads)
    # Start with CPU determinism. GPU-specific reproducibility needs separate
    # validation of the installed backend and is not assumed by this baseline.
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    training = TrajectoryDataset(args.data, "train", config.max_order, args.allow_failures)
    validation = TrajectoryDataset(args.data, "validation", config.max_order, args.allow_failures)
    # Round-trip tuples to JSON lists for comparison with the stored manifest.
    if json.loads(json.dumps(asdict(config))) != training.manifest["config"]:
        raise ValueError("Training config must match the dataset generation config")
    normalizer = Normalizer.fit(training)
    training.normalizer = validation.normalizer = normalizer
    model = OrderConditionedMLP(config.max_order, args.width, args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    loader_options = {"batch_size": args.batch_size, "num_workers": args.loader_workers,
                      "worker_init_fn": seed_worker}
    if args.loader_workers:
        loader_options["multiprocessing_context"] = "spawn"
    train_loader = DataLoader(training, shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(validation, shuffle=False, **loader_options)
    args.run.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "incomplete", "config": asdict(config),
        "architecture": model.architecture, "normalizer": normalizer.to_dict(),
        "seed": config.seed, "device": str(device), "epochs": [],
        "data": str(args.data.resolve()), "dataset_manifest_sha256": sha256(
            (args.data / "manifest.json").read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ("numpy", "scipy", "torch")},
        "source_sha256": {path.name: sha256(path.read_bytes()).hexdigest()
                          for path in sorted(Path(__file__).parent.glob("*.py"))},
        "training_options": {key: str(value) if isinstance(value, Path) else value
                             for key, value in vars(args).items()},
        "reference_failures": training.manifest["failed"],
        "test_evaluated": False,
    }
    best = float("inf")
    report_path = args.run / "run.json"
    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, train_loader, device, optimizer)
            validation_loss = run_epoch(model, val_loader, device)
            report["epochs"].append({"epoch": epoch, "train_mse": train_loss, "validation_mse": validation_loss})
            if validation_loss < best:
                best = validation_loss
                checkpoint_path = args.run / "best.pt"
                temporary_path = args.run / "best.pending.pt"
                torch.save({
                    "schema_version": 1, "encoding": "linear-padded-v1",
                    "architecture": model.architecture, "state_dict": model.state_dict(),
                    "normalizer": normalizer.to_dict(), "epoch": epoch,
                    "config": asdict(config), "validation_mse": best,
                    "dataset_manifest_sha256": report["dataset_manifest_sha256"],
                }, temporary_path)
                temporary_path.replace(checkpoint_path)
                report["best_epoch"] = epoch
            report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        report["status"] = "complete"
        report["checkpoint_sha256"] = sha256((args.run / "best.pt").read_bytes()).hexdigest()
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
