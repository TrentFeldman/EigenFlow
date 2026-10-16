"""Train the mixed-order trajectory model and select weights on validation data.

The trainer loads generated splits, fits training-only normalization, batches
examples into the model, and saves weights plus preprocessing for prediction.
The CLI prints a plan unless --train requests execution with data/run paths.
"""

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import random
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import GeneratorConfig, provenance
from .dataset import Normalizer, TrajectoryDataset, feature_width
from .model import OrderConditionedMLP, masked_mse
from .evaluate import METRIC, evaluate_model
from .resources import TRAINING_KEYS, load_resources


# Bound the combined gradient norm to limit unusually large optimizer updates.
MAX_GRADIENT_NORM = 1.0


def resolve_device(requested: str) -> torch.device:
    """Resolve the requested CPU/CUDA policy to a PyTorch device.

    Args:
        requested: 'cpu', 'cuda' or 'auto'.
    Returns:
        CPU device, or CUDA when available and requested by cuda/auto.
    Raises:
        ValueError: The policy is unknown.
        RuntimeError: CUDA is explicitly requested but unavailable.
    """
    if requested not in ("cpu", "cuda", "auto"):
        raise ValueError("device must be cpu, cuda or auto")

    available = torch.cuda.is_available() if requested != "cpu" else False

    if requested == "cuda" and not available:
        raise RuntimeError("CUDA device unavailable; use a compatible GPU-host PyTorch build or --device cpu")

    return torch.device("cuda" if available else "cpu")


def seed_everything(seed):
    """Seed the training random streams and request deterministic algorithms.

    Args:
        seed: Nonnegative unsigned 64-bit seed from GeneratorConfig.
    Note: Identical seeds do not guarantee identical results across devices/releases.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    """Seed NumPy and Python from the worker seed assigned by PyTorch.

    Args:
        worker_id: DataLoader worker index; PyTorch has already set its own seed.
    """
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def run_epoch(model, loader, device, optimizer=None):
    """Compute sample-weighted masked MSE, optionally updating model weights.

    Args:
        model: OrderConditionedMLP to train or validate.
        loader: Batches of float32 features (B, feature_width(M)), target/mask (B, M).
        device: PyTorch device receiving each batch.
        optimizer: Optimizer for training; None evaluates without gradients.
    Returns:
        Float mean squared error in standardized state units.
    Raises:
        ValueError: The loader has no examples.
        RuntimeError: Loss or gradients become non-finite.
    """
    model.train(optimizer is not None)
    total, count = 0.0, 0

    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            features, target, mask = (
                batch[name].to(device, non_blocking=device.type == "cuda")
                for name in ("features", "target", "mask")
            )

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            loss = masked_mse(model(features), target, mask)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; inspect data/scaling and optimization")

            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    MAX_GRADIENT_NORM,
                    error_if_nonfinite=True,
                )
                optimizer.step()

            # Weight by actual batch size: batches of 4 and 1 count as 4:1.
            total += float(loss.detach()) * len(features)
            count += len(features)

    if not count:
        raise ValueError("Cannot run an epoch with an empty dataset")

    return total / count


def parse_args(argv=None):
    """Parse training options and validate counts, rates and required paths.

    Args:
        argv: CLI argument strings; None reads the process arguments.
    Returns:
        argparse.Namespace containing the validated CLI options.
    Raises:
        SystemExit: Options are invalid or requested training lacks data/run paths.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs/generator.json",
    )
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
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument(
        "--preload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load standardized examples into host RAM",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Opt in after inspecting failed reference solves",
    )
    parser.add_argument("--resources", type=Path, help="Execution profile; CLI overrides take precedence")
    parser.add_argument("--preload-limit-gib", type=float, help="Maximum GiB for preload arrays, excluding other process memory")
    profile = load_resources(parser.parse_known_args(argv)[0].resources)
    parser.set_defaults(**{key: value for key, value in profile.items() if key in TRAINING_KEYS})
    args = parser.parse_args(argv)

    if args.preload_limit_gib is not None and (not np.isfinite(args.preload_limit_gib) or args.preload_limit_gib <= 0):
        parser.error("--preload-limit-gib must be finite and positive")

    if args.preload and args.loader_workers:
        parser.error("Use --loader-workers 0 with --preload to avoid duplicating the dataset")

    for name in ("epochs", "batch_size", "width", "depth", "threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be a positive integer")

    if args.loader_workers < 0:
        parser.error("--loader-workers must be a nonnegative integer")

    if not np.isfinite(args.learning_rate):
        parser.error("--learning-rate must be finite")

    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")

    if args.train and (args.data is None or args.run is None):
        parser.error("--train requires --data and a new --run directory")

    return args


def _training_plan(args, config):
    """Describe model sizes and execution choices using the dry-run JSON schema."""
    return {
        "mode": "dry-run; no model trained or files written",
        "architecture": {
            "max_order": config.max_order,
            "width": args.width,
            "depth": args.depth,
        },
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "cpu_threads": args.threads,
        "loader_workers": args.loader_workers,
        "requested_device": args.device,
        "preload": args.preload,
        "preload_limit_gib": args.preload_limit_gib,
        "selection": "validation mean trajectory relative L2; test set remains reserved",
    }


def _load_datasets(args, config):
    """Check scientific settings, fit training statistics and optionally preload."""
    training = TrajectoryDataset(args.data, "train", config.max_order, args.allow_failures)
    validation = TrajectoryDataset(args.data, "validation", config.max_order, args.allow_failures)
    dataset_config = GeneratorConfig.from_dict(training.manifest["config"])

    if replace(dataset_config, backend=config.backend, workers=config.workers) != config:
        raise ValueError("Scientific training config must match the dataset generation config")

    normalizer = Normalizer.fit(training)
    training.normalizer = validation.normalizer = normalizer

    if args.preload:
        required = (len(training) + len(validation)) * (feature_width(config.max_order) + 2 * config.max_order) * 4
        if args.preload_limit_gib is not None and required > args.preload_limit_gib * 1024**3:
            raise ValueError(f"Preload arrays need {required / 1024**3:.3f} GiB; increase --preload-limit-gib or use --no-preload")
        training.preload()
        validation.preload()

    return training, validation, normalizer


def _build_loaders(training, validation, args, device, seed):
    """Shuffle training reproducibly and preserve validation example order."""
    generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.loader_workers,
        "worker_init_fn": seed_worker,
        "pin_memory": device.type == "cuda",
    }

    if args.loader_workers:
        loader_options["multiprocessing_context"] = "spawn"

    train_loader = DataLoader(training, shuffle=True, generator=generator, **loader_options)
    validation_loader = DataLoader(validation, shuffle=False, **loader_options)

    return train_loader, validation_loader


def build_report(args, config, model, normalizer, training, device):
    """Initialize the run report with reproducibility and dataset evidence.

    Args:
        args: Parsed CLI options, including data/run paths.
        config: Validated scientific configuration.
        model: Model whose architecture will be saved.
        normalizer: Training-fitted preprocessing vectors.
        training: Training split supplying the manifest failure count.
        device: Resolved PyTorch device.
    Returns:
        JSON-compatible report dict with incomplete status and an empty epoch list.
    """
    return {
        "status": "incomplete",
        "config": asdict(config),
        "architecture": model.architecture,
        "normalizer": normalizer.to_dict(),
        "seed": config.seed,
        "device": str(device),
        "epochs": [],
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor()
        ),
        "cuda_build": torch.version.cuda,
        "hip_build": torch.version.hip,
        "data": str(args.data.resolve()),
        "dataset_manifest_sha256": sha256(
            (args.data / "manifest.json").read_bytes()
        ).hexdigest(),
        **provenance(("numpy", "scipy", "torch")),
        "training_options": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "reference_failures": training.manifest["failed"],
        "test_evaluated": False,
        "metric": METRIC,
        "selection": "validation mean full-state trajectory relative L2",
    }


def save_checkpoint(run, model, normalizer, config, epoch, validation_mse, manifest_sha256, validation_relative_l2=None):
    """Atomically replace best.pt with the selected model and preprocessing.

    Args:
        run: Existing run directory as a Path.
        model: Model with float32 parameter tensors to serialize.
        normalizer: Fitted feature/target statistics saved alongside the weights.
        config: Scientific configuration for the training data.
        epoch: One-based epoch number selected by validation loss.
        validation_mse: Selected scalar validation loss.
        manifest_sha256: Dataset manifest digest identifying the training source.
        validation_relative_l2: Frozen selection metric; optional for legacy callers.
    Raises:
        OSError: The checkpoint cannot be written or replaced.
    """
    checkpoint_path = run / "best.pt"
    temporary_path = run / "best.pending.pt"
    # A complete write precedes replacement so interruption preserves the old best.
    torch.save(
        {
            "schema_version": 1,
            "encoding": "linear-padded-v1",
            "architecture": model.architecture,
            "state_dict": model.state_dict(),
            "normalizer": normalizer.to_dict(),
            "epoch": epoch,
            "config": asdict(config),
            "validation_mse": validation_mse,
            "validation_relative_l2": validation_relative_l2,
            "metric": METRIC,
            "dataset_manifest_sha256": manifest_sha256,
        },
        temporary_path,
    )
    temporary_path.replace(checkpoint_path)


def _fit(model, optimizer, loaders, device, normalizer, config, args, report):
    """Run epochs, save improvements and retain incomplete evidence on interruption."""
    train_loader, validation_loader = loaders
    best = float("inf")
    report_path = args.run / "run.json"
    started = perf_counter()

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_started = perf_counter()
            train_loss = run_epoch(model, train_loader, device, optimizer)
            validation_loss = run_epoch(model, validation_loader, device)
            validation_metrics = evaluate_model(
                model, normalizer, validation_loader.dataset, device, args.batch_size)
            relative_l2 = validation_metrics["overall"]["mean"]
            if relative_l2 is None:
                raise RuntimeError("Nonfinite validation predictions")
            report["epochs"].append(
                {
                    "epoch": epoch,
                    "train_mse": train_loss,
                    "validation_mse": validation_loss,
                    "validation_relative_l2": relative_l2,
                    "validation_by_family": {name: value["mean"] for name, value
                                             in validation_metrics["by_family"].items()},
                    "seconds": perf_counter() - epoch_started,
                }
            )

            if relative_l2 < best:
                best = relative_l2
                save_checkpoint(
                    args.run,
                    model,
                    normalizer,
                    config,
                    epoch,
                    validation_loss,
                    report["dataset_manifest_sha256"],
                    relative_l2,
                )
                report["best_epoch"] = epoch
                report["best_validation_relative_l2"] = best

            print(f"epoch {epoch}: validation relative L2={relative_l2:.6f}", flush=True)

            report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

        report["status"] = "complete"
        report["checkpoint_sha256"] = sha256((args.run / "best.pt").read_bytes()).hexdigest()
    finally:
        report["training_seconds"] = perf_counter() - started
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main(argv=None) -> int:
    """Print a training plan or execute the requested training run.

    Args:
        argv: CLI argument strings; None reads the process arguments.
    Returns:
        Exit code 0 after a successful dry run or completed training.
    Raises:
        ValueError: Configuration, dataset or normalization checks fail.
        FileExistsError: The requested run directory already exists.
        RuntimeError: The requested device or optimization fails.
    """
    args = parse_args(argv)
    config = GeneratorConfig.load(args.config)

    if not args.train:
        print(json.dumps(_training_plan(args, config), indent=2))
        return 0

    # cuBLAS reads this before CUDA initialization; respect a site-provided value.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = resolve_device(args.device)
    seed_everything(config.seed)
    torch.set_num_threads(args.threads)

    training, validation, normalizer = _load_datasets(args, config)

    model = OrderConditionedMLP(config.max_order, args.width, args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loaders = _build_loaders(training, validation, args, device, config.seed)

    args.run.mkdir(parents=True, exist_ok=False)
    report = build_report(args, config, model, normalizer, training, device)

    _fit(model, optimizer, loaders, device, normalizer, config, args, report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
