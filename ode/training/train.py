"""Train the mixed-order trajectory model and select weights on validation data.

The trainer loads generated splits, fits training-only normalization, batches
examples into the model, and saves weights plus preprocessing for prediction.
The CLI prints a plan unless --train requests execution with data/run paths.
"""

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
from importlib.util import find_spec
import json
import math
import os
from pathlib import Path
import platform
import random
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import CONFIG_DIR, GeneratorConfig, provenance
from .dataset import Normalizer, TrajectoryDataset, feature_width
from .model import OrderConditionedMLP, masked_mse
from ..evaluation.evaluate import METRIC, evaluate_model
from ..resources import TRAINING_KEYS, available_cpus, load_resources


# Bound the combined gradient norm to limit unusually large optimizer updates.
MAX_GRADIENT_NORM = 1.0
# --loss relative divides by each trajectory's size, but never by less than this
# (nondimensional state units), so near-zero solutions cannot dominate the loss.
RELATIVE_LOSS_FLOOR = 0.05


def relative_loss_weights(dataset):
    """Per-element weights (N*T, M) turning masked MSE into the evaluation score.

    Evaluation divides each trajectory's RMS error, in original units, by the
    RMS of its exact answer. A standardized error times target_scale is the
    original-unit error, so weighting component k by target_scale[k]**2 /
    max(RMS, RELATIVE_LOSS_FLOOR)**2 makes the loss each row's squared
    relative error. Weights are rescaled to average 1, so the gradient-clipping
    threshold and learning rates keep their usual meaning.
    """
    raw = dataset.raw_targets
    mask = dataset._preloaded["mask"].reshape(raw.shape)
    size = np.sqrt((raw ** 2 * mask).sum((1, 2)) / mask.sum((1, 2)))
    weight = mask * dataset.normalizer.target_scale ** 2 / np.maximum(size, RELATIVE_LOSS_FLOOR)[:, None, None] ** 2
    return (weight / weight[mask > 0].mean()).reshape(-1, raw.shape[-1]).astype(np.float32)


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


class TensorLoader:
    """Serve a preloaded split as device-resident batches, replacing DataLoader.

    DataLoader builds every batch by calling __getitem__ once per example and
    stacking the results in Python; for this small MLP that kept the GPU idle
    most of the time. Here the float32 arrays (features (N, F), target/mask
    (N, M)) are copied to the device once and batches are tensor slices.

    Args:
        dataset: TrajectoryDataset after preload().
        batch_size: Examples per batch; the last batch may be smaller.
        device: Device that holds the tensors.
        generator: CPU torch.Generator for shuffling; None keeps stored order.
        weight: Optional (N, M) loss weights served as batch["weight"].
        focus_hard: Draw rows from hard trajectories more often (see below).
    Note: Each epoch consumes the generator exactly as DataLoader does (one
    int64 worker-seed draw, the epoch's torch.randperm(N), one trailing
    randperm), so a training seed gives the same batch order as runs made
    before this loader existed.

    With focus_hard, each epoch draws N rows with replacement instead: a
    trajectory is picked with probability half uniform, half proportional to
    its RMS training loss in the previous epoch (record() collects it), then
    one of its time points uniformly. Every trajectory keeps at least half its
    usual share; the ones the model gets most wrong get up to many times more.
    trajectory_weights holds those probabilities (saved in last.pt).
    """

    def __init__(self, dataset, batch_size, device, generator=None, weight=None, focus_hard=False):
        self.dataset, self.batch_size, self.generator = dataset, batch_size, generator
        self.tensors = {name: torch.as_tensor(array, device=device)
                        for name, array in dataset._preloaded.items()}
        if weight is not None:
            self.tensors["weight"] = torch.as_tensor(weight, device=device)
        mask = self.tensors["mask"]
        # Checked once here so masked_mse needs no per-batch GPU->CPU sync.
        if bool(((mask != 0) & (mask != 1)).any()) or bool((mask.sum(-1) == 0).any()):
            raise ValueError("Masks must be binary with at least one active derivative")
        trajectories = len(dataset.records)
        self.time_points = len(dataset) // trajectories
        self.trajectory_weights = torch.full((trajectories,), 1 / trajectories, dtype=torch.float64) if focus_hard else None

    def __len__(self):
        return -(-len(self.dataset) // self.batch_size)

    def __iter__(self):
        if self.trajectory_weights is not None:
            yield from self._focused()
            return
        count = len(self.dataset)
        order = None
        if self.generator is not None:
            torch.empty((), dtype=torch.int64).random_(generator=self.generator)  # DataLoader's base seed
            order = torch.randperm(count, generator=self.generator).to(self.tensors["mask"].device)
        for start in range(0, count, self.batch_size):
            rows = (slice(start, start + self.batch_size) if order is None
                    else order[start:start + self.batch_size])
            yield {name: tensor[rows] for name, tensor in self.tensors.items()}
        if self.generator is not None:
            # RandomSampler draws one more (unused) permutation when an epoch ends.
            torch.randperm(count, generator=self.generator)

    def _focused(self):
        """One focus_hard epoch; afterwards trajectory_weights follows record()ed losses."""
        count, device = len(self.dataset), self.tensors["mask"].device
        # Inverse-CDF draw on the device: torch.multinomial took ~5 s per 40M rows on the CPU.
        cdf = torch.cumsum(self.trajectory_weights, 0).to(device)
        uniform = torch.rand(count, dtype=torch.float64, generator=self.generator).to(device) * cdf[-1]
        picked = torch.searchsorted(cdf, uniform, right=True).clamp_(max=len(cdf) - 1)
        times = torch.randint(self.time_points, (count,), generator=self.generator).to(device)
        order = picked * self.time_points + times
        self._loss_sum = torch.zeros(len(self.trajectory_weights), dtype=torch.float64, device=device)
        self._loss_count = torch.zeros_like(self._loss_sum)
        for start in range(0, count, self.batch_size):
            rows = order[start:start + self.batch_size]
            yield {**{name: tensor[rows] for name, tensor in self.tensors.items()},
                   "trajectory": rows // self.time_points}
        error = (self._loss_sum / self._loss_count.clamp(min=1)).sqrt()
        visited = self._loss_count > 0
        error = torch.where(visited, error, error[visited].mean())  # unvisited: assume average
        self.trajectory_weights = (0.5 / len(error) + 0.5 * error / error.sum()).cpu()

    def record(self, trajectory, losses):
        """Add one batch's per-row training losses to their trajectories' totals."""
        self._loss_sum.index_add_(0, trajectory, losses.double())
        self._loss_count.index_add_(0, trajectory, torch.ones_like(self._loss_sum[trajectory]))


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
    Note: The loss total stays on the device and is read once per epoch.
    Reading it every batch forced the CPU to wait for the GPU each step. A
    non-finite gradient therefore surfaces at the end of its epoch (the loss
    turns NaN) rather than at the exact step; best.pt is never overwritten
    by a NaN epoch because selection compares finite validation scores.
    """
    model.train(optimizer is not None)
    total, count = torch.zeros((), dtype=torch.float64, device=device), 0

    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            features, target, mask = (
                batch[name].to(device, non_blocking=device.type == "cuda")
                for name in ("features", "target", "mask")
            )

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            rows = masked_mse(model(features), target, mask, batch.get("weight"), per_row=True)
            loss = rows.mean()
            if "trajectory" in batch:
                loader.record(batch["trajectory"], rows.detach())

            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRADIENT_NORM)
                optimizer.step()

            # Weight by actual batch size: batches of 4 and 1 count as 4:1.
            total += loss.detach().double() * len(features)
            count += len(features)

    if not count:
        raise ValueError("Cannot run an epoch with an empty dataset")

    total = float(total)

    if not math.isfinite(total):
        raise RuntimeError("Non-finite loss; inspect data/scaling and optimization")

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
        help="Scientific recipe; defaults to the one recorded in --data's manifest",
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--learning-rate-schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--training-seed", type=int, help="Model/shuffle seed; defaults to the unchanged dataset seed")
    parser.add_argument("--save-training-state", action=argparse.BooleanOptionalAction, default=True,
                        help="Save last.pt after every epoch so --resume can continue the run")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an interrupted --run directory from its last.pt")
    parser.add_argument("--loss", choices=("mse", "relative"), default="mse",
                        help="relative: weight each trajectory's error by 1/its size, matching how runs are scored")
    parser.add_argument("--focus-hard", action="store_true",
                        help="Train more often on the trajectories with the largest current error")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the model with CUDA graphs on a GPU: about 1.6x faster steps")
    parser.add_argument("--snapshot-every", type=int, default=0, metavar="N",
                        help="Every N epochs (and at epoch 0 of a fresh run) save the weights to "
                             "<run>/snapshots/ so `python -m ode neurons` can animate the learning; 0 = off")
    parser.add_argument("--initialize-from", type=Path,
                        help="Start from another run's best.pt (same width and depth) instead of random weights, "
                             "e.g. to train an under-trained model further. A checkpoint from a different "
                             "dataset also lends its input/target scaling")
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=available_cpus(),
                        help="CPU threads for PyTorch; defaults to the CPUs this job may use")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto",
                        help="auto uses the GPU when PyTorch can see one (ROCm GPUs count as cuda)")
    parser.add_argument(
        "--preload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hold standardized examples in memory (on the GPU when training there); "
             "--no-preload streams archives through DataLoader for splits too large to hold",
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

    if not np.isfinite(args.minimum_learning_rate) or args.minimum_learning_rate < 0:
        parser.error("--minimum-learning-rate must be finite and nonnegative")
    if args.learning_rate_schedule == "cosine" and (
            args.epochs < 2 or args.minimum_learning_rate > args.learning_rate):
        parser.error("Cosine scheduling needs >=2 epochs and minimum rate <= initial rate")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay must be finite and nonnegative")
    if args.training_seed is not None and not 0 <= args.training_seed < 2**64:
        parser.error("--training-seed must fit an unsigned 64-bit integer")

    if args.train and (args.data is None or args.run is None):
        parser.error("--train requires --data and a new --run directory")

    if (args.loss == "relative" or args.focus_hard) and not args.preload:
        parser.error("--loss relative and --focus-hard need --preload")

    if args.snapshot_every < 0:
        parser.error("--snapshot-every must be nonnegative")

    if args.resume and not args.save_training_state:
        parser.error("--resume needs --save-training-state to keep last.pt current")

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
        "training_seed": config.seed if args.training_seed is None else args.training_seed,
        "optimizer": {"name": "AdamW", "learning_rate": args.learning_rate,
                      "weight_decay": args.weight_decay,
                      "schedule": args.learning_rate_schedule,
                      "minimum_learning_rate": args.minimum_learning_rate},
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
    if args.initialize_from:
        # Copied weights only make sense on the input scale they learned, so a different dataset
        # keeps the source's preprocessing instead of refitting it.
        source = torch.load(args.initialize_from, map_location="cpu", weights_only=True)
        if source["dataset_manifest_sha256"] != sha256((args.data / "manifest.json").read_bytes()).hexdigest():
            normalizer = Normalizer(**source["normalizer"])
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

    if args.preload:
        weight = relative_loss_weights(training) if args.loss == "relative" else None
        return (TensorLoader(training, args.batch_size, device, generator, weight, args.focus_hard),
                TensorLoader(validation, args.batch_size, device))

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
        "seed": args.training_seed,
        "data_seed": config.seed,
        "device": str(device),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
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


def build_scheduler(optimizer, args):
    """Return a constant policy (None) or epoch-wise cosine learning-rate decay.

    With E>=2 epochs, epoch 1 uses the initial rate and epoch E uses the minimum.
    Step only between epochs, after optimizer updates. No warm restarts occur.
    """
    if args.learning_rate_schedule == "constant":
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - 1, eta_min=args.minimum_learning_rate)


def save_training_state(run, model, optimizer, scheduler, loader, normalizer, config, report, epoch):
    """Atomically save the last epoch's state for inspection or a future continuation.

    Saves float32 weights, AdamW moments, scheduler, CPU/CUDA/Python/NumPy RNGs,
    training-loader shuffle state and dataset identity. NumPy arrays become lists
    so torch.load(weights_only=True) can read this artifact. This does not add a
    resume CLI; best.pt remains the independently loadable inference checkpoint.
    """
    numpy_state = np.random.get_state()
    payload = {
        "schema_version": 1, "kind": "training-state", "epoch": epoch,
        "architecture": model.architecture, "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "normalizer": normalizer.to_dict(), "config": asdict(config),
        "dataset_manifest_sha256": report["dataset_manifest_sha256"],
        "best_epoch": report["best_epoch"], "best_validation_relative_l2": report["best_validation_relative_l2"],
        "report": report,
        "trajectory_weights": getattr(loader, "trajectory_weights", None),
        "rng": {"torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
                "loader": loader.generator.get_state(), "python": random.getstate(),
                "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]]},
    }
    temporary = run / "last.pending.pt"
    torch.save(payload, temporary)
    temporary.replace(run / "last.pt")


def save_snapshot(run, model, epoch):
    """Write the current weights to run/snapshots/epoch-NNNN.pt (inference weights only)."""
    folder = run / "snapshots"
    folder.mkdir(exist_ok=True)
    torch.save({"epoch": epoch, "architecture": model.architecture,
                "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()}},
               folder / f"epoch-{epoch:04d}.pt")


def _duration(seconds):
    """Format seconds as a short human-readable span, e.g. 3725 -> '1h02m'."""
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{seconds:02d}s"


def _fit(model, optimizer, loaders, device, normalizer, config, args, report, scheduler=None,
         start_epoch=1, compiled=None):
    """Run epochs, save improvements and retain incomplete evidence on interruption.

    compiled: optional torch.compile wrapper of model used for the training and
    validation-loss passes; evaluation and checkpoints use model itself.
    """
    train_loader, validation_loader = loaders
    forward = model if compiled is None else compiled  # `compiled or model` would call len() on it
    best = report.get("best_validation_relative_l2", float("inf"))
    report_path = args.run / "run.json"
    started = perf_counter()
    previous_seconds = report.get("training_seconds", 0.0)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_started = perf_counter()
            learning_rate = optimizer.param_groups[0]["lr"] if optimizer is not None else None
            train_loss = run_epoch(forward, train_loader, device, optimizer)
            validation_loss = run_epoch(forward, validation_loader, device)
            validation_metrics = evaluate_model(model, normalizer, validation_loader.dataset, device)
            relative_l2 = validation_metrics["overall"]["mean"]
            if relative_l2 is None:
                raise RuntimeError("Nonfinite validation predictions")
            report["epochs"].append(
                {
                    "epoch": epoch,
                    "learning_rate": learning_rate,
                    "train_mse" if getattr(args, "loss", "mse") == "mse" else "train_relative_mse": train_loss,
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

            elapsed = perf_counter() - started
            remaining = elapsed / (epoch - start_epoch + 1) * (args.epochs - epoch)
            print(f"epoch {epoch}/{args.epochs}  validation error {relative_l2:.3%}  "
                  f"(best {report['best_validation_relative_l2']:.3%} at epoch {report['best_epoch']})  "
                  f"{report['epochs'][-1]['seconds']:.1f}s/epoch  ETA {_duration(remaining)}", flush=True)

            if getattr(args, "snapshot_every", 0) and epoch % args.snapshot_every == 0:
                save_snapshot(args.run, model, epoch)
            if scheduler is not None and epoch < args.epochs:
                scheduler.step()
            if getattr(args, "save_training_state", False):
                save_training_state(args.run, model, optimizer, scheduler, train_loader,
                                    normalizer, config, report, epoch)

            report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

        report["status"] = "complete"
        report["checkpoint_sha256"] = sha256((args.run / "best.pt").read_bytes()).hexdigest()
    finally:
        report["training_seconds"] = previous_seconds + perf_counter() - started
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def _initialize(args, model):
    """Copy weights from args.initialize_from and return its SHA-256.

    A checkpoint from another dataset is fine: _load_datasets then keeps its
    normalizer. The architecture must match; a mismatch raises ValueError.
    """
    source = torch.load(args.initialize_from, map_location="cpu", weights_only=True)
    if source["architecture"] != model.architecture:
        raise ValueError(f"{args.initialize_from} holds a {source['architecture']} model; "
                         "pass its --width/--depth")
    model.load_state_dict(source["state_dict"])
    return sha256(args.initialize_from.read_bytes()).hexdigest()


def _resume(args, config, model, optimizer, scheduler, train_loader):
    """Restore run.json state, weights, optimizer and RNGs from last.pt.

    Returns the report to continue and the first epoch still to run. The run
    must use the same dataset and scientific config; a mismatch raises
    ValueError instead of silently mixing two experiments. Use the original
    command plus --resume; --epochs may be raised to extend a finished run.
    """
    state = torch.load(args.run / "last.pt", weights_only=True)
    manifest_sha256 = sha256((args.data / "manifest.json").read_bytes()).hexdigest()
    saved = replace(GeneratorConfig.from_dict(state["config"]), workers=config.workers, backend=config.backend)
    if state["dataset_manifest_sha256"] != manifest_sha256 or saved != config:
        raise ValueError("last.pt was trained on a different dataset or config")
    if state["architecture"] != model.architecture:
        raise ValueError(f"{args.run} holds a {state['architecture']} model but this command builds "
                         f"{model.architecture}; resume with the original --width/--depth, "
                         "or start a different model under a new --name")
    model.load_state_dict(state["state_dict"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    rng = state["rng"]
    torch.set_rng_state(rng["torch_cpu"])
    if rng["torch_cuda"]:
        torch.cuda.set_rng_state_all(rng["torch_cuda"])
    train_loader.generator.set_state(rng["loader"])
    if hasattr(train_loader, "trajectory_weights") and state.get("trajectory_weights") is not None:
        train_loader.trajectory_weights = state["trajectory_weights"]
    random.setstate(rng["python"])
    name, keys, *rest = rng["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), *rest))
    report = state["report"]
    report["status"] = "incomplete"
    report["training_options"] = {name: str(value) if isinstance(value, Path) else value
                                  for name, value in vars(args).items()}
    print(f"resuming {args.run} after epoch {state['epoch']}", flush=True)
    return report, state["epoch"] + 1


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
    # Without --config, train on whatever recipe generated --data.
    if args.config is None and args.data is not None:
        config = GeneratorConfig.from_dict(json.loads((args.data / "manifest.json").read_text())["config"])
    else:
        config = GeneratorConfig.load(args.config or CONFIG_DIR / "generator.json")

    if not args.train:
        print(json.dumps(_training_plan(args, config), indent=2))
        return 0

    if args.training_seed is None:
        args.training_seed = config.seed

    # cuBLAS reads this before CUDA initialization; respect a site-provided value.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = resolve_device(args.device)
    seed_everything(args.training_seed)
    torch.set_num_threads(args.threads)

    training, validation, normalizer = _load_datasets(args, config)

    model = OrderConditionedMLP(config.max_order, args.width, args.depth).to(device)
    # fused=True does the whole AdamW update in one GPU kernel: ~1.4x faster steps.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                                  fused=device.type == "cuda")
    # reduce-overhead replays recorded CUDA graphs, removing most per-step launch cost.
    if args.compile and device.type == "cuda" and find_spec("triton") is None:
        print("--compile needs the triton package (not available on Windows); training without it")
        args.compile = False
    compiled = torch.compile(model, mode="reduce-overhead") if args.compile and device.type == "cuda" else None
    scheduler = build_scheduler(optimizer, args)
    loaders = _build_loaders(training, validation, args, device, args.training_seed)

    if args.resume:
        report, start_epoch = _resume(args, config, model, optimizer, scheduler, loaders[0])
    else:
        initialized_from = _initialize(args, model) if args.initialize_from else None
        args.run.mkdir(parents=True, exist_ok=False)
        report, start_epoch = build_report(args, config, model, normalizer, training, device), 1
        report["initialized_from_sha256"] = initialized_from
        if args.snapshot_every:
            save_snapshot(args.run, model, 0)

    _fit(model, optimizer, loaders, device, normalizer, config, args, report, scheduler, start_epoch, compiled)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
