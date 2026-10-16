"""Evaluate complete trajectories using the frozen nondimensional v1 metric.

Training uses the same metric for checkpoint selection. CLI reports preserve
nonfinite predictions as failures, and never pool time points across cases.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import TrajectoryDataset

METRIC = {
    "version": "trajectory-rms-v1",
    "primary": "full_state_relative_l2",
    "component_scales": "one for each nondimensional derivative; no centering",
    "time_weights": "uniform over all stored points, including endpoints",
    "denominator_rms_floor": 1e-6,
    "failure_threshold": 0.05,
    "aggregation": "equal weight per trajectory; nonfinite results prevent passing",
}


def trajectory_metrics(reference, prediction):
    """Return relative L2 and absolute errors for matching arrays (T, n).

    For K=T*n active values, E=RMS(prediction-reference), R=RMS(reference),
    relative error is E/max(R, 1e-6). Away from zero this is the L2 norm ratio.
    Thus the norm denominator floor is 1e-6*sqrt(K). Components use K=T.
    Near-zero means R<=1e-6. Invalid shapes/references raise ValueError;
    nonfinite predictions yield explicit failures and JSON-compatible nulls.
    """
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if (reference.ndim != 2 or 0 in reference.shape or prediction.shape != reference.shape
            or not np.isfinite(reference).all()):
        raise ValueError("Expected matching nonempty (T, n) arrays and finite reference")

    def measure(ref, pred):
        rms = float(np.sqrt(np.mean(ref**2)))
        with np.errstate(over="ignore", invalid="ignore"):
            absolute_rms = float(np.sqrt(np.mean((pred - ref)**2)))
            absolute_max = float(np.max(np.abs(pred - ref)))
            relative = absolute_rms / max(rms, METRIC["denominator_rms_floor"])
        finite = bool(np.isfinite([absolute_rms, absolute_max, relative]).all())
        return {
            "relative_l2": relative if finite else None,
            "absolute_rmse": absolute_rms if finite else None,
            "absolute_max": absolute_max if finite else None,
            "reference_rms": rms,
            "near_zero": rms <= METRIC["denominator_rms_floor"],
            "nonfinite": not finite,
            "failed": not finite or relative >= METRIC["failure_threshold"],
        }

    result = measure(reference, prediction)
    result["components"] = {
        f"d{index}": measure(reference[:, index], prediction[:, index])
        for index in range(reference.shape[1])
    }
    return result


def summarize(rows):
    """Aggregate metric dictionaries, including every threshold/numerical failure.

    Primary mean/worst are null if any result is nonfinite. Finite-only tail
    statistics are labeled explicitly. Empty groups have null statistics.
    """
    values = [row["relative_l2"] for row in rows if row["relative_l2"] is not None]
    nonfinite = sum(row["nonfinite"] for row in rows)
    near_zero = [row for row in rows if row["near_zero"]]

    def stats(items):
        return {"mean": float(np.mean(items)), "median": float(np.median(items)),
                "p95": float(np.percentile(items, 95)), "worst": float(np.max(items))} if items else {
                    name: None for name in ("mean", "median", "p95", "worst")}

    finite = stats(values)
    return {
        "count": len(rows), "finite_count": len(values), "nonfinite_count": nonfinite,
        "failure_count": sum(row["failed"] for row in rows),
        "threshold_failure_count": sum(row["failed"] and not row["nonfinite"] for row in rows),
        "mean": finite["mean"] if not nonfinite else None,
        "median": finite["median"], "p95": finite["p95"],
        "worst": finite["worst"] if not nonfinite else None,
        "finite_only": finite,
        "absolute_rmse": stats([r["absolute_rmse"] for r in rows if r["absolute_rmse"] is not None]),
        "absolute_max": max((r["absolute_max"] for r in rows if r["absolute_max"] is not None), default=None),
        "near_zero_count": len(near_zero),
        "near_zero_absolute_rmse": stats([r["absolute_rmse"] for r in near_zero if r["absolute_rmse"] is not None]),
        "target_passed": bool(values) and not nonfinite and finite["mean"] < METRIC["failure_threshold"],
    }


def evaluate_model(model, normalizer, dataset, device, batch_size=4096):
    """Evaluate a split, returning aggregates and all trajectory metric records.

    Raw targets are (T, M); only the first n components contribute. Each
    inference batch contains at most batch_size points. Statistics are never
    refitted. Archive corruption raises; nonfinite outputs remain failures.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for index, record in enumerate(dataset.records):
                features, target, mask = dataset.raw_trajectory(index)
                features = normalizer.transform_features(features, mask)
                outputs = []
                for start in range(0, len(features), batch_size):
                    tensor = torch.as_tensor(features[start:start + batch_size], dtype=torch.float32, device=device)
                    outputs.append(model(tensor).cpu().numpy())
                prediction = normalizer.inverse_target(np.concatenate(outputs))
                order = record["order"]
                rows.append({"trajectory_id": record["trajectory_id"], "order": order,
                             "family": record["family"], "split": dataset.split,
                             **trajectory_metrics(target[:, :order], prediction[:, :order])})
    finally:
        model.train(was_training)

    def breakdown(group):
        result = summarize(group)
        result["components"] = {
            component: summarize([row["components"][component] for row in group if component in row["components"]])
            for component in sorted({key for row in group for key in row["components"]})
        }
        return result

    return {
        "overall": breakdown(rows),
        "by_family": {f"order{order}/{family}": breakdown([
            row for row in rows if row["order"] == order and row["family"] == family])
            for order, family in sorted({(row["order"], row["family"]) for row in rows})},
        "trajectories": rows,
    }


def main(argv=None):
    """Write new evaluation JSON; test evaluation must be explicitly requested."""
    from .predict import Predictor
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["validation"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    predictor = Predictor(args.checkpoint, args.device)
    manifest_hash = sha256((args.data / "manifest.json").read_bytes()).hexdigest()
    if manifest_hash != predictor.checkpoint.get("dataset_manifest_sha256"):
        raise ValueError("Evaluation dataset must match the checkpoint manifest")
    report = {"metric": METRIC, "checkpoint_sha256": sha256(args.checkpoint.read_bytes()).hexdigest(),
              "dataset_manifest_sha256": manifest_hash, "device": args.device,
              "test_evaluated": "test" in args.splits, "splits": {}}
    for split in args.splits:
        dataset = TrajectoryDataset(args.data, split, predictor.model.max_order)
        report["splits"][split] = evaluate_model(predictor.model, predictor.normalizer, dataset,
                                                 predictor.device, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: value["overall"]["mean"] for name, value in report["splits"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
