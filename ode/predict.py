"""Standalone checkpoint inference: no dataset or trainer is needed.

The JSON CLI takes coefficients in ascending derivative order, initial state,
forcing, and elapsed times; it writes nondimensional states in the same order.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import Normalizer, encode
from .equations import LinearODE
from .model import OrderConditionedMLP


class Predictor:
    """Load weights/preprocessing once for repeated CPU or CUDA inference.

    checkpoint_path is a linear-padded-v1 checkpoint. Device is cpu or cuda;
    unavailable devices and invalid artifacts raise. Saved training settings
    are exposed as checkpoint['config']; architecture capacity is not coverage.
    """

    def __init__(self, checkpoint_path, device="cpu"):
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Use cpu or cuda")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device unavailable")
        self.checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (self.checkpoint.get("schema_version") != 1
                or self.checkpoint.get("encoding") != "linear-padded-v1"):
            raise ValueError("Unsupported checkpoint/feature encoding")
        self.model = OrderConditionedMLP(**self.checkpoint["architecture"]).to(self.device)
        self.model.load_state_dict(self.checkpoint["state_dict"])
        self.model.eval()
        self.normalizer = Normalizer(**self.checkpoint["normalizer"])
        if len(self.normalizer.target_mean) != self.model.max_order:
            raise ValueError("Checkpoint architecture and normalization disagree")

    def predict(self, equation: LinearODE, elapsed_times, batch_size=4096):
        """Return float64 states (T, n) for finite nonnegative times (T,).

        Inference uses float32 weights, then restores nondimensional scale.
        Empty times, invalid shapes/orders and nonfinite predictions raise.
        Requests beyond the saved scientific domain are unsupported; the CLI
        rejects them unless --allow-extrapolation explicitly opts in.
        """
        features, mask = encode(equation, elapsed_times, self.model.max_order)
        if not len(features) or batch_size < 1:
            raise ValueError("Need at least one time and positive batch_size")
        features = self.normalizer.transform_features(features, mask)
        with torch.inference_mode():
            outputs = [self.model(torch.as_tensor(features[start:start + batch_size],
                       dtype=torch.float32, device=self.device)).cpu().numpy()
                       for start in range(0, len(features), batch_size)]
        states = self.normalizer.inverse_target(np.concatenate(outputs))[:, :equation.order]
        if not np.isfinite(states).all():
            raise RuntimeError("Model produced a non-finite prediction")
        return states


def domain_violations(equation, times, config):
    """Describe requests outside the saved stable-mode distribution/horizon.

    Roots of [1, a_(n-1), ..., a_0] must match the sampled complex pair plus
    optional real root, with 1e-7 numerical slack. Coefficient box bounds alone
    would admit untrained/unstable equations, so check the mode structure too.
    """
    if not config:
        return ["checkpoint has no scientific domain metadata"]
    errors = []
    if equation.order not in config["orders"] or equation.family not in config["families"]:
        errors.append("unsupported order or forcing family")
    tolerance = 1e-7
    def within(values, bounds):
        return all(bounds[0] - tolerance <= value <= bounds[1] + tolerance for value in values)
    roots = np.roots([1, *reversed(equation.coefficients)])
    complex_roots = roots[np.abs(roots.imag) > tolerance]
    if (len(complex_roots) != 2 * (equation.order // 2)
            or not within(-roots.real, config["decay_range"])
            or not within(np.abs(complex_roots.imag), config["frequency_range"])):
        errors.append("coefficients outside sampled stable-mode family")
    if not within(equation.initial_state, config["initial_state_range"]):
        errors.append("initial state outside training bounds")
    if equation.amplitude and (not within([equation.amplitude], config["amplitude_range"])
                               or not within([equation.frequency], config["frequency_range"])):
        errors.append("forcing outside training bounds")
    if not within(times, [0, config["t_stop"] - config["t_start"]]):
        errors.append("elapsed time outside training horizon")
    return errors


def predict(checkpoint_path, equation: LinearODE, elapsed_times) -> np.ndarray:
    """Compatibility helper: load on CPU and return float64 states (T, n)."""
    return Predictor(checkpoint_path).predict(equation, elapsed_times)


def main(argv=None):
    """Read equation/times JSON and write new prediction JSON independently."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--allow-extrapolation", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    request = json.loads(args.input.read_text())
    equation = LinearODE(**request["equation"])
    times = np.asarray(request["elapsed_times"], dtype=float)
    predictor = Predictor(args.checkpoint, args.device)
    # Validate shape/finiteness before inspecting scientific coverage.
    encode(equation, times, predictor.model.max_order)
    violations = domain_violations(equation, times, predictor.checkpoint.get("config"))
    if violations and not args.allow_extrapolation:
        parser.error("; ".join(violations) + "; use --allow-extrapolation for an unsupported probe")
    states = predictor.predict(equation, times)
    result = {**request, "states": states.tolist(), "domain_violations": violations,
              "units": "nondimensional; components y, y', ..."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
