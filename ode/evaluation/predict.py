"""Standalone checkpoint inference: no dataset or trainer is needed.

The JSON CLI takes coefficients in ascending derivative order, initial state,
forcing, and elapsed times; it writes nondimensional states in the same order.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..training.dataset import Normalizer, encode
from ..generation.equations import LinearODE
from ..training.model import OrderConditionedMLP


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

    def predict_rescaled(self, equation: LinearODE, elapsed_times, state_scale=0.5, handoff=0.5):
        """Predict equations outside the trained box using the ODE's own symmetries.

        Three exact properties of linear constant-coefficient ODEs move any
        request into the box the model saw: time scaling (s = c t turns every
        root λ into λ/c), superposition (the free response from the initial
        state plus the forced response from rest, each scaled to size
        state_scale and back) and time shift (the predicted state `handoff` of
        the way through the model's time window starts the next piece; the
        model is least accurate near the end of its window, and halfway
        measured best). Returns float64 states (T, n) and the number of pieces
        stitched; error grows slowly with that count.
        Raises ValueError when no time scale fits (see time_scale).
        """
        config = self.checkpoint.get("config")
        if not config:
            raise ValueError("checkpoint has no scientific domain metadata")
        times = np.asarray(elapsed_times, dtype=float)
        if not len(times) or not np.isfinite(times).all() or times.min() < 0:
            raise ValueError("Need at least one finite nonnegative time")
        c, n = time_scale(equation, config), equation.order
        step = handoff * (config["t_stop"] - config["t_start"]) / c
        down = c ** -np.arange(n, dtype=float)  # y^(k) in original time -> z^(k) in model time
        coefficients = tuple(equation.coefficients[k] * c ** (k - n) for k in range(n))

        def piece(state, phase, offsets):
            """Original-unit states at offsets (each <= step) after `state`."""
            model_times = np.asarray(offsets) * c
            result = np.zeros((len(model_times), n))
            start = state * down
            size = np.abs(start).max()
            if size > 0:
                free = LinearODE(coefficients, tuple(start * state_scale / size))
                result += self.predict(free, model_times) * size / state_scale
            if equation.amplitude:
                forced = LinearODE(coefficients, (0.0,) * n, state_scale, equation.frequency / c,
                                   float(np.angle(np.exp(1j * phase))))
                result += self.predict(forced, model_times) * equation.amplitude / c ** n / state_scale
            return result / down

        states, state, start, pieces = np.empty((len(times), n)), np.array(equation.initial_state), 0.0, 0
        while True:
            phase = equation.phase + equation.frequency * start
            inside = (times >= start) & (times <= start + step)
            if inside.any():
                states[inside] = piece(state, phase, times[inside] - start)
            pieces += 1
            if start + step >= times.max():
                return states, pieces
            state = piece(state, phase, [step])[0]
            start += step


def time_scale(equation, config):
    """Return c so the equation with time sped up c-fold fits the trained box.

    Every root λ becomes λ/c and the forcing frequency ω becomes ω/c, so what
    must have been trained is only their ratios (e.g. decay/frequency). Picks
    the geometric middle of the allowed c values. Raises ValueError for an
    untrained order, family or mode structure, unstable or undamped modes, or
    ratios no single c can fit.
    """
    if equation.order not in config["orders"] or equation.family not in config["families"]:
        raise ValueError("unsupported order or forcing family")
    roots = np.roots([1, *reversed(equation.coefficients)])
    complex_roots = roots[np.abs(roots.imag) > 1e-7]
    if len(complex_roots) != 2 * (equation.order // 2):
        raise ValueError("needs exactly one oscillating pair of modes, like the training data")
    ranges = [(-roots.real, config["decay_range"])]
    if len(complex_roots):  # a first-order equation has no oscillating pair
        ranges.append((np.abs(complex_roots.imag), config["frequency_range"]))
    if equation.amplitude:
        ranges.append((np.array([equation.frequency]), config["frequency_range"]))
    low, high = 0.0, np.inf
    for values, (lower, upper) in ranges:
        low, high = max(low, values.max() / upper), min(high, values.min() / lower)
    if not 0 < low <= high * (1 + 1e-9):
        raise ValueError("decay/frequency ratios are outside anything the model was trained on")
    return float(np.sqrt(low * high))


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
        errors.append(f"coefficients outside sampled stable-mode family (decay {np.round(-roots.real, 3).tolist()}, "
                      f"frequency {np.round(np.abs(complex_roots.imag), 3).tolist()}; trained on decay "
                      f"{config['decay_range']}, frequency {config['frequency_range']})")
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
    """Read equation/times JSON; write prediction JSON and/or an exact-vs-model chart."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New JSON file with predicted and exact states")
    parser.add_argument("--csv", type=Path, help="CSV table: time, exact y, y', ..., predicted y, y', ...; "
                        "default: next to --input when no output option is given")
    parser.add_argument("--plot", type=Path, help="SVG chart comparing the prediction with the exact answer; "
                        "default: next to --input (my_equation.json -> my_equation.svg) when --output is not given")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--allow-extrapolation", action="store_true")
    args = parser.parse_args(argv)
    if not (args.output or args.plot or args.csv):
        args.plot, args.csv = args.input.with_suffix(".svg"), args.input.with_suffix(".csv")
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    request = json.loads(args.input.read_text())
    equation = LinearODE(**request["equation"])
    predictor = Predictor(args.checkpoint, args.device)
    # Without elapsed_times, use the trained grid: 101 points over the horizon.
    config = predictor.checkpoint.get("config") or {}
    times = np.asarray(request.get("elapsed_times") or np.linspace(
        0, config.get("t_stop", 5.0) - config.get("t_start", 0.0), config.get("time_points", 101)), dtype=float)
    # Validate shape/finiteness before inspecting scientific coverage.
    encode(equation, times, predictor.model.max_order)
    violations = domain_violations(equation, times, predictor.checkpoint.get("config"))
    pieces = 0
    if violations and not args.allow_extrapolation:
        try:
            states, pieces = predictor.predict_rescaled(equation, times)
        except ValueError as error:
            parser.error("; ".join(violations) + f"; rescaling cannot help: {error}; "
                         "use --allow-extrapolation for an unsupported probe")
        print(f"outside the trained range: rescaled time and stitched {pieces} model run(s)")
    else:
        states = predictor.predict(equation, times)
    # The exact answer is a matrix exponential: cheap, and what the model is judged against.
    from ..generation.analytic import linear_reference
    from .evaluate import trajectory_metrics
    exact = linear_reference(equation, times)
    error = trajectory_metrics(exact, states)["relative_l2"]
    if args.csv:
        names = [f"y{chr(39) * k}" for k in range(equation.order)]  # ASCII y' so Excel reads the header
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(args.csv, np.column_stack([times, exact, states]), delimiter=",", comments="", encoding="utf-8",
                   header=",".join(["time", *(f"exact {n}" for n in names), *(f"predicted {n}" for n in names)]))
        print(f"wrote {args.csv}")
    if args.output:
        result = {**request, "elapsed_times": times.tolist(), "states": states.tolist(),
                  "exact_states": exact.tolist(), "relative_l2": error,
                  "domain_violations": violations, "rescaled_pieces": pieces,
                  "units": "nondimensional; components y, y', ..."}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"wrote {args.output}")
    if args.plot:
        # Imported here: plot_results imports this module.
        from .plot_results import _equation_text, _family_name, trajectory_svg
        subtitle = [*_equation_text(equation)]
        if pieces:
            subtitle.append(f"Outside the trained range, so time was rescaled and the answer stitched "
                            f"from {pieces} model run(s).")
        elif violations:
            subtitle.append("Outside what the model was trained on: " + "; ".join(violations))
        title = f"Your equation: {_family_name(equation.order, equation.family)} — off by {error:.1%}"
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        args.plot.write_text(trajectory_svg(times, exact, states, title, subtitle), encoding="utf-8")
        print(f"wrote {args.plot}: model is off by {error:.2%} from the exact answer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
