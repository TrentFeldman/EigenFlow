"""Build a portable, interactive HTML explorer of a trained model (the demo page).

    python -m ode viz                                   # ode/models/current -> its explorer.html
    python -m ode viz --model ode/models/current --compare ode/models/arc \\
        --input ode/equations/try_equation.json --output demo.html

The output is ONE self-contained .html file: no internet, no Python and no
server are needed to view it, so it can be emailed, put on a USB stick or
opened on Windows by double-clicking. The network's weights are embedded and
the page runs the network itself (in JavaScript), so moving a slider re-solves
the equation instantly. The page also solves the equation exactly
(matrix exponential, like generation/analytic.py) to show how far off the
network is.

Scenes (tabs in the page, or `#scene=<name>` in the address):
    map         every neuron and the strongest connections, lit up for one moment
    trace       the same map, following where one output's value came from
    lights      neuron grids lighting up as time runs through the solution
    sweep       slowly change one setting (e.g. damping) and watch where it breaks
    phase       the hidden layers squeezed to 2D next to the real y-vs-y' picture
    roles       neurons grouped by how they behave over time (rising, wiggling, ...)
    importance  which inputs drive the answer at each moment (gradient x input)
    compare     a second model (--compare) on the same equation
    reel        the worst validation equations from the model's evaluation file

Every scene draws on one 1920x1080 canvas; the page's Record button saves what
plays as a WebM video (`ffmpeg -i in.webm out.mp4` converts it), and Picture
saves a PNG. On load the page re-runs the starting equation and compares with
the PyTorch numbers stored here; a mismatch shows a red warning.
"""

import argparse
import base64
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from ..config import GeneratorConfig
from ..evaluation.evaluate import trajectory_metrics
from ..evaluation.plot_results import _equation_for
from ..evaluation.predict import Predictor
from ..generation.analytic import linear_reference
from ..generation.equations import LinearODE

PAGE = Path(__file__).with_name("viz.html")
DEFAULT_MODEL = Path("ode/models/current")
# Config keys the page needs: what was trained (for the "outside the trained range" warning) and the time grid.
CONFIG_KEYS = ("orders", "families", "decay_range", "frequency_range", "initial_state_range", "amplitude_range",
               "t_start", "t_stop", "time_points")


def _checkpoint(path):
    """Accept a model folder (holding best.pt) or the .pt file itself."""
    path = Path(path)
    return path / "best.pt" if path.is_dir() else path


def _floats(array):
    """Base64 of little-endian float32 values: 4 bytes per weight instead of ~10 as JSON text."""
    return base64.b64encode(np.ascontiguousarray(array, dtype="<f4").tobytes()).decode()


def export_model(predictor, name):
    """Return the JSON-ready weights, normalizer and trained range of one checkpoint.

    Each Linear layer becomes {"in", "out", "w", "b"} with w stored row-major as
    (out, in), the same layout as PyTorch, so out[i] = b[i] + sum_k w[i*in + k] x[k].
    Hidden layers are followed by SiLU; the last layer is not.
    """
    state = predictor.checkpoint["state_dict"]
    linear = [key[:-len(".weight")] for key in state if key.endswith(".weight")]
    layers = [{"in": state[f"{key}.weight"].shape[1], "out": state[f"{key}.weight"].shape[0],
               "w": _floats(state[f"{key}.weight"].numpy()), "b": _floats(state[f"{key}.bias"].numpy())}
              for key in linear]
    config = predictor.checkpoint["config"]
    return {"name": name, "architecture": predictor.checkpoint["architecture"],
            "parameters": sum(value.numel() for value in state.values()), "layers": layers,
            "normalizer": predictor.normalizer.to_dict(),
            "config": {key: config[key] for key in CONFIG_KEYS},
            "validation_error": predictor.checkpoint.get("validation_relative_l2")}


def modes(equation):
    """Describe an equation by its modes, the way the page's sliders do.

    The training data was built from roots (see equations.stable_coefficients):
    n // 2 oscillating pairs -d +/- iw plus one real root -r for odd n. Returns
    {"pairs": [[d, w], ...] (sorted by w), "real": [r] or []}, all decay rates
    d, r >= 0. Raises ValueError for any other structure (e.g. two real roots
    instead of a pair) or a growing mode, which the sliders cannot represent.
    """
    roots = np.roots([1, *reversed(equation.coefficients)])
    pairs = sorted(([-root.real, root.imag] for root in roots if root.imag > 1e-7), key=lambda pair: pair[1])
    real = [-root.real for root in roots if abs(root.imag) <= 1e-7]
    if len(pairs) != equation.order // 2 or len(real) != equation.order % 2:
        raise ValueError(f"needs {equation.order // 2} oscillating pair(s) of modes"
                         f"{' and one real mode' if equation.order % 2 else ''}, like the training data")
    if any(decay < -1e-9 for decay in [pair[0] for pair in pairs] + real):
        raise ValueError("the equation has a growing mode; the explorer only shows decaying ones")
    return {"pairs": [[max(d, 0.0), w] for d, w in pairs], "real": [max(r, 0.0) for r in real]}


def page_equation(equation):
    """The page's equation state: modes plus starting values and forcing."""
    return {"order": equation.order, **modes(equation), "init": list(equation.initial_state),
            "amplitude": equation.amplitude, "frequency": equation.frequency, "phase": equation.phase}


def worst_cases(predictor, checkpoint, summary_path, count):
    """Rebuild the `count` worst validation equations recorded in an evaluation file.

    Equations are regenerated from their trajectory ids and the checkpoint's
    generator config (the dataset itself is not needed), then re-scored; a
    rebuilt equation must reproduce its recorded error, or the reel would show
    the wrong equation. Returns the page's reel object.
    Raises ValueError if the evaluation belongs to a different checkpoint.
    """
    report = json.loads(Path(summary_path).read_text())
    if report["checkpoint_sha256"] != sha256(checkpoint.read_bytes()).hexdigest():
        raise ValueError(f"{summary_path} was made for a different checkpoint than {checkpoint}")
    rows = [row for row in report["splits"]["validation"]["trajectories"] if row["relative_l2"] is not None]
    rows.sort(key=lambda row: row["relative_l2"], reverse=True)
    config = GeneratorConfig.from_dict(predictor.checkpoint["config"])
    times = np.linspace(0, config.t_stop - config.t_start, config.time_points)
    cases = []
    for row in rows[:count]:
        equation = _equation_for(row["trajectory_id"], config)
        rebuilt = trajectory_metrics(linear_reference(equation, times), predictor.predict(equation, times))
        if not np.isclose(rebuilt["relative_l2"], row["relative_l2"], rtol=1e-3):
            raise ValueError(f"Rebuilt {row['trajectory_id']} does not reproduce its evaluated error")
        cases.append({"id": row["trajectory_id"], "error": row["relative_l2"],
                      "reference_rms": row["reference_rms"], "equation": page_equation(equation)})
    return {"total": len(rows), "typical_rms": float(np.median([row["reference_rms"] for row in rows])),
            "source": Path(summary_path).name, "cases": cases}


def build(model, input_path=None, compare=None, evaluation=None, worst=8):
    """Return the finished HTML text for one model (and an optional second one).

    Args:
        model: Model folder or checkpoint path; the page's main network.
        input_path: Optional equation JSON (as for `predict`) to start on;
            default is a gently damped spring.
        compare: Optional second model folder or checkpoint for the compare scene.
        evaluation: Evaluation JSON for the worst-case reel; default
            <model folder>/evaluation_summary.json or evaluation.json, skipped if absent.
        worst: Number of worst cases to put in the reel.
    Raises:
        ValueError: An unusable equation, or an evaluation of another checkpoint.
    """
    checkpoint = _checkpoint(model)
    predictors = [Predictor(checkpoint)]
    names = [checkpoint.parent.name]
    if compare:
        predictors.append(Predictor(_checkpoint(compare)))
        names.append(_checkpoint(compare).parent.name)
    if evaluation is None:
        evaluation = next((path for path in (checkpoint.parent / "evaluation_summary.json",
                                             checkpoint.parent / "evaluation.json") if path.exists()), None)
    reel = worst_cases(predictors[0], checkpoint, evaluation, worst) if evaluation and worst else None

    if input_path:
        equation = LinearODE(**json.loads(Path(input_path).read_text())["equation"])
    else:
        equation = LinearODE((2.0, 0.6), (1.0, 0.0))  # a gently damped spring, like try_equation.json
    start = page_equation(equation)
    if equation.order not in predictors[0].checkpoint["config"]["orders"]:
        raise ValueError(f"the model was not trained on order {equation.order}")

    # PyTorch's answers for the starting equation: the page recomputes them on load as a self-check.
    config = predictors[0].checkpoint["config"]
    times = np.linspace(0, config["t_stop"] - config["t_start"], config["time_points"])
    check = {"exact": linear_reference(equation, times).tolist(),
             "predicted": [predictor.predict(equation, times).tolist()
                           if equation.order in predictor.checkpoint["config"]["orders"] else None
                           for predictor in predictors]}
    data = {"models": [export_model(predictor, name) for predictor, name in zip(predictors, names)],
            "start": start, "check": check, "reel": reel,
            "built": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    # "</" cannot appear inside a <script> element; base64 and numbers never contain it, but names might.
    return PAGE.read_text(encoding="utf-8").replace("__DATA__", json.dumps(data).replace("</", "<\\/"))


def main(argv=None):
    """Write the explorer page; return 0."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Model folder or best.pt (default: %(default)s)")
    parser.add_argument("--compare", type=Path, help="Second model folder or best.pt for the compare scene")
    parser.add_argument("--input", type=Path, help="Equation JSON to start on, as for `predict`")
    parser.add_argument("--evaluation", type=Path, help="Evaluation JSON for the worst-case reel "
                        "(default: evaluation_summary.json or evaluation.json next to the model, if present)")
    parser.add_argument("--worst", type=int, default=8, help="Worst cases in the reel; 0 = no reel (default: %(default)s)")
    parser.add_argument("--output", type=Path, help="HTML file to write (default: explorer.html next to the model)")
    args = parser.parse_args(argv)
    if args.worst < 0:
        parser.error("--worst must be nonnegative")
    try:
        html = build(args.model, args.input, args.compare, args.evaluation, args.worst)
    except ValueError as error:
        parser.error(str(error))
    output = args.output or _checkpoint(args.model).parent / "explorer.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB): open it in any browser, no internet needed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
