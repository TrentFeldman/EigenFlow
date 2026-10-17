"""Draw charts of a finished run that someone outside the project can read.

Writes standalone SVG files (open them in any browser) into <run>/plots/:

- summary.svg: how far off the predictions are, per equation type, and how
  many validation equations land in each error range.
- training.svg: validation error after every epoch, so you can see whether
  training was still improving when it stopped.
- example-*.svg: individual equations (the typical case of each type and the
  worst few), exact solution and prediction drawn on top of each other.

Needs only NumPy and the standard library. Inputs come from files the other
stages already wrote: best.pt, run.json and evaluation.json in the run folder.
Every chart has a plain-English title, a "how to read this" line, labelled
axes and hover tooltips. Light and dark themes follow the viewer's setting.
"""

import argparse
from hashlib import sha256
from html import escape
import json
from pathlib import Path

import numpy as np
import torch

from ..training.dataset import TrajectoryDataset
from ..generation.equations import LinearODE
from .predict import Predictor


# Validated categorical slots 1-2 (blue, orange) plus neutral chart ink, with
# separately chosen dark-theme steps. Plain class rules (no CSS variables) so
# image viewers, slide software and librsvg render the light theme correctly;
# browsers also apply the dark block. Text never uses the series colours.
LIGHT = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
         "grid": "#e1e0d9", "axis": "#c3c2b7", "exact": "#2a78d6", "model": "#eb6834", "bad": "#d03b3b"}
DARK = {**LIGHT, "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "grid": "#2c2c2a",
        "axis": "#383835", "exact": "#3987e5", "model": "#d95926"}


def _colour_rules(colours):
    """CSS rules: .<name>-fill / .<name>-line plus text, tick, grid and axis ink."""
    rules = [f".{name}-fill {{ fill: {value}; }} .{name}-line {{ stroke: {value}; }}" for name, value in colours.items()]
    rules += [f"text {{ fill: {colours['ink2']}; }} .title, .label {{ fill: {colours['ink']}; }}",
              f".tick {{ fill: {colours['muted']}; }} .grid {{ stroke: {colours['grid']}; }}",
              f".axis {{ stroke: {colours['axis']}; }}"]
    return "\n".join(rules)


STYLE = f"""<style>
text {{ font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; font-size: 12px; }}
.title {{ font-size: 18px; font-weight: 600; }}
.note {{ font-size: 13px; }}
.tick {{ font-size: 11px; font-variant-numeric: tabular-nums; }}
.label {{ font-weight: 600; }}
{_colour_rules(LIGHT)}
@media (prefers-color-scheme: dark) {{
{_colour_rules(DARK)}
}}
</style>"""

FAMILY_NAMES = {"linear_homogeneous": "no outside force", "linear_sinusoidal": "pushed by a sine wave"}
ORDINALS = {1: "1st", 2: "2nd", 3: "3rd"}


def _family_name(order, family):
    """'order3/linear_sinusoidal' pieces -> '3rd-order, pushed by a sine wave'."""
    return f"{ORDINALS.get(order, f'{order}th')}-order, {FAMILY_NAMES[family]}"


def _derivative_name(index):
    """0 -> 'y', 1 -> "y'", 2 -> "y''", 4 -> 'y⁽⁴⁾' (the notation in MATH_GUIDE.md)."""
    return "y" + ("′" * index if index < 4 else f"⁽{index}⁾")


def _equation_text(equation):
    """Write the ODE the way a textbook would, e.g. "y'' + 0.50 y' + 2.00 y = 0"."""
    terms = [_derivative_name(equation.order)]
    for index in reversed(range(equation.order)):
        value = equation.coefficients[index]
        terms.append(f"{'−' if value < 0 else '+'} {abs(value):.2f} {_derivative_name(index)}")
    right = (f"{equation.amplitude:.2f} sin({equation.frequency:.2f} t {'−' if equation.phase < 0 else '+'} "
             f"{abs(equation.phase):.2f})" if equation.amplitude else "0")
    start = ", ".join(f"{_derivative_name(k)}(0) = {value:.2f}" for k, value in enumerate(equation.initial_state))
    return f"{' '.join(terms)} = {right}", f"starting from {start}"


def _ticks(lower, upper, count=5):
    """Return about `count` round tick values covering [lower, upper]."""
    step = (upper - lower) / max(count - 1, 1) or 1.0
    magnitude = 10 ** np.floor(np.log10(step))
    step = min((m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= step), default=step)
    return np.arange(np.ceil(lower / step) * step, upper + step * 1e-9, step)


def _svg(width, height, body):
    """Wrap chart parts in a standalone SVG document with the shared style."""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img">\n{STYLE}\n'
            f'<rect width="100%" height="100%" class="surface-fill"/>\n' + "\n".join(body) + "\n</svg>\n")


def _panel(body, box, x_range, y_range, x_label, y_label, percent_log=False):
    """Draw gridlines, ticks and axis labels; return x(), y() pixel mappers.

    box is (left, top, width, height) in pixels; ranges are data bounds.
    percent_log draws y on a log scale with 1-2-5 percentage ticks.
    """
    left, top, width, height = box
    scale = np.log10 if percent_log else (lambda value: value)
    low, high = scale(y_range[0]), scale(y_range[1])

    def x(value):
        return left + (value - x_range[0]) / (x_range[1] - x_range[0]) * width

    def y(value):
        return top + height - (scale(value) - low) / (high - low) * height

    y_ticks = ([value for value in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100)
                if y_range[0] <= value <= y_range[1]] if percent_log else _ticks(*y_range))
    for value in y_ticks:
        label = f"{value:g}%" if percent_log else f"{value + 0.0:.3g}"
        body.append(f'<line class="grid" x1="{left}" x2="{left + width}" y1="{y(value):.1f}" y2="{y(value):.1f}"/>')
        body.append(f'<text class="tick" x="{left - 6}" y="{y(value) + 4:.1f}" text-anchor="end">{label}</text>')
    for value in _ticks(*x_range, 6):
        body.append(f'<text class="tick" x="{x(value):.1f}" y="{top + height + 16}" text-anchor="middle">{value:g}</text>')
    body.append(f'<line class="axis" x1="{left}" x2="{left + width}" y1="{top + height}" y2="{top + height}"/>')
    body.append(f'<text x="{left + width / 2}" y="{top + height + 34}" text-anchor="middle">{escape(x_label)}</text>')
    body.append(f'<text class="label" x="{left}" y="{top - 10}">{escape(y_label)}</text>')
    return x, y


def _polyline(xs, ys, color, dashed=False, tooltip=None):
    """One 2px data line; dashed only to tell the prediction apart without colour."""
    points = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys))
    dash = ' stroke-dasharray="7 4"' if dashed else ""
    title = f"<title>{escape(tooltip)}</title>" if tooltip else ""
    return (f'<polyline points="{points}" fill="none" class="{color}-line" stroke-width="2" '
            f'stroke-linejoin="round"{dash}>{title}</polyline>')


def trajectory_svg(times, reference, prediction, title, subtitle):
    """Return an SVG comparing exact and predicted states for one equation.

    times (T,) must increase; reference and prediction are finite (T, n) arrays
    in derivative order [y, y', ...]. Each derivative gets two side-by-side
    panels: both curves overlaid, and the gap (prediction minus exact) on its
    own scale so small misses stay visible. Invalid arrays raise ValueError.
    """
    times, reference, prediction = map(np.asarray, (times, reference, prediction))
    if (times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0)
            or reference.ndim != 2 or reference.shape != prediction.shape
            or reference.shape[0] != len(times)
            or not all(np.isfinite(array).all() for array in (times, reference, prediction))):
        raise ValueError('Need increasing finite times and matching finite (T,n) states')
    order, row_height = reference.shape[1], 250
    height = 150 + row_height * order
    body = [f'<text class="title" x="32" y="34">{escape(title)}</text>']
    body += [f'<text class="note" x="32" y="{58 + 19 * i}">{escape(line)}</text>' for i, line in enumerate(subtitle)]
    legend_y = 58 + 19 * len(subtitle) + 8
    body += [f'<line x1="32" x2="60" y1="{legend_y}" y2="{legend_y}" class="exact-line" stroke-width="2"/>',
             f'<text x="66" y="{legend_y + 4}">exact answer (numerical solver)</text>',
             f'<line x1="290" x2="318" y1="{legend_y}" y2="{legend_y}" class="model-line" '
             f'stroke-width="2" stroke-dasharray="7 4"/>',
             f'<text x="324" y="{legend_y + 4}">neural-network prediction</text>']
    meanings = ["the solution", "its rate of change", "rate of change of y′"]
    for k in range(order):
        top = legend_y + 50 + k * row_height
        name = _derivative_name(k) + (f" ({meanings[k]})" if k < len(meanings) else "")
        exact, model = reference[:, k], prediction[:, k]
        gap = model - exact
        low, high = float(min(exact.min(), model.min())), float(max(exact.max(), model.max()))
        pad = max((high - low) * 0.08, 1e-6)
        x, y = _panel(body, (80, top, 470, 170), (times[0], times[-1]), (low - pad, high + pad), "time", name)
        body.append(_polyline(x(times), y(exact), "exact", tooltip=f"exact {name}"))
        body.append(_polyline(x(times), y(model), "model", dashed=True, tooltip=f"predicted {name}"))
        limit = max(float(np.abs(gap).max()) * 1.15, 1e-6)
        worst = int(np.argmax(np.abs(gap)))
        x, y = _panel(body, (660, top, 410, 170), (times[0], times[-1]), (-limit, limit), "time",
                      f"how far off: predicted − exact {_derivative_name(k)}")
        body.append(f'<line class="axis" x1="660" x2="1070" y1="{y(0):.1f}" y2="{y(0):.1f}"/>')
        body.append(_polyline(x(times), y(gap), "model",
                              tooltip=f"largest miss {gap[worst]:+.3g} at time {times[worst]:.2f}"))
    return _svg(1110, height, body)


def training_svg(epochs):
    """Return an SVG of validation error (%) after each epoch, on a log scale.

    epochs is run.json's list of per-epoch dicts with 'epoch' and
    'validation_relative_l2' (a fraction). The best epoch is marked.
    """
    numbers = np.array([row["epoch"] for row in epochs], dtype=float)
    errors = np.array([row["validation_relative_l2"] for row in epochs]) * 100
    best = int(np.argmin(errors))
    body = ['<text class="title" x="32" y="34">How the model improved during training</text>',
            '<text class="note" x="32" y="58">Average prediction error on validation equations '
            '(held back from training) after each epoch — one full pass over the training data.</text>',
            '<text class="note" x="32" y="77">Lower is better. A line that is still falling at the right '
            'edge means more epochs would likely help.</text>',
            f'<text class="label" x="32" y="100">● Best: {errors[best]:.2f}% at epoch {numbers[best]:g} '
            '(those weights are saved as best.pt)</text>']
    x_range = (numbers[0], numbers[-1]) if len(numbers) > 1 else (numbers[0] - 1, numbers[0] + 1)
    x, y = _panel(body, (80, 140, 860, 280), x_range, (errors.min() * 0.8, errors.max() * 1.2),
                  "epoch", "validation error (log scale)", percent_log=True)
    body.append(_polyline(x(numbers), y(errors), "exact", tooltip="validation error after each epoch"))
    body.append(f'<circle cx="{x(numbers[best]):.1f}" cy="{y(errors[best]):.1f}" r="5" class="exact-fill surface-line" '
                f'stroke-width="2"><title>best epoch {numbers[best]:g}: {errors[best]:.2f}%</title></circle>')
    return _svg(980, 480, body)


def summary_svg(split_report, split_name):
    """Return an SVG with error per equation type and the error histogram.

    split_report is one split from evaluation.json: 'overall', 'by_family'
    (keys 'order<n>/<family>') and 'trajectories' with 'relative_l2' fractions.
    """
    overall = split_report["overall"]
    values = np.array([row["relative_l2"] for row in split_report["trajectories"]
                       if row["relative_l2"] is not None]) * 100
    threshold = 5.0
    within = float(np.mean(values < threshold) * 100)
    mean = "n/a" if overall["mean"] is None else f"{overall['mean']:.1%}"
    body = [f'<text class="title" x="32" y="34">On {overall["count"]:,} equations it never trained on, the '
            f'prediction is off by {mean} on average</text>',
            f'<text class="note" x="32" y="58">{within:.0f}% of them are within {threshold:g}% of the exact '
            f'answer. Error = size of the miss relative to the size of the solution ({escape(split_name)} split).</text>']
    # Left: average error per equation type, one colour (one series).
    families = sorted(split_report["by_family"].items())
    left, top, bar_width = 260, 120, 300
    largest = max(max(item["mean"] or 0 for _, item in families), 1e-9) * 100
    body.append('<text class="label" x="32" y="104">Average error by type of equation</text>')
    for i, (key, item) in enumerate(families):
        order, family = key.removeprefix("order").split("/")
        row_y = top + i * 54
        length = item["mean"] * 100 / largest * bar_width if item["mean"] is not None else 0
        body.append(f'<text x="{left - 10}" y="{row_y + 15}" text-anchor="end">'
                    f'{escape(_family_name(int(order), family))}</text>')
        body.append(f'<rect x="{left}" y="{row_y}" width="{max(length, 1):.1f}" height="20" rx="4" '
                    f'class="exact-fill"><title>{item["count"]} equations: average {item["mean"]:.2%}, '
                    f'typical {item["median"]:.2%}, 95% below {item["p95"]:.2%}, worst {item["worst"]:.2%}'
                    f'</title></rect>')
        body.append(f'<text class="label" x="{left + length + 8:.1f}" y="{row_y + 15}">{item["mean"]:.1%}'
                    f'<tspan class="tick" dx="8">95% under {item["p95"]:.1%}</tspan></text>')
    # Right: histogram of per-equation errors with the 5% line.
    edges = [0, 0.5, 1, 2, 3, 5, 10, 20, float("inf")]
    counts = [int(np.sum((values >= a) & (values < b))) for a, b in zip(edges, edges[1:])]
    hx, htop, hwidth, hheight = 700, 120, 420, 210
    body.append('<text class="label" x="700" y="104">How many equations fall in each error range</text>')
    slot = hwidth / len(counts)
    for i, count in enumerate(counts):
        bar_height = count / max(max(counts), 1) * hheight
        name = f"{edges[i]:g}–{edges[i + 1]:g}%" if edges[i + 1] != float("inf") else f"over {edges[i]:g}%"
        color = "bad" if edges[i] >= threshold else "exact"
        body.append(f'<rect x="{hx + i * slot + 1:.1f}" y="{htop + hheight - bar_height:.1f}" width="{slot - 2:.1f}" '
                    f'height="{max(bar_height, 0.5):.1f}" rx="2" class="{color}-fill">'
                    f'<title>{count} equations with error {name}</title></rect>')
        body.append(f'<text class="tick" x="{hx + (i + 0.5) * slot:.1f}" y="{htop + hheight + 16}" '
                    f'text-anchor="middle">{name.replace("over ", "&gt;")}</text>')
        body.append(f'<text x="{hx + (i + 0.5) * slot:.1f}" y="{htop + hheight - bar_height - 6:.1f}" '
                    f'text-anchor="middle">{count}</text>')
    body.append(f'<line class="axis" x1="{hx}" x2="{hx + hwidth}" y1="{htop + hheight}" y2="{htop + hheight}"/>')
    body.append(f'<text class="note" x="{hx}" y="{htop + hheight + 40}">Red bars: off by {threshold:g}% or more '
                f'({100 - within:.0f}% of equations). Fewer is better.</text>')
    return _svg(1160, 400, body)


def _select_examples(rows, worst):
    """Pick the worst `worst` trajectories and the median one of each type."""
    ranked = sorted(rows, key=lambda row: float('inf') if row['relative_l2'] is None else row['relative_l2'])
    selected = {row['trajectory_id']: f'worst case #{rank} of {worst}'
                for rank, row in enumerate(reversed(ranked[-worst:]), 1)}
    for order, family in sorted({(row['order'], row['family']) for row in rows}):
        group = [row for row in ranked if row['order'] == order and row['family'] == family]
        selected.setdefault(group[len(group) // 2]['trajectory_id'], 'typical case')
    return [(row, selected[row['trajectory_id']]) for row in ranked if row['trajectory_id'] in selected]


def main(argv=None):
    """Write summary, training-progress and example charts for one run folder."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True, help='Run folder with best.pt, run.json, evaluation.json')
    parser.add_argument('--data', type=Path, help="Dataset folder; defaults to the one recorded in run.json")
    parser.add_argument('--output', type=Path, help='New folder for the SVGs; defaults to <run>/plots')
    parser.add_argument('--split', default='validation')
    parser.add_argument('--worst', type=int, default=3)
    args = parser.parse_args(argv)
    if args.worst < 1:
        parser.error('--worst must be positive')
    checkpoint, output = args.run / 'best.pt', args.output or args.run / 'plots'
    run = json.loads((args.run / 'run.json').read_text())
    report = json.loads((args.run / 'evaluation.json').read_text())
    data = args.data or Path(run['data'])
    if (report['checkpoint_sha256'] != sha256(checkpoint.read_bytes()).hexdigest()
            or report['dataset_manifest_sha256'] != sha256((data / 'manifest.json').read_bytes()).hexdigest()):
        raise ValueError('Plot checkpoint/dataset does not match evaluation')
    torch.set_num_threads(4)
    predictor = Predictor(checkpoint)
    dataset = TrajectoryDataset(data, args.split, predictor.model.max_order)
    split_report = report['splits'][args.split]
    output.mkdir(parents=True, exist_ok=False)
    (output / 'summary.svg').write_text(summary_svg(split_report, args.split))
    (output / 'training.svg').write_text(training_svg(run['epochs']))
    index = {record['trajectory_id']: i for i, record in enumerate(dataset.records)}
    average = split_report['overall']['mean'] or float('nan')
    artifacts = [{'path': 'summary.svg', 'selection': 'error summary'},
                 {'path': 'training.svg', 'selection': 'training progress'}]
    for row, selection in _select_examples(split_report['trajectories'], args.worst):
        identity = row['trajectory_id']
        if row['nonfinite']:
            artifacts.append({'trajectory_id': identity, 'selection': selection,
                              'status': 'nonfinite prediction; cannot draw a finite curve'})
            continue
        record = dataset.records[index[identity]]
        features, targets, _ = dataset.raw_trajectory(index[identity])
        times = features[:, -1]
        equation = LinearODE(**record['equation'])
        prediction = predictor.predict(equation, times)
        title = (f"{selection.capitalize()}: {_family_name(equation.order, equation.family)} — "
                 f"off by {row['relative_l2']:.1%}")
        subtitle = [*_equation_text(equation),
                    f"For comparison, the average error across all {split_report['overall']['count']:,} "
                    f"validation equations is {average:.1%}. (Case {identity}.)"]
        filename = f"example-{selection.split(' of')[0].replace(' #', '-').replace(' ', '-')}-{identity}.svg"
        (output / filename).write_text(trajectory_svg(times, targets[:, :equation.order], prediction, title, subtitle))
        artifacts.append({'trajectory_id': identity, 'selection': selection, 'path': filename})
    (output / 'plots.json').write_text(json.dumps(artifacts, indent=2) + '\n')
    print(f"wrote {len(artifacts)} charts to {output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
