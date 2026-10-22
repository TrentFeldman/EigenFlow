"""Draw one chart of a finished run that someone outside the project can read.

Writes report.svg (open it in any browser) into the run folder:

- the headline: average error, and what half, 95% and the worst of the
  validation equations stay under;
- three validation equations (typical, 1-in-100 and worst) with the exact
  answer and the prediction drawn on top of each other;
- average error per equation type (rows: order, columns: forcing), so it
  works for any number of orders;
- validation error after every epoch, to show whether training had finished.

Needs only NumPy and the standard library, plus the files the other stages
already wrote: best.pt, run.json and evaluation.json. No dataset is needed:
the three example equations are rebuilt from their seeds and solved exactly.
Light and dark themes follow the viewer's setting; every mark has a tooltip.
"""

import argparse
from hashlib import sha256
from html import escape
import json
from pathlib import Path

import re

import numpy as np
import torch

from ..config import FAMILIES, SPLITS, GeneratorConfig
from ..generation.analytic import linear_reference
from ..generation.equations import LinearODE
from ..generation.generate_data import _group_jobs
from .evaluate import trajectory_metrics
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
    rules += [f"text {{ fill: {colours['ink2']}; }} .title, .label, .section, .hero, .stat {{ fill: {colours['ink']}; }}",
              f".tick {{ fill: {colours['muted']}; }} .grid {{ stroke: {colours['grid']}; }}",
              f".axis {{ stroke: {colours['axis']}; }}"]
    return "\n".join(rules)


STYLE = f"""<style>
text {{ font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; font-size: 13px; }}
.title {{ font-size: 18px; font-weight: 600; }}
.note {{ font-size: 13px; }}
.tick {{ font-size: 11px; font-variant-numeric: tabular-nums; }}
.label {{ font-weight: 600; }}
.section {{ font-size: 15px; font-weight: 600; }}
.hero {{ font-size: 52px; font-weight: 600; }}
.stat {{ font-size: 28px; font-weight: 600; }}
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
    ticks = np.arange(np.ceil(lower / step) * step, upper + step * 1e-9, step)
    return np.where(np.abs(ticks) < step * 1e-6, 0.0, ticks)  # no '-5.4e-20' for zero


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


def _percent(fraction):
    """0.0023 -> '0.23%', 0.076 -> '7.6%', 0.31 -> '31%': two significant-ish digits."""
    value = fraction * 100
    return f"{value:.0f}%" if value >= 10 else f"{value:.1f}%" if value >= 1 else f"{value:.2f}%"


def _families(by_family):
    """by_family keys 'order<n>/<family>' -> sorted orders, families and {(order, family): stats}."""
    cells = {}
    for key, item in by_family.items():
        order, family = key.removeprefix("order").split("/")
        cells[int(order), family] = item
    orders = sorted({order for order, _ in cells})
    families = [family for family in FAMILIES if any(f == family for _, f in cells)]
    return orders, families, cells


def report_svg(split_report, epochs, examples, title):
    """Return the one-page SVG report for a run.

    split_report: one split from evaluation.json ('overall', 'by_family',
    'trajectories' with 'relative_l2' fractions). epochs: run.json's list with
    'epoch' and 'validation_relative_l2'. examples: dicts with 'label',
    'error' (fraction), 'kind', 'equation', 'times', 'exact' and 'predicted'
    (the solution y only), plus optional 'note' lines shown under the plot. Any number of orders and forcing types fits.
    """
    overall = split_report["overall"]
    values = np.array([row["relative_l2"] for row in split_report["trajectories"]
                       if row["relative_l2"] is not None])
    orders, families, cells = _families(split_report["by_family"])
    order_names = [ORDINALS.get(order, f"{order}th") for order in orders]
    span = order_names[0] if len(orders) == 1 else f"{order_names[0]}- to {order_names[-1]}"
    over = int(np.sum(values >= 0.05))
    body = [f'<text class="title" x="32" y="38">{escape(title)}</text>',
            f'<text class="note" x="32" y="62">Tested on {overall["count"]:,} equations it never saw while '
            f'training: {span}-order linear ODEs, '
            f'{" or ".join(FAMILY_NAMES[f] for f in families)}.</text>']

    # Headline: one hero number, three supporting tiles.
    tiles = [("half of the equations are under", overall["median"], None),
             ("95% of the equations are under", overall["p95"], None),
             ("worst single equation", overall["worst"],
              f"{over:,} of {overall['count']:,} are off by 5% or more")]
    body += ['<text x="32" y="104">average error</text>',
             f'<text class="hero" x="32" y="156">{_percent(overall["mean"])}</text>']
    for i, (label, value, extra) in enumerate(tiles):
        x = 300 + i * 240
        body += [f'<text x="{x}" y="104">{label}</text>',
                 f'<text class="stat" x="{x}" y="140">{_percent(value)}</text>']
        if extra:
            body.append(f'<text class="tick" x="{x}" y="162">{extra}</text>')
    body.append('<text class="note" x="32" y="196">Error = how far the prediction is from the exact answer, '
                'compared with the size of the answer. 1% means the two curves differ by about '
                'a hundredth of their height.</text>')

    # Proof: three equations, exact vs predicted.
    body += ['<text class="section" x="32" y="246">What that looks like on three of the test equations</text>',
             '<line x1="720" x2="748" y1="242" y2="242" class="exact-line" stroke-width="2"/>',
             '<text x="754" y="246">exact answer (numerical solver)</text>',
             '<line x1="958" x2="986" y1="242" y2="242" class="model-line" stroke-width="2" stroke-dasharray="7 4"/>',
             '<text x="992" y="246">network prediction</text>']
    gap, width = 40, (1200 - 64 - 2 * 40) / 3
    for i, example in enumerate(examples):
        left = 32 + i * (width + gap)
        times, exact, predicted = example["times"], example["exact"], example["predicted"]
        low, high = float(min(exact.min(), predicted.min())), float(max(exact.max(), predicted.max()))
        pad = max((high - low) * 0.08, 1e-6)
        body += [f'<text class="label" x="{left}" y="282">{escape(example["label"])} — off by '
                 f'{_percent(example["error"])}</text>',
                 f'<text class="tick" x="{left}" y="300">{escape(example["kind"])}</text>']
        x, y = _panel(body, (left + 44, 332, width - 44, 150), (times[0], times[-1]), (low - pad, high + pad),
                      "time", "y, the solution")
        tooltip = example["equation"]
        body.append(_polyline(x(times), y(exact), "exact", tooltip=f"exact: {tooltip}"))
        body.append(_polyline(x(times), y(predicted), "model", dashed=True, tooltip=f"predicted: {tooltip}"))
        for k, line in enumerate(example.get("note", ())):
            body.append(f'<text class="tick" x="{left}" y="{536 + 15 * k}">{escape(line)}</text>')

    # Error per type: rows are orders, columns are forcing types.
    top, row = 630, 30
    body.append('<text class="section" x="32" y="596">Average error by type of equation</text>')
    column = (620 - 140) / len(families)
    largest = max(item["mean"] or 0 for item in cells.values()) or 1
    for j, family in enumerate(families):
        body.append(f'<text class="label" x="{140 + j * column}" y="{top - 8}">{FAMILY_NAMES[family]}</text>')
    for i, (order, name) in enumerate(zip(orders, order_names)):
        y_row = top + i * row
        body.append(f'<text x="32" y="{y_row + 16}">{name} order</text>')
        for j, family in enumerate(families):
            x0, item = 140 + j * column, cells.get((order, family))
            if item is None or item["mean"] is None:
                body.append(f'<text class="tick" x="{x0}" y="{y_row + 16}">not trained</text>')
                continue
            length = max(item["mean"] / largest * (column - 80), 2)
            body += [f'<rect x="{x0}" y="{y_row + 5}" width="{length:.1f}" height="14" rx="3" class="muted-fill">'
                     f'<title>{item["count"]:,} equations: average {_percent(item["mean"])}, half under '
                     f'{_percent(item["median"])}, 95% under {_percent(item["p95"])}, worst '
                     f'{_percent(item["worst"])}</title></rect>',
                     f'<text class="label" x="{x0 + length + 6:.1f}" y="{y_row + 16}">'
                     f'{_percent(item["mean"])}</text>']
    table_height = len(orders) * row

    # Training: validation error after every epoch.
    numbers = np.array([entry["epoch"] for entry in epochs], dtype=float)
    errors = np.array([entry["validation_relative_l2"] for entry in epochs]) * 100
    best = int(np.argmin(errors))
    plot_height = max(table_height, 150)
    body.append('<text class="section" x="700" y="596">How the error fell during training</text>')
    x_range = (numbers[0], numbers[-1]) if len(numbers) > 1 else (numbers[0] - 1, numbers[0] + 1)
    x, y = _panel(body, (744, top + 4, 416, plot_height), x_range, (errors.min() * 0.8, errors.max() * 1.2),
                  "epoch (one pass over the training data)", f"validation error, log scale — best "
                  f"{_percent(errors[best] / 100)} at epoch {numbers[best]:g}", percent_log=True)
    body.append(_polyline(x(numbers), y(errors), "ink2", tooltip="validation error after each epoch"))
    body.append(f'<circle cx="{x(numbers[best]):.1f}" cy="{y(errors[best]):.1f}" r="5" class="ink2-fill surface-line" '
                f'stroke-width="2"><title>best epoch {numbers[best]:g}: {errors[best]:.2f}%</title></circle>')
    bottom = top + 4 + plot_height + 34
    body.append(f'<text class="tick" x="744" y="{bottom + 20}">Still falling at the right edge = more epochs '
                f'would likely help.</text>')
    return _svg(1200, bottom + 40, body)


def _size_note(row, typical_size):
    """Explain a big percentage that comes from a tiny answer rather than a big miss."""
    ratio = typical_size / max(row['reference_rms'], 1e-12)
    return ((f"This answer is {ratio:,.0f}x smaller than a typical one,",
             "so even a tiny miss is a big percentage.") if ratio >= 10 else ())


def _equation_for(identity, config):
    """Rebuild a core-split equation from its trajectory id, e.g. 'validation-o2-linear_homogeneous-g5-i1'."""
    split, order, family, group, _ = re.fullmatch(r"(\w+)-o(\d+)-(\w+)-g(\d+)-i(\d+)", identity).groups()
    jobs = _group_jobs(config, SPLITS.index(split), int(order), family, int(group))
    return next(job.equation for job in jobs if job.trajectory_id == identity)


def main(argv=None):
    """Write report.svg for one evaluated run folder."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True, help='Run folder with best.pt, run.json, evaluation.json')
    parser.add_argument('--output', type=Path, help='New SVG file; defaults to <run>/report.svg')
    parser.add_argument('--title', help="Chart title; defaults to one naming the run folder")
    parser.add_argument('--split', default='validation', choices=['validation'])
    args = parser.parse_args(argv)
    checkpoint, output = args.run / 'best.pt', args.output or args.run / 'report.svg'
    run = json.loads((args.run / 'run.json').read_text())
    report = json.loads((args.run / 'evaluation.json').read_text())
    if report['checkpoint_sha256'] != sha256(checkpoint.read_bytes()).hexdigest():
        raise ValueError('best.pt does not match evaluation.json')
    torch.set_num_threads(4)
    predictor = Predictor(checkpoint)
    config = GeneratorConfig.from_dict(run['config'])
    times = np.linspace(0, config.t_stop - config.t_start, config.time_points)
    split_report = report['splits'][args.split]
    ranked = sorted((row for row in split_report['trajectories'] if row['relative_l2'] is not None),
                    key=lambda row: row['relative_l2'])
    picks = [("Typical equation", ranked[len(ranked) // 2]),
             ("1-in-100 equation", ranked[int(0.99 * (len(ranked) - 1))]),
             (f"Worst of {len(ranked):,}", ranked[-1])]
    typical_size = float(np.median([row['reference_rms'] for row in ranked]))
    examples = []
    for label, row in picks:
        equation = _equation_for(row['trajectory_id'], config)
        exact, predicted = linear_reference(equation, times), predictor.predict(equation, times)
        # The rebuilt equation must be the one that was scored, or the chart would lie.
        if not np.isclose(trajectory_metrics(exact, predicted)['relative_l2'], row['relative_l2'], rtol=1e-3):
            raise ValueError(f"Rebuilt {row['trajectory_id']} does not reproduce its evaluated error")
        examples.append({'label': label, 'error': row['relative_l2'],
                         'kind': _family_name(equation.order, equation.family), 'note': _size_note(row, typical_size),
                         'equation': "; ".join(_equation_text(equation)) + f" (case {row['trajectory_id']})",
                         'times': times, 'exact': exact[:, 0], 'predicted': predicted[:, 0]})
    title = args.title or f"{args.run.name}: how close the network gets to the exact answer"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        handle.write(report_svg(split_report, run['epochs'], examples, title))
    print(f"wrote {output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
