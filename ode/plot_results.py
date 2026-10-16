"""Export reference/prediction and signed-error plots as standalone SVG files.

Uses NumPy plus the standard library, so plotting needs no environment changes.
Each derivative has its own axes in nondimensional units. Selection uses a
saved evaluation report: median case per family and the worst cases overall.
"""

import argparse
from hashlib import sha256
from html import escape
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import TrajectoryDataset
from .equations import LinearODE
from .predict import Predictor


def trajectory_svg(times, reference, prediction, title, subtitle):
    """Return SVG for times (T,) and matching finite states (T,n), T>=2.

    Left panels overlay reference (blue) and prediction (orange dashed); right
    panels show signed prediction-reference error. Every panel has numeric
    ticks and its own scale. Invalid arrays raise ValueError.
    """
    times, reference, prediction = map(np.asarray, (times, reference, prediction))
    if (times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0)
            or reference.ndim != 2 or reference.shape != prediction.shape
            or reference.shape[0] != len(times)
            or not all(np.isfinite(array).all() for array in (times, reference, prediction))):
        raise ValueError('Need increasing finite times and matching finite (T,n) states')
    height = 125 + 240 * reference.shape[1]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" font-size="12" fill="#172033">',
             f'<text x="35" y="28" font-size="16">{escape(title)}</text>',
             f'<text x="35" y="51">{escape(subtitle)}</text>',
             '<text x="35" y="74" fill="#1464b4">Reference: blue</text>',
             '<text x="180" y="74" fill="#c75a0a">Prediction: orange dashed</text>',
             '<text x="570" y="74">Signed error: prediction - reference</text>']
    for component in range(reference.shape[1]):
        for panel in range(2):
            x0, y0, width, panel_height = 80 + panel * 535, 115 + component * 240, 425, 160
            curves = ([reference[:, component], prediction[:, component]] if panel == 0
                      else [prediction[:, component] - reference[:, component]])
            lower, upper = float(np.min(curves)), float(np.max(curves))
            margin = max((upper - lower) * .08, 1e-8)
            lower, upper = lower - margin, upper + margin
            def x(value):
                return x0 + (value - times[0]) / (times[-1] - times[0]) * width
            def y(value):
                return y0 + panel_height - (value - lower) / (upper - lower) * panel_height
            for value in np.linspace(lower, upper, 5):
                pixel = y(value)
                parts.append(f'<path d="M{x0},{pixel:.2f} h{width}" stroke="#dde2e9"/>')
                parts.append(f'<text x="{x0-8}" y="{pixel+4:.2f}" text-anchor="end">{value:.3g}</text>')
            for value in np.linspace(times[0], times[-1], 6):
                parts.append(f'<text x="{x(value):.2f}" y="{y0+panel_height+18}" text-anchor="middle">{value:.2g}</text>')
            parts.append(f'<text x="{x0}" y="{y0-14}">d{component}: {"state" if panel == 0 else "signed error"} (nondimensional)</text>')
            parts.append(f'<text x="{x0+width/2}" y="{y0+panel_height+39}" text-anchor="middle">Elapsed time (nondimensional)</text>')
            for curve_index, curve in enumerate(curves):
                points = ' '.join(f'{x(t):.2f},{y(value):.2f}' for t, value in zip(times, curve))
                color = '#1464b4' if panel == 0 and curve_index == 0 else '#c75a0a'
                dash = ' stroke-dasharray="6 4"' if panel == 0 and curve_index == 1 else ''
                parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.8"{dash}/>')
    parts.append('</g></svg>')
    return '\n'.join(parts) + '\n'


def main(argv=None):
    """Plot median-per-family and worst cases from an existing evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', default='validation')
    parser.add_argument('--worst', type=int, default=3)
    args = parser.parse_args(argv)
    if args.worst < 1:
        parser.error('--worst must be positive')
    report = json.loads(args.report.read_text())
    if (report['checkpoint_sha256'] != sha256(args.checkpoint.read_bytes()).hexdigest()
            or report['dataset_manifest_sha256'] != sha256((args.data / 'manifest.json').read_bytes()).hexdigest()):
        raise ValueError('Plot checkpoint/dataset does not match evaluation')
    torch.set_num_threads(4)
    predictor = Predictor(args.checkpoint)
    dataset = TrajectoryDataset(args.data, args.split, predictor.model.max_order)
    rows = report['splits'][args.split]['trajectories']
    ranked = sorted(rows, key=lambda row: float('inf') if row['relative_l2'] is None else row['relative_l2'])
    selected = {row['trajectory_id']: 'worst' for row in ranked[-args.worst:]}
    for order, family in sorted({(row['order'], row['family']) for row in rows}):
        group = [row for row in ranked if row['order'] == order and row['family'] == family]
        selected.setdefault(group[len(group)//2]['trajectory_id'], 'family median')
    args.output.mkdir(parents=True, exist_ok=False)
    index = {record['trajectory_id']: i for i, record in enumerate(dataset.records)}
    artifacts = []
    for row in ranked:
        identity = row['trajectory_id']
        if identity not in selected:
            continue
        if row['nonfinite']:
            artifacts.append({'trajectory_id': identity, 'selection': selected[identity],
                              'status': 'nonfinite prediction; cannot draw a finite curve'})
            continue
        position = index[identity]
        record = dataset.records[position]
        features, targets, _ = dataset.raw_trajectory(position)
        times = features[:, -1]
        equation = LinearODE(**record['equation'])
        prediction = predictor.predict(equation, times)
        subtitle = f"a={np.round(equation.coefficients, 3)}; IC={np.round(equation.initial_state, 3)}; A={equation.amplitude:.3g}; omega={equation.frequency:.3g}; phi={equation.phase:.3g}"
        title = f"{selected[identity]} | {identity} | relative L2={row['relative_l2']:.2%}"
        filename = f'{identity}.svg'
        (args.output / filename).write_text(trajectory_svg(times, targets[:, :equation.order], prediction, title, subtitle))
        artifacts.append({'trajectory_id': identity, 'selection': selected[identity], 'path': filename})
    (args.output / 'plots.json').write_text(json.dumps(artifacts, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
