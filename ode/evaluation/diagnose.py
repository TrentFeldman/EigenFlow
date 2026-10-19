"""Diagnose validation errors, initial conditions and derivative consistency.

This developer-facing report helps decide which accuracy experiment is useful.
It measures physical nondimensional states and time derivatives, never fits
preprocessing, and deliberately excludes the spent test split from its CLI.
Residuals are diagnostics, not substitutes for reference trajectory error.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from ..config import provenance
from ..training.dataset import TrajectoryDataset, encode, feature_slices
from ..generation.equations import LinearODE
from .evaluate import METRIC, summarize, trajectory_metrics
from .predict import Predictor


DEVELOPMENT_SPLITS = ('train', 'validation', 'validation_parameter', 'validation_initial')


def state_and_time_derivative(model, normalizer, equation, elapsed_times, device='cpu'):
    """Return physical states and autograd time derivatives, each float64 (T,n).

    Inputs match independent prediction: equation order n <= padding M, finite
    nonnegative times (T,), and saved normalization. Model must be a pointwise
    differentiable map (as the current MLP is); cross-time layers invalidate the
    summed-output derivative below. The derivative includes input-time and
    output-state scale factors. Computation uses float32 model precision.
    Invalid inputs or nonfinite output/gradients raise; no cases are discarded.
    """
    features, mask = encode(equation, elapsed_times, model.max_order)
    if not len(features):
        raise ValueError('Need at least one requested time')
    normalized = normalizer.transform_features(features, mask)
    time_column = feature_slices(model.max_order)['time'].start
    with torch.enable_grad():
        times = torch.tensor(np.asarray(elapsed_times), dtype=torch.float32,
                             device=device, requires_grad=True)
        encoded = torch.as_tensor(normalized, dtype=torch.float32, device=device)
        # Preserve the inference path's input rounding. This zero-valued term
        # supplies d(standardized time)/d(physical time) = 1/time_scale.
        correction = (times - times.detach()) / normalizer.input_scale[time_column]
        encoded = torch.cat((encoded[:, :time_column],
                             encoded[:, time_column:time_column+1] + correction[:, None],
                             encoded[:, time_column+1:]), dim=1)
        outputs = model(encoded)
        state_scale = torch.as_tensor(normalizer.target_scale, dtype=torch.float32, device=device)
        physical = outputs * state_scale
        derivatives = torch.stack([
            torch.autograd.grad(physical[:, component].sum(), times,
                                retain_graph=component < equation.order-1)[0]
            for component in range(equation.order)
        ], dim=1).detach().cpu().numpy().astype(np.float64)
        states = normalizer.inverse_target(outputs.detach().cpu().numpy())[:, :equation.order]
    if not np.isfinite(states).all() or not np.isfinite(derivatives).all():
        raise RuntimeError('Nonfinite diagnostic prediction or derivative')
    return states, derivatives


def regime_tags(equation, reference_rms):
    """Return overlapping, predeclared regime labels from inputs/reference RMS.

    The complex root's imaginary part is the damped mode frequency, not an
    exact forced-resonance location. Gap <=0.25 is a descriptive slice only.
    All units are nondimensional, and no thresholds are fitted to test data.
    """
    roots = np.roots([1., *reversed(equation.coefficients)])
    mode_frequency = float(np.max(np.abs(roots.imag)))
    minimum_decay = float(np.min(-roots.real))
    tags = []
    if reference_rms < .1:
        tags.append('reference_rms_lt_0_1')
    if minimum_decay <= .25:
        tags.append('minimum_decay_le_0_25')
    if mode_frequency >= 1.5:
        tags.append('damped_mode_frequency_ge_1_5')
    if equation.amplitude:
        if equation.frequency >= 1.5:
            tags.append('forcing_frequency_ge_1_5')
        if abs(equation.frequency - mode_frequency) <= .25:
            tags.append('forcing_gap_to_damped_mode_le_0_25')
    return tags


def _scalar_summary(values):
    """Summarize a nonempty sequence of finite scalar nondimensional errors."""
    return {'mean': float(np.mean(values)), 'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)), 'worst': float(np.max(values))}


def diagnose(model, normalizer, dataset, device):
    """Return numerical diagnostics for all successful cases in a development split.

    Uses (T,M) padded references from dataset, slicing each order n. Time-profile
    full-state RMSE gives equal trajectory weights, with component counts kept
    separately. RHS residual is d(s_hat)/dt-rhs(t,s_hat); its first n-1 slots
    measure kinematic consistency and its last slot the highest-order dynamics.
    Raises if a trajectory cannot be diagnosed rather than hiding failures.
    """
    if dataset.split not in DEVELOPMENT_SPLITS:
        raise ValueError('Diagnostics are restricted to development splits')
    rows = []
    time_square = np.zeros(dataset.time_points)
    component_square = np.zeros((dataset.time_points, dataset.max_order))
    component_count = np.zeros(dataset.max_order, dtype=int)
    time_grid = None
    was_training = model.training
    model.eval()
    try:
        for index, record in enumerate(dataset.records):
            features, target, _ = dataset.raw_trajectory(index)
            time_grid = features[:, feature_slices(dataset.max_order)['time'].start]
            equation = LinearODE(**record['equation'])
            states, derivative = state_and_time_derivative(model, normalizer, equation, time_grid, device)
            reference = target[:, :equation.order]
            metrics = trajectory_metrics(reference, states)
            rhs = np.stack([equation.rhs(time, state) for time, state in zip(time_grid, states)])
            residual = derivative - rhs
            squared = (states-reference)**2
            time_square += squared.mean(axis=1)
            component_square[:, :equation.order] += squared
            component_count[:equation.order] += 1
            row = {'trajectory_id': record['trajectory_id'],
                   'family': f"order{equation.order}/{equation.family}",
                   **metrics,
                   'initial_state_rmse': float(np.sqrt(np.mean(squared[0]))),
                   'initial_state_max_absolute': float(np.max(np.abs(states[0]-reference[0]))),
                   'rhs_residual_rmse': float(np.sqrt(np.mean(residual**2))),
                   'dynamics_residual_rmse': float(np.sqrt(np.mean(residual[:, -1]**2))),
                   'kinematic_residual_rmse': float(np.sqrt(np.mean(residual[:, :-1]**2))) if equation.order > 1 else 0.,
                   'regime_tags': regime_tags(equation, metrics['reference_rms'])}
            rows.append(row)
    finally:
        model.train(was_training)
    if not rows:
        raise ValueError('No successful trajectories to diagnose')

    def aggregate(group):
        return {'trajectory_error': summarize(group),
                **{name: _scalar_summary([row[name] for row in group]) for name in (
                    'initial_state_rmse', 'initial_state_max_absolute', 'rhs_residual_rmse',
                    'dynamics_residual_rmse', 'kinematic_residual_rmse')}}

    return {'split': dataset.split, 'count': len(rows), 'test_evaluated': False,
            'overall': aggregate(rows),
            'by_family': {name: aggregate([row for row in rows if row['family']==name])
                          for name in sorted({row['family'] for row in rows})},
            'by_regime': {name: aggregate([row for row in rows if name in row['regime_tags']])
                          for name in sorted({tag for row in rows for tag in row['regime_tags']})},
            'time_profile': {
                'elapsed_times': time_grid.tolist(),
                'full_state_absolute_rmse': np.sqrt(time_square/len(rows)).tolist(),
                'component_absolute_rmse': {f'd{k}': np.sqrt(component_square[:, k]/count).tolist()
                                            for k, count in enumerate(component_count) if count}},
            'worst_trajectories': sorted(rows, key=lambda row: row['relative_l2'], reverse=True)[:10],
            'interpretation': 'Residuals quantify consistency of predicted functions, not guaranteed solution error; regime slices overlap.'}


def main(argv=None):
    """Write a new development diagnostic report tied to checkpoint/data hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=DEVELOPMENT_SPLITS, default='validation')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    predictor = Predictor(args.checkpoint, args.device)
    manifest_hash = sha256((args.data/'manifest.json').read_bytes()).hexdigest()
    if manifest_hash != predictor.checkpoint.get('dataset_manifest_sha256'):
        raise ValueError('Diagnostic dataset must match the training manifest')
    dataset = TrajectoryDataset(args.data, args.split, predictor.model.max_order, allow_failures=True)
    report = {'checkpoint_sha256': sha256(args.checkpoint.read_bytes()).hexdigest(),
              'dataset_manifest_sha256': manifest_hash, 'metric': METRIC,
              **provenance(('numpy', 'scipy', 'torch')),
              **diagnose(predictor.model, predictor.normalizer, dataset, predictor.device)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        output.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'count': report['count'],
                      'relative_l2': report['overall']['trajectory_error']['mean'],
                      'initial_state_rmse': report['overall']['initial_state_rmse']['mean'],
                      'rhs_residual_rmse': report['overall']['rhs_residual_rmse']['mean']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
