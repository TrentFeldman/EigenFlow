"""Render a run's snapshots as an MP4 of the network learning one equation.

    python -m ode train ... --snapshot-every 5            # saves weights as it learns
    python -m ode video --run ode/runs/<name> --input ode/equations/try_equation.json

The 1080p video sweeps through the saved epochs (neuron grids plus the
network's answer drawn over the exact solution), then holds on the final epoch.
The same picture as `python -m ode neurons`, as a file you can send around.
Side panels show the validation error per epoch and the network's error on
this equation, magnified, so small fine-tuning gains stay visible. Use
--changes to colour each neuron by how far it has moved since the first
snapshot instead of by its value (best for fine-tunes, where values barely move),
or --changes recent for the movement since the previous snapshot, which fades
as the learning rate falls.
Needs Pillow and the ffmpeg program on PATH.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import torch

from ..evaluation.plot_results import _equation_text
from ..evaluation.predict import Predictor
from ..generation.analytic import linear_reference
from ..generation.equations import LinearODE
from ..training.dataset import encode
from ..training.model import OrderConditionedMLP
from .neurons import _activations

W, H, FPS = 1280, 720, 12  # layout units; frames are drawn SCALE times larger, then shrunk to OUTPUT
SCALE, OUTPUT = 3, (1920, 1080)  # supersampling smooths every edge and keeps small text sharp
BG, INK, MUTED, NEUTRAL, BLUE, ORANGE = ((251, 251, 249), (28, 28, 26), (107, 107, 102),
                                         (230, 230, 225), (42, 120, 214), (235, 104, 52))
FONT = "/usr/share/fonts/noto/NotoSans-{}.ttf"
TRAIL = 4  # earlier snapshots drawn as fading ghosts behind the current answer and error
CONTRAST = 2.0  # colour gain on top of each layer's scale; higher saturates more neurons


def _font(size, bold=False):
    """Noto Sans when installed, otherwise Pillow's built-in font."""
    from PIL import ImageFont
    try:
        return ImageFont.truetype(FONT.format("Bold" if bold else "Regular"), size * SCALE)
    except OSError:
        return ImageFont.load_default(size * SCALE)


class _Scaled:
    """ImageDraw stand-in taking layout coordinates and drawing SCALE times larger."""

    def __init__(self, draw):
        self.draw = draw

    @staticmethod
    def _xy(xy):
        return [value * SCALE for point in xy for value in (point if isinstance(point, tuple) else (point,))]

    def text(self, xy, text, **options):
        self.draw.text((xy[0] * SCALE, xy[1] * SCALE), text, **options)

    def line(self, xy, width=1, **options):
        self.draw.line(self._xy(xy), width=width * SCALE, joint="curve", **options)

    def rectangle(self, xy, **options):
        self.draw.rectangle(self._xy(xy), width=SCALE, **options) if "outline" in options \
            else self.draw.rectangle(self._xy(xy), **options)

    def ellipse(self, xy, **options):
        self.draw.ellipse(self._xy(xy), width=SCALE, **options)

    def polygon(self, xy, **options):
        self.draw.polygon(self._xy(xy), **options)


def _colour(value):
    """Grey near zero, orange for positive, blue for negative."""
    target = ORANGE if value >= 0 else BLUE
    strength = float(np.tanh(CONTRAST * abs(value))) ** 0.6  # the power brightens weak values most
    return tuple(int(n + (t - n) * strength) for n, t in zip(NEUTRAL, target))


def main(argv=None):
    """Write the MP4 for one run and one equation; return 0."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="Run folder trained with --snapshot-every")
    parser.add_argument("--input", type=Path, required=True, help="Equation JSON, as for `predict`")
    parser.add_argument("--output", type=Path, help="MP4 to write; default <run>/training.mp4")
    parser.add_argument("--title", default="Watching a neural network learn")
    parser.add_argument("--changes", nargs="?", const="start", choices=("start", "recent"),
                        help="Colour neurons by their change instead of their value: since the first "
                             "snapshot (start, the default) or since the previous one (recent)")
    args = parser.parse_args(argv)
    snapshots = sorted((args.run / "snapshots").glob("epoch-*.pt"))
    if len(snapshots) < 2:
        parser.error(f"{args.run}/snapshots needs at least 2 files; train with --snapshot-every N")
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is not on PATH; install it to encode the video")
    from PIL import Image, ImageDraw
    output = args.output or args.run / "training.mp4"

    equation = LinearODE(**json.loads(args.input.read_text())["equation"])
    # best.pt supplies the preprocessing, the same for every epoch.
    reference = Predictor(args.run / "best.pt")
    config = reference.checkpoint["config"]
    times = np.linspace(0, config["t_stop"] - config["t_start"], config["time_points"])
    features, mask = encode(equation, times, reference.model.max_order)
    inputs = torch.as_tensor(reference.normalizer.transform_features(features, mask), dtype=torch.float32)
    exact = linear_reference(equation, times)
    errors = {row["epoch"]: row["validation_relative_l2"]
              for row in json.loads((args.run / "run.json").read_text())["epochs"]}

    epochs, activations, predictions = [], [], []
    for path in snapshots:
        snapshot = torch.load(path, map_location="cpu", weights_only=True)
        model = OrderConditionedMLP(**snapshot["architecture"])
        model.load_state_dict(snapshot["state_dict"])
        model.eval()
        epochs.append(snapshot["epoch"])
        activations.append(_activations(model, inputs))
        with torch.inference_mode():
            predictions.append(reference.normalizer.inverse_target(model(inputs).numpy())[:, :equation.order])
    layers = np.stack(activations)  # (snapshots, layers, T, width)
    if args.changes == "start":
        layers = layers - layers[:1]
    elif args.changes == "recent":
        layers = layers - np.concatenate([layers[:1], layers[:-1]])
    scales = np.quantile(np.abs(layers), 0.95, axis=(0, 2, 3)) + 1e-9  # one colour scale per layer
    input_scale = np.abs(inputs.numpy()).max() or 1.0
    count, depth, steps, width = layers.shape
    side = int(round(width ** 0.5))
    peak = max(np.abs(exact).max(), max(np.abs(p).max() for p in predictions)) * 1.1
    # Fixed so the curve visibly shrinks: early epochs overflow it, the final error fills about a quarter.
    error_peak = 4 * np.abs(predictions[-1][:, 0] - exact[:, 0]).max() + 1e-12
    history = sorted(errors.items())
    low, high = np.log10(min(v for _, v in history)) - 0.1, np.log10(max(v for _, v in history)) + 0.1
    title_font, small_font, bold_font = _font(26, True), _font(15), _font(17, True)
    title, start = _equation_text(equation)

    def frame(s, t):
        image = Image.new("RGB", (W * SCALE, H * SCALE), BG)
        draw = _Scaled(ImageDraw.Draw(image))
        draw.text((32, 22), args.title, fill=INK, font=title_font)
        draw.text((32, 60), f"{title}   ({start})", fill=MUTED, font=small_font)
        error = errors.get(epochs[s])
        draw.text((W - 360, 24), f"epoch {epochs[s]}" + (f"   validation error {error:.2%}" if error is not None
                                                         else "   (before training)"), fill=INK, font=bold_font)
        cell, gap, top = 11, 12, 112
        draw.text((32, top - 22), "inputs", fill=MUTED, font=small_font)
        for k, value in enumerate(inputs.numpy()[t]):
            draw.rectangle([32, top + k * 12, 62, top + k * 12 + 10], fill=_colour(value / input_scale))
        for layer in range(depth):
            left = 92 + layer * (side * cell + gap)
            draw.text((left, top - 22), f"layer {layer + 1}" + (" (change)" if args.changes else ""),
                      fill=MUTED, font=small_font)
            row = layers[s, layer, t] / scales[layer]
            for k in range(width):
                x, y = left + (k % side) * cell, top + (k // side) * cell
                draw.rectangle([x, y, x + cell - 2, y + cell - 2], fill=_colour(row[k]))
        px, py, pw, ph = 92, 360, 720, 300
        draw.text((px, py - 26), "the answer: pale = exact, dark = network now (blue y, orange y')",
                  fill=MUTED, font=small_font)
        draw.line([px, py + ph / 2, px + pw, py + ph / 2], fill=NEUTRAL, width=1)
        x_of = lambda k: px + pw * times[k] / times[-1]
        y_of = lambda v: py + ph / 2 - v / peak * ph / 2
        for component, colour in ((0, BLUE), (1, ORANGE))[:equation.order]:
            pale = tuple(int(0.45 * a + 0.55 * b) for a, b in zip(colour, BG))
            draw.line([(x_of(k), y_of(exact[k, component])) for k in range(steps)], fill=pale, width=7)
            for back in range(TRAIL, 0, -1):  # where the network's answer was a few snapshots ago
                if s - back >= 0:
                    fade = tuple(int(c + (b - c) * (0.5 + 0.45 * back / TRAIL)) for c, b in zip(colour, BG))
                    draw.line([(x_of(k), y_of(predictions[s - back][k, component])) for k in range(steps)],
                              fill=fade, width=1)
            draw.line([(x_of(k), y_of(predictions[s][k, component])) for k in range(steps)], fill=colour, width=3)
        draw.line([x_of(t), py, x_of(t), py + ph], fill=MUTED, width=1)
        draw.text((px + pw / 2 - 60, py + ph + 14), f"time t = {times[t]:.2f}   (neurons shown at this moment)",
                  fill=MUTED, font=small_font)
        # Validation error after every epoch (log scale), with this frame's epoch marked.
        qx, qw, qh = 850, 390, 120
        draw.text((qx, py - 26), "validation error by epoch (log scale)", fill=MUTED, font=small_font)
        draw.rectangle([qx, py, qx + qw, py + qh], outline=NEUTRAL)
        last = max(e for e, _ in history)
        point = lambda e, v: (qx + qw * e / last, py + qh - qh * (np.log10(v) - low) / (high - low))
        draw.line([point(e, v) for e, v in history], fill=NEUTRAL, width=2)  # the road ahead, faint
        done = [point(e, v) for e, v in history if e <= epochs[s]]
        if len(done) > 1:
            draw.line(done, fill=ORANGE, width=3)  # drawn in as the video plays
        if epochs[s] in errors:
            x, y = point(epochs[s], errors[epochs[s]])
            draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=ORANGE, outline=INK)
            draw.text((min(x + 10, qx + qw - 70), max(y - 24, py + 2)), f"{errors[epochs[s]]:.2%}", fill=INK, font=bold_font)
        # This equation's error in y, magnified, on a fixed scale.
        ey, eh = py + qh + 50, ph - qh - 50
        draw.text((qx, ey - 24), f"network minus exact y, magnified {peak / error_peak:,.0f}x", fill=MUTED, font=small_font)
        draw.rectangle([qx, ey, qx + qw, ey + eh], outline=NEUTRAL)
        draw.line([qx, ey + eh / 2, qx + qw, ey + eh / 2], fill=NEUTRAL)
        clip = lambda v: ey + eh / 2 - np.clip(v / error_peak, -1, 1) * eh / 2
        curve = lambda i: [(qx + qw * times[k] / times[-1], clip(predictions[i][k, 0] - exact[k, 0])) for k in range(steps)]
        for back in range(TRAIL, 0, -1):  # fading ghosts of the previous snapshots
            if s - back >= 0:
                fade = tuple(int(c + (b - c) * (0.35 + 0.6 * back / TRAIL)) for c, b in zip(BLUE, BG))
                draw.line(curve(s - back), fill=fade, width=2)
        now = curve(s)
        draw.polygon([(qx, ey + eh / 2)] + now + [(qx + qw, ey + eh / 2)],
                     fill=tuple(int(0.18 * c + 0.82 * b) for c, b in zip(BLUE, BG)))
        draw.line(now, fill=BLUE, width=3)
        return image

    middle = steps // 2
    plan = [(s, middle) for s in range(count) for _ in range(3)]  # epoch sweep, 3 frames per snapshot
    plan += [(count - 1, middle)] * 3 * FPS  # hold the finished model for 3 s
    with tempfile.TemporaryDirectory() as folder:
        rendered = {}
        for index, key in enumerate(plan):
            if key not in rendered:  # repeated frames are drawn once and linked
                rendered[key] = Path(folder) / f"f{len(rendered):05d}.png"
                frame(*key).resize(OUTPUT, Image.Resampling.LANCZOS).save(rendered[key])
            (Path(folder) / f"seq{index:05d}.png").symlink_to(rendered[key])
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{folder}/seq%05d.png",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(output)], check=True)
    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB, {len(plan) / FPS:.0f} s, epochs {epochs[0]}..{epochs[-1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
