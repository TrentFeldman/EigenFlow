"""Animate how the network's neurons change while it trains.

    python -m ode train ... --snapshot-every 5            # saves weights as it learns
    python -m ode neurons --run ode/runs/<name> --input ode/equations/try_equation.json

Writes one self-contained HTML page (open it in a browser). Each square is one
neuron's reaction to the chosen equation at one moment; blue is negative,
orange positive, grey near zero. One slider moves through training epochs, the
other through time within the solution. A toy view for building intuition.
"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch

from ..evaluation.plot_results import _equation_text
from ..evaluation.predict import Predictor, domain_violations
from ..generation.analytic import linear_reference
from ..generation.equations import LinearODE
from ..training.dataset import encode
from ..training.model import OrderConditionedMLP

TIME_STRIDE = 2  # show every 2nd time point: halves the page size, looks the same


def _activations(model, inputs):
    """Return the post-activation values of every hidden layer, shape (layers, T, width)."""
    layers = []
    net = model.network
    hooks = [net[i].register_forward_hook(lambda _m, _i, out: layers.append(out.numpy()))
             for i in range(1, len(net) - 1, 2)]
    with torch.inference_mode():
        model(inputs)
    for hook in hooks:
        hook.remove()
    return np.stack(layers)


def main(argv=None):
    """Write the animation page for one run and one equation; return 0."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="Run folder trained with --snapshot-every")
    parser.add_argument("--input", type=Path, required=True, help="Equation JSON, as for `predict`")
    parser.add_argument("--output", type=Path, help="HTML file to write; default <run>/neurons.html")
    args = parser.parse_args(argv)
    snapshots = sorted((args.run / "snapshots").glob("epoch-*.pt"))
    if len(snapshots) < 2:
        parser.error(f"{args.run}/snapshots needs at least 2 files; train with --snapshot-every N")
    output = args.output or args.run / "neurons.html"
    equation = LinearODE(**json.loads(args.input.read_text())["equation"])
    # best.pt supplies the preprocessing (the same for every epoch) and the trained range.
    reference = Predictor(args.run / "best.pt")
    config = reference.checkpoint["config"]
    times = np.linspace(0, config["t_stop"] - config["t_start"], config["time_points"])
    violations = domain_violations(equation, times, config)
    if violations:
        parser.error("; ".join(violations) + "; pick an equation inside the trained range")

    features, mask = encode(equation, times, reference.model.max_order)
    inputs = torch.as_tensor(reference.normalizer.transform_features(features, mask), dtype=torch.float32)
    shown = slice(None, None, TIME_STRIDE)
    epochs, activations, predictions = [], [], []
    for path in snapshots:
        snapshot = torch.load(path, map_location="cpu", weights_only=True)
        model = OrderConditionedMLP(**snapshot["architecture"])
        model.load_state_dict(snapshot["state_dict"])
        model.eval()
        epochs.append(snapshot["epoch"])
        activations.append(_activations(model, inputs)[:, shown])
        with torch.inference_mode():
            standardized = model(inputs).numpy()
        predictions.append(reference.normalizer.inverse_target(standardized)[:, :equation.order])

    layers = np.stack(activations)  # (snapshots, layers, T', width)
    scales = np.quantile(np.abs(layers), 0.99, axis=(0, 2, 3)) + 1e-9  # one colour scale per layer
    quantized = np.clip(np.round(layers / scales[None, :, None, None] * 127 / 4), -127, 127).astype(np.int8)
    title, start = _equation_text(equation)
    data = {"title": title, "subtitle": start, "epochs": epochs, "order": equation.order,
            "times": times[shown].tolist(), "layers": layers.shape[1], "width": layers.shape[3],
            "scales": scales.tolist(), "inputs": np.round(inputs.numpy()[shown], 3).tolist(),
            "activations": base64.b64encode(quantized.tobytes()).decode(),
            "exact": np.round(linear_reference(equation, times)[shown], 5).tolist(),
            "predicted": [np.round(p[shown], 5).tolist() for p in predictions]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(PAGE.replace("__DATA__", json.dumps(data)), encoding="utf-8")
    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB, {len(epochs)} snapshots): open it in a browser")
    return 0


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Neurons learning</title>
<style>
:root{--bg:#fbfbf9;--ink:#1c1c1a;--muted:#6b6b66;--neutral:#e6e6e1;--blue:#2a78d6;--orange:#eb6834}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--ink:#ecece8;--muted:#9a9a93;--neutral:#2a2a27;--blue:#4a9af0;--orange:#f08558}}
body{margin:0;padding:16px;background:var(--bg);color:var(--ink);font:15px/1.4 system-ui,sans-serif;max-width:1130px;margin-inline:auto}
h1{font-size:20px;margin:0 0 2px}p{margin:2px 0;color:var(--muted)}
canvas{width:100%;display:block;margin-top:10px}
.row{display:flex;gap:12px;align-items:center;margin-top:8px}.row span{min-width:92px}
input[type=range]{flex:1}button,select{font:inherit;padding:4px 14px;border:1px solid var(--muted);background:none;color:var(--ink);border-radius:6px;cursor:pointer}
</style></head><body>
<h1>Watching a neural network learn</h1>
<p id="eq"></p><p id="start"></p>
<p>Each square is one neuron's reaction to this equation at one moment. Orange = positive, blue = negative, grey = quiet. The epoch slider is training time; the time slider is time inside the solution.</p>
<canvas id="c" width="1100" height="640"></canvas>
<div class="row"><span id="elabel"></span><input id="e" type="range" min="0" value="0"></div>
<div class="row"><span id="tlabel"></span><input id="t" type="range" min="0" value="0"></div>
<div class="row"><button id="play">Play</button><select id="mode"><option value="e">animate training</option><option value="t">animate time</option></select></div>
<script>
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const hex = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
const mix = (a, b, f) => a.map((v, i) => Math.round(v + (b[i] - v) * f));
const S = D.epochs.length, T = D.times.length, L = D.layers, W = D.width, cv = document.getElementById("c"), g = cv.getContext("2d");
const raw = atob(D.activations), A = new Int8Array(raw.length); for (let i = 0; i < raw.length; i++) A[i] = raw.charCodeAt(i);
const cellAt = (s, l, t, k) => A[((s * L + l) * T + t) * W + k] * 4 / 127;  // value in units of the layer's scale
const eS = document.getElementById("e"), tS = document.getElementById("t"); eS.max = S - 1; tS.max = T - 1;
document.getElementById("eq").textContent = D.title; document.getElementById("start").textContent = D.subtitle;
const inScale = Math.max(...D.inputs.flat().map(Math.abs)) || 1;
const colour = (f, neutral, pos, neg) => "rgb(" + mix(neutral, f >= 0 ? pos : neg, Math.tanh(Math.abs(f))).join(",") + ")";
const side = Math.round(Math.sqrt(W)), cell = 10, gap = 10, gridTop = 22;
function draw() {
  const s = +eS.value, i = +tS.value, neutral = hex(css("--neutral")), pos = hex(css("--orange")), neg = hex(css("--blue")), ink = css("--ink"), muted = css("--muted");
  g.clearRect(0, 0, 1100, 640); g.font = "12px system-ui"; g.fillStyle = muted; g.fillText("inputs", 4, 14);
  D.inputs[i].forEach((v, k) => { g.fillStyle = colour(v / inScale, neutral, pos, neg); g.fillRect(4, 22 + k * 12, 36, 10); });
  for (let l = 0; l < L; l++) {
    const x0 = 66 + l * (side * cell + gap); let on = 0;
    for (let k = 0; k < W; k++) { const f = cellAt(s, l, i, k); if (Math.tanh(Math.abs(f)) > 0.5) on++;
      g.fillStyle = colour(f, neutral, pos, neg); g.fillRect(x0 + (k % side) * cell, gridTop + Math.floor(k / side) * cell, cell - 1, cell - 1); }
    g.fillStyle = muted; g.fillText("layer " + (l + 1), x0, 14); g.fillText(Math.round(100 * on / W) + "% lit", x0, gridTop + side * cell + 14);
  }
  const px = 66, pw = 1000, py = 270, ph = 330, hi = Math.max(...D.exact.flat().concat(D.predicted.flat(2)).map(Math.abs)) * 1.1 || 1;
  g.strokeStyle = muted; g.globalAlpha = .4; g.beginPath(); g.moveTo(px, py + ph / 2); g.lineTo(px + pw, py + ph / 2); g.stroke(); g.globalAlpha = 1;
  g.fillStyle = muted; g.fillText("the answer: solid = exact, dashed = what the network says at this epoch (y, then y' = its rate of change)", px, py - 8);
  const X = k => px + pw * D.times[k] / D.times[T - 1], Y = v => py + ph / 2 - v / hi * ph / 2;
  for (let c = 0; c < Math.min(2, D.order); c++) for (const [src, dash, wide] of [[D.exact, [], 4], [D.predicted[s], [7, 4], 2]]) {
    g.setLineDash(dash); g.lineWidth = wide; g.globalAlpha = dash.length ? 1 : .35; g.strokeStyle = c ? css("--orange") : css("--blue"); g.beginPath();
    src.forEach((row, k) => k ? g.lineTo(X(k), Y(row[c])) : g.moveTo(X(k), Y(row[c]))); g.stroke(); }
  g.setLineDash([]); g.globalAlpha = 1; g.lineWidth = 1; g.strokeStyle = ink; g.beginPath(); g.moveTo(X(i), py); g.lineTo(X(i), py + ph); g.stroke();
  g.fillStyle = muted; g.fillText("time", px + pw / 2, py + ph + 18);
  document.getElementById("elabel").textContent = "epoch " + D.epochs[s]; document.getElementById("tlabel").textContent = "t = " + D.times[i].toFixed(2);
}
let playing = false, last = 0;
function tick(now) { if (!playing) return; if (now - last > 120) { last = now; const sl = document.getElementById("mode").value === "e" ? eS : tS; sl.value = (+sl.value + 1) % (+sl.max + 1); draw(); } requestAnimationFrame(tick); }
document.getElementById("play").onclick = e => { playing = !playing; e.target.textContent = playing ? "Pause" : "Play"; if (playing) requestAnimationFrame(tick); };
eS.oninput = tS.oninput = draw; matchMedia("(prefers-color-scheme:dark)").onchange = draw; draw();
</script></body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
