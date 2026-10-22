# Arc

Linear, constant-coefficient ODEs of order 1-4 whose solutions decay while oscillating (underdamped),
unforced or sine-forced. 512 wide x 6 layers. SHA-256 `a7c8159a1e4dc8a0c825c9b634417255db22a2da0c10da2e2865b15422a6533f`.

Trained range: decay 0.01-1.5, frequency 0.5-4, starting values within +-1, forcing 0.1-1, times 0-5
(larger scales go through the rescaling wrapper). Not covered: overdamped or growing solutions.

## Error (validation, 39,999 equations; test split not scored)

Chart (open in a browser): [report.svg](report.svg).

| | Mean | p95 | p99 |
|---|---|---|---|
| **All** | **0.227%** | 0.509% | 0.946% |
| Order 1 (homogeneous / sine) | 0.137 / 0.191% | 0.282 / 0.367% | 1.554 / 0.573% |
| Order 2 | 0.120 / 0.276% | 0.228 / 0.577% | 0.379 / 0.974% |
| Order 3 | 0.187 / 0.325% | 0.357 / 0.649% | 0.610 / 1.017% |
| Order 4 | 0.243 / 0.333% | 0.529 / 0.692% | 0.999 / 1.322% |

19 equations at or above 5%; the worst (51%) are order-1 cases starting near zero, where a tiny absolute miss
is a large relative one. For orders 2-3 alone, `../current` is more accurate (0.149%).

## How it was made

1. From scratch on `configs/generator_orders4.json` (480k, equal per order), 350 epochs, lr 1e-3 cosine.
2. Fine-tuned 150 epochs on `configs/generator_orders4_biased.json` (train data per type: order 1 25k, 2-3 50k,
   4 75k), lr 1e-4 to 1e-6, `--loss relative --focus-hard`. See `ode/flagship.sh`.

`python -m ode predict --checkpoint ode/models/arc/best.pt --input ode/equations/order4_example.json`
