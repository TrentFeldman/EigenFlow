#!/usr/bin/env bash
# Orders 1-4 models on the old range (decay 0.01-1.5, frequency 0.5-4, times 0-5).
# Pick a step from the menu or pass it: bash ode/flagship.sh 1|2|3
#   1) generate the order-biased dataset   2) fine-tune Arc on it   3) both
# Edit this file only while it is NOT running: bash reads scripts as it goes.
set -euo pipefail
cd "$(dirname "$0")/.."
# Train split per type: order 1 25k, orders 2-3 50k, order 4 75k (150k for order 4 in total);
# validation and test stay balanced at 5k per type, so scores compare with Arc's.
CONFIG=ode/configs/generator_orders4_biased.json
D=ode/data/generator_orders4_biased-480000
ARC=ode/runs/20261018-arc-350/best.pt
NAME=20261019-arc-biased
RUN=ode/runs/$NAME

generate() { [ -f $D/manifest.json ] && echo "$D already exists" || \
  .venv/bin/python -m ode generate --generate --config $CONFIG --output $D; }
finetune() {
  .venv/bin/python -m ode pipeline --name $NAME --preset $CONFIG \
    --epochs 150 --batch-size 1024 --width 512 --depth 6 --learning-rate-schedule cosine \
    --learning-rate 0.0001 --minimum-learning-rate 0.000001 --initialize-from $ARC \
    --compile --loss relative --focus-hard --snapshot-every 1 --allow-failures
  # Training-split score for the overfitting gap (same metric as validation). Never touches test.
  .venv/bin/python -m ode evaluate --checkpoint $RUN/best.pt --data $D --splits train validation \
    --output $RUN/evaluation-train-vs-validation.json --device cuda || echo "(skipped)"
}

choice=${1:-}
if [ -z "$choice" ]; then
  echo "1) generate the order-biased dataset (480,000 equations, ~30 min; skipped if it exists)"
  echo "2) fine-tune Arc on it, 150 epochs at width 512 (~5 h; generates the data first if missing)"
  echo "3) 1 then 2"
  read -rp "Which step? " choice
fi
case $choice in
  1) generate ;;
  2|3) generate; finetune ;;
  *) echo "Pick 1, 2 or 3"; exit 1 ;;
esac
.venv/bin/python -m ode runs
