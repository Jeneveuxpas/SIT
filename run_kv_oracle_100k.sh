#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="sit-xl2-attnscaf-dinob12-s8-t100k-cons-none-smooth-none-repa-none"

bash ./run_oracle_experiment.sh \
  --config "configs/${EXP_NAME}.yaml" \
  --exp-name "$EXP_NAME" \
  --step 0100000 \
  --gpu "${GPU:-0,1}" \
  --num-gpus 2 \
  --eval-batch-size "${EVAL_BATCH_SIZE:-256}" \
  --reference-pairing correct \
  --vae ema \
  --inference-dtype fp32


